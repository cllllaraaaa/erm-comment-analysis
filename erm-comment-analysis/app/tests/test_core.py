"""
Unit tests for the pure (no-Streamlit, no-network) parts of the pipeline.

Run from the project root:  pytest
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from lib import citations, regex_labels, domain, intensity, matcher, dedupe, highlight, ocr, labeling  # noqa: E402


# ------------------------------------------------------------------ citations
def test_extract_default_whitelist():
    text = "This violates NEPA and the Clean Water Act; see 40 CFR 1502 and the DEIS."
    got = citations.extract(text)
    assert "NEPA (National Environmental Policy Act)" in got
    assert "CWA (Clean Water Act)" in got
    assert "DEIS (Draft EIS)" in got
    assert any(g.startswith("40 CFR") for g in got)


def test_extract_ignores_unknown_acronyms():
    assert citations.extract("I sent this from my IBM PC via HTTP.") == []


def test_extract_with_domain_config():
    cfg = citations.DomainConfig(
        whitelist={"FERC"},
        canon={"FERC": "FERC (Federal Energy Regulatory Commission)"},
        category={"FERC": "Agency"})
    got = citations.extract("FERC must reject this pipeline.", cfg)
    assert got == ["FERC (Federal Energy Regulatory Commission)"]
    assert citations.category(got[0], cfg) == "Agency"
    # and the config does NOT leak into calls that don't pass it
    assert citations.extract("FERC must reject this pipeline.") == []


def test_category_defaults():
    assert citations.category("NEPA (National Environmental Policy Act)") == "Statute"
    assert citations.category("Executive Order 13990") == "Executive order"


# ---------------------------------------------------------------- regex labels
def test_stance_regression_marad_2019_0093_0022():
    """Real failure case (MARAD-2019-0093-0022): an opposing comment that ends
    with the words 'Please support us and vote NO'. The old cross-check matched
    the literal word 'support' and agreed with a wrong LLM 'support' label, so
    the miss never reached the review queue. The pattern must read
    'against this project' as opposition FIRST."""
    text = ("We too are against this project in our rural community. "
            "We would like you to please not give the permit of this tank farm "
            "to be built in this area. Please support us and vote NO")
    assert regex_labels.stance(text) == "oppose"


def test_stance_regression_marad_2019_0093_0021():
    """Second real miss: 'I am speaking against the proposed site location'.
    Neither 'oppose' nor 'against this project' appears literally, so the old
    pattern fell through to the word 'support' elsewhere in the text."""
    text = ("Along with other residents of Jones Creek I am speaking against "
            "the proposed site location of the tank farm project. "
            "Please support our community.")
    assert regex_labels.stance(text) == "oppose"


def test_stance_regex():
    assert regex_labels.stance("I strongly oppose this project") == "oppose"
    # MARAD-2019-0093-10010: 'greatly opposed' must read as opposition even
    # though the rest of the letter contains supportive-sounding legal language
    assert regex_labels.stance(
        "I am greatly opposed to the oil export terminal.") == "oppose"
    assert regex_labels.stance("Please approve the permit, I support it") == "support"
    # 'concerned' alone must NOT read as opposition (v10 fix)
    assert regex_labels.stance("I am concerned about the timeline") == "unclear"


def test_topics_set_custom_patterns():
    pats = {"noise": [r"\bnoise", r"\bdecibel"]}
    assert regex_labels.topics_set("The noise levels are terrible", pats) == {"noise"}
    assert regex_labels.topics_set("The noise levels are terrible", {}) == set()


# --------------------------------------------------------------------- domain
def test_pack_from_schema_text_plain_eis_lines():
    """Bare lines pasted from an EIS topic-area list become topics."""
    pack = domain.pack_from_schema_text("Air Quality\nNoise\nCultural Resources")
    codes = domain.topic_codes(pack)
    assert codes == ["air_quality", "noise", "cultural_resources"]
    assert domain.topic_labels(pack)["air_quality"] == "Air Quality"


def test_pack_from_schema_text_keeps_base_keywords():
    base = {"topics": [{"code": "noise", "label": "Noise",
                        "keywords": ["decibel", "loud"], "description": "old"}],
            "acronyms": {"FAA": {"full": "Federal Aviation Administration",
                                 "category": "Agency"}}}
    pack = domain.pack_from_schema_text("noise: new description", base=base)
    t = pack["topics"][0]
    assert t["keywords"] == ["decibel", "loud"]
    assert t["description"] == "new description"
    assert "FAA" in pack["acronyms"]


def test_normalise_coerces_shapes():
    pack = domain.normalise({"topics": [
        {"code": "Air Quality!!", "keywords": "smog, ozone"},
        "not-a-dict",
    ], "acronyms": {"ferc": "Federal Energy Regulatory Commission"}})
    assert domain.topic_codes(pack) == ["air_quality"]
    assert pack["topics"][0]["keywords"] == ["smog", "ozone"]
    assert pack["acronyms"]["FERC"]["category"] == "Other"


def test_default_pack_matches_offline_schema():
    pack = domain.default_pack()
    assert len(pack["topics"]) == 8
    assert domain.has_keyword_coverage(pack, domain.topic_codes(pack))
    cfg = domain.citation_config(pack)
    assert "DWPA" in cfg.whitelist


# -------------------------------------------------------------------- matcher
def test_matcher_finds_topics():
    m = matcher.get({"air_quality": ["ozone", "air pollution"],
                     "noise": ["decibel"]})
    assert m.topics_in("The Ozone levels and AIR POLLUTION here are bad") == {"air_quality"}
    assert m.topics_in("no keywords here") == set()


def test_matcher_word_boundary():
    m = matcher.get({"ports": ["port"]})
    # 'port' must not fire inside 'important' or 'support'
    assert m.topics_in("This is important, I support it") == set()
    assert m.topics_in("Build the port elsewhere") == {"ports"}


# ------------------------------------------------------------------ intensity
def test_intensity_labels():
    calm = intensity.score("I think this is fine.")
    angry = intensity.score("ABSOLUTELY NOT!!! I DEMAND you REJECT this "
                            "catastrophic, devastating project NOW!!!")
    assert angry > calm
    assert intensity.label(angry) in ("strong", "moderate")
    assert intensity.label(calm) == "mild"


# -------------------------------------------------------------------- dedupe
def test_near_dup_groups_template_variants():
    base = ("I am writing to oppose the proposed tank farm in Jones Creek. "
            "The risk of oil spills, the damage to wetlands and the impact on "
            "our fishing industry make this project unacceptable for our "
            "community and for the Texas coast as a whole. " * 3)
    a = "Dear Sir, " + base
    b = "To whom it may concern, " + base + " Sincerely, a resident of Freeport."
    c = ("The council should extend the library opening hours because many "
         "students have nowhere quiet to study in the evenings. " * 6)
    groups = dedupe.near_dup_groups([a, b, c])
    assert groups[0] == groups[1]          # same template, small edits
    assert groups[2] != groups[0]          # different letter stays apart


def test_near_dup_groups_short_texts_untouched():
    groups = dedupe.near_dup_groups(["I object.", "I object.", "No."])
    assert len(set(groups)) == 3           # too short to fingerprint: never merged


# ----------------------------------------------------------------- highlight
def test_highlighter_marks_all_three_kinds():
    hl = highlight.Highlighter(keyword_map={"spills": ["oil spill"]})
    out = hl.render("I oppose this project: an oil spill would violate NEPA.")
    assert "background:#F2DCCD" in out          # 'oppose' stance wording
    assert ">oil spill</span>" in out           # topic keyword
    assert ">NEPA</span>" in out                # cited document
    assert "<script" not in out


def test_highlighter_escapes_html():
    hl = highlight.Highlighter(keyword_map={"x": ["ozone"]})
    out = hl.render("<b>ozone</b> & <script>alert(1)</script>")
    assert "<b>" not in out and "&lt;b&gt;" in out
    assert "<script>" not in out
    assert ">ozone</span>" in out


# ----------------------------------------------------------------------- ocr
def test_all_urls_splits_separators_and_filters():
    cell = "https://a.gov/1.pdf | https://a.gov/2.jpg, not-a-url , https://a.gov/3"
    assert ocr.all_urls(cell) == ["https://a.gov/1.pdf", "https://a.gov/2.jpg",
                                  "https://a.gov/3"]
    assert ocr.all_urls(None) == []
    assert ocr.first_url(cell) == "https://a.gov/1.pdf"


def test_sniff_mime_by_signature():
    assert ocr.sniff_mime(b"%PDF-1.7 rest") == "application/pdf"
    assert ocr.sniff_mime(b"\x89PNG\r\n\x1a\nxxxx") == "image/png"
    assert ocr.sniff_mime(b"\xff\xd8\xff\xe0JFIF") == "image/jpeg"
    assert ocr.sniff_mime(b"II*\x00tiffdata") == "image/tiff"
    assert ocr.sniff_mime(b"PK\x03\x04 a docx") is None   # zip/docx unsupported


# --------------------------------------------------------------- batch parse
def test_parse_batch_valid_and_garbage():
    raw = ('{"results": [{"i": 0, "topics": ["noise", "bogus"], "stance": "oppose"},'
           ' {"i": 1, "topics": [], "stance": "support"},'
           ' {"i": 9, "topics": [], "stance": "oppose"}, "junk"]}')
    got = labeling._parse_batch(raw, 2, {"noise"})
    assert set(got) == {0, 1}
    assert got[0]["topics"] == ["noise"]          # invalid code filtered out
    assert got[0]["failed"] is False
    assert labeling._parse_batch("not json at all", 2, set()) is None
    assert labeling._parse_batch('{"results": "nope"}', 2, set()) is None


# ---------------------------------------------------------- letter families
def test_families_groups_personalised_template_variants():
    template = ("I strongly oppose the proposed GulfLink deepwater port project "
                "because it threatens our coastal communities, worsens climate "
                "change, endangers marine wildlife and is not in the national "
                "interest. Please deny this application and protect the Gulf "
                "coast for future generations of residents and visitors alike.")
    variant_a = template + " Sincerely, a concerned resident of Galveston."
    variant_b = "Dear Administrator, " + template + " I have lived here 30 years."
    one_off = ("As a marine biologist with two decades of fieldwork in the bay, "
               "my concern is specific: the ballast discharge volumes in the "
               "application exceed the levels studied in the FEIS appendix and "
               "no monitoring plan is proposed for the affected oyster reefs.")
    uniq = [template, variant_a, variant_b, one_off]
    counts = {template: 500, variant_a: 1, variant_b: 1, one_off: 1}
    rep_of = labeling._families(uniq, counts=counts)
    # all three template variants share ONE representative: the most-signed one
    assert rep_of[template] == template
    assert rep_of[variant_a] == template
    assert rep_of[variant_b] == template
    # the one-of-a-kind letter keeps its own (individual LLM call)
    assert rep_of[one_off] == one_off
    # only two representatives -> two LLM units instead of four
    assert len(set(rep_of.values())) == 2


def test_families_short_or_single_texts_map_to_self():
    assert labeling._families(["short text"]) == {"short text": "short text"}
    rep_of = labeling._families(["too short to fingerprint", "also very short"])
    assert rep_of["too short to fingerprint"] == "too short to fingerprint"
    assert rep_of["also very short"] == "also very short"

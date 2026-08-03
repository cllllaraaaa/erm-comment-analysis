"""
Near-duplicate letter grouping.

Advocacy campaigns send template letters that signers lightly personalise: a
different salutation, one added sentence, their own town name. Exact matching
misses those variants, so the 'distinct letters' count runs high and the truly
unique letters (the ones consultants read closely) are diluted.

Method: word-shingle MinHash + LSH banding, then candidate pairs are verified
with real Jaccard similarity before groups are merged (union-find). Pure
numpy; deterministic (no randomness at runtime); no external dependencies.

Guidance on the threshold: 0.7 groups letters sharing roughly two-thirds of
their 5-word phrases, which in practice means the same template with small
personal edits. Raise it toward 0.9 to be stricter.
"""
from __future__ import annotations
import re
import zlib

import numpy as np

_WORDS = re.compile(r"[a-z']+")

_N_HASH = 64          # minhash permutations
_BANDS = 16           # LSH bands (x 4 rows each)
_ROWS = _N_HASH // _BANDS
_PRIME = (1 << 61) - 1

# deterministic permutation parameters
_rng = np.random.RandomState(1234)
_A = _rng.randint(1, _PRIME, size=_N_HASH, dtype=np.int64)
_B = _rng.randint(0, _PRIME, size=_N_HASH, dtype=np.int64)


def _shingles(text: str, k: int = 5) -> np.ndarray:
    words = _WORDS.findall(str(text or "").lower())
    if len(words) < k:
        return np.empty(0, dtype=np.int64)
    hs = {zlib.crc32(" ".join(words[i:i + k]).encode()) & 0xFFFFFFFF
          for i in range(len(words) - k + 1)}
    return np.fromiter(hs, dtype=np.int64, count=len(hs))


def _minhash(sh: np.ndarray) -> np.ndarray:
    # (n_hash,) signature; sh is (n_shingles,)
    v = (sh[None, :] * _A[:, None] + _B[:, None]) % _PRIME
    return v.min(axis=1)


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def near_dup_groups(texts, threshold: float = 0.7, min_shingles: int = 8):
    """Group indices of near-identical texts.

    Returns a list of ints, one per input text: texts sharing a group id are
    near-duplicates of one another. Texts too short to fingerprint reliably
    (fewer than `min_shingles` shingles) always keep their own group.
    """
    texts = list(texts)
    n = len(texts)
    shingle_sets = [_shingles(t) for t in texts]
    eligible = [i for i in range(n) if len(shingle_sets[i]) >= min_shingles]
    uf = _UF(n)
    if eligible:
        sigs = np.stack([_minhash(shingle_sets[i]) for i in eligible])
        # LSH: same band signature -> candidate pair
        buckets: dict = {}
        for row, i in enumerate(eligible):
            for b in range(_BANDS):
                key = (b, sigs[row, b * _ROWS:(b + 1) * _ROWS].tobytes())
                buckets.setdefault(key, []).append(i)
        seen = set()

        def _check(i, j):
            pair = (i, j) if i < j else (j, i)
            if pair in seen:
                return
            seen.add(pair)
            a, b = shingle_sets[pair[0]], shingle_sets[pair[1]]
            inter = np.intersect1d(a, b, assume_unique=True).size
            union = a.size + b.size - inter
            if union and inter / union >= threshold:
                uf.union(pair[0], pair[1])

        for members in buckets.values():
            if len(members) < 2:
                continue
            if len(members) <= 30:          # small bucket: verify all pairs
                for x in range(len(members)):
                    for y in range(x + 1, len(members)):
                        _check(members[x], members[y])
            else:                            # huge bucket: almost surely one big
                head = members[0]            # template; verify against the head
                for other in members[1:]:
                    _check(head, other)
    return [uf.find(i) for i in range(n)]

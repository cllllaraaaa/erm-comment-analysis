"""
One place for model / endpoint configuration.

Change the model here and every caller (labelling, domain-pack builder, OCR,
AI assistant) follows — it used to be hard-coded in four separate files.
"""
GEMINI_MODEL = "gemini-2.5-flash-lite"


def endpoint(model: str | None = None) -> str:
    return ("https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model or GEMINI_MODEL}:generateContent")

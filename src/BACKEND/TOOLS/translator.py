from transformers import pipeline

# Load the model once
_translator = pipeline("translation_en_to_fr", model="t5-small")

def translate_en_to_fr(text: str) -> str:
    if not text.strip():
        return "❌ No text provided."
    # returns list of dicts: [{"translation_text": "..."}]
    out = _translator(text, max_length=256, do_sample=False)[0]["translation_text"]
    return out

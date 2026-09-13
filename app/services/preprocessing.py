import re

def clean_text(text: str) -> str:
    """
    Cleans raw customer review text by removing excessive whitespace and normalizing characters.
    """
    if not text:
        return ""
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_into_sentences(text: str) -> list[str]:
    """
    Splits text into individual sentence clauses for aspect-level analysis.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    # Split on sentence delimiters or conjunctions like 'but', 'however'
    sentences = re.split(r'(?<=[.!?])\s+|,\s*(?=but|however|although|furthermore|yet)', cleaned, flags=re.IGNORECASE)
    return [s.strip() for s in sentences if len(s.strip()) > 3]

import re

_HTML_TAG = re.compile(r'<[^>]+>')
_WHITESPACE = re.compile(r'\s+')

# Coordinators that introduce a new opinion clause. The comma is optional:
# "great phone but the refund never came" has to split exactly like
# "great phone, but the refund never came" — the comma-only rule used before
# left contrastive sentences fused, so the positive half masked the complaint.
_CLAUSE_BREAK = re.compile(
    r'(?<=[.!?])\s+'
    r'|\s*[;]\s*'
    r'|,?\s+(?=(?:but|however|although|though|yet|whereas|furthermore|'
    r'moreover|additionally|also|plus)\b)'
    r'|\s+and\s+(?=\w)',
    re.IGNORECASE,
)

MIN_CLAUSE_LENGTH = 3


def clean_text(text: str) -> str:
    """
    Cleans raw customer review text by removing excessive whitespace and normalizing characters.
    """
    if not text:
        return ""
    # Strip HTML tags
    text = _HTML_TAG.sub(' ', text)
    # Replace multiple spaces/newlines with a single space
    return _WHITESPACE.sub(' ', text).strip()


def split_into_sentences(text: str) -> list[str]:
    """
    Splits text into individual opinion clauses for aspect-level analysis.

    Splits on sentence terminators, semicolons, contrastive coordinators
    (with or without a preceding comma) and the coordinating "and", because a
    single sentence routinely carries one opinion per conjunct:
    "delivery was late and the box was crushed" is two separate issues for
    two separate departments.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    parts = _CLAUSE_BREAK.split(cleaned)
    return [
        stripped
        for stripped in (p.strip(' ,;') for p in parts if p)
        if len(stripped) > MIN_CLAUSE_LENGTH
    ]

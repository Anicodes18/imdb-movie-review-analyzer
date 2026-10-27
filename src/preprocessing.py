import re
import html


def clean_review(text):
    """
    Clean movie review text by decoding HTML entities,
    removing HTML tags, normalizing whitespace, and lowercasing.
    """
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip().lower()

    return text
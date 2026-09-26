import re
import unicodedata

# Collapse any run of whitespace into one space.
WHITESPACE_RE = re.compile(r"\s+")

# Remove punctuation while preserving Unicode letters/numbers.
PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_unicode(text: str) -> str:
    """
    Unicode-safe canonical normalization.

    NFKC handles compatibility variants while preserving scripts
    such as Devanagari, Kannada, etc.
    """
    if not text:
        return ""

    return unicodedata.normalize("NFKC", text)


def normalize_basic(text: str) -> str:
    """
    Conservative normalization shared by names and addresses.

    We intentionally do NOT yet:
      - remove legal suffixes
      - expand Ltd / Pvt / Rd / St
      - sort tokens
      - transliterate to ASCII
      - remove numeric components

    Those choices should be evaluated from the data first.
    """
    if not text:
        return ""

    text = normalize_unicode(text)

    # Better than lower() for general Unicode text.
    text = text.casefold()

    # Treat these as equivalent textual forms.
    text = text.replace("&", " and ")

    # Convert punctuation into spaces rather than deleting it.
    #
    # Example:
    #   "abc-pvt" -> "abc pvt"
    # rather than:
    #   "abcpvt"
    text = PUNCT_RE.sub(" ", text)

    text = WHITESPACE_RE.sub(" ", text)

    return text.strip()


def normalize_name(text: str) -> str:
    return normalize_basic(text)


def normalize_address(text: str) -> str:
    return normalize_basic(text)
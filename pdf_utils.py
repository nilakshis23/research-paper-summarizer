"""
pdf_utils.py
------------
Handles reading a research paper PDF and turning it into plain text
that we can feed to the LLM.
"""

from pypdf import PdfReader

# Small local models (e.g. qwen2.5:3b) have a limited context window.
# We cap the text we send so the prompt + text stay comfortably within it.
# Roughly 4 chars ~= 1 token, so 60,000 chars ~= 15,000 tokens.
MAX_CHARS = 60_000


def extract_text_from_pdf(path: str) -> str:
    """Extract all text from a PDF file, page by page."""
    reader = PdfReader(path)

    pages_text = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages_text.append(text)

    full_text = "\n".join(pages_text)
    return full_text.strip()


def prepare_paper_text(path: str) -> tuple[str, bool]:
    """
    Extract and lightly clean text from a PDF, truncating if needed.

    Returns:
        (text, was_truncated)
    """
    raw_text = extract_text_from_pdf(path)

    # Collapse excessive blank lines that PDF extraction tends to leave behind.
    lines = [line.strip() for line in raw_text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)

    was_truncated = False
    if len(cleaned) > MAX_CHARS:
        cleaned = cleaned[:MAX_CHARS]
        was_truncated = True

    return cleaned, was_truncated

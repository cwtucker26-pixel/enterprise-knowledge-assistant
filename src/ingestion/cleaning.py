"""Light cleaning only: whitespace and PDF line-wrap artifacts. Never rewrites content."""
import re

SENTENCE_END = (".", ":", ";", "!", "?")
HEADING = re.compile(r"^\d+\.\s")  # numbered section headings like "2. Scope"
WRAP_WIDTH = 80  # PDF lines wrapped by the generator are ~100 chars; titles/headers are shorter


def clean_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def rejoin_pdf_lines(text: str) -> str:
    """PDF text comes out hard-wrapped mid-sentence. Join a long line that does not end a
    sentence with the line after it, unless that next line starts a new numbered section."""
    lines = text.split("\n")
    out = [lines[0]] if lines else []
    for line in lines[1:]:
        prev = out[-1]
        wrapped = len(prev) >= WRAP_WIDTH and not prev.endswith(SENTENCE_END)
        if wrapped and line and not HEADING.match(line):
            out[-1] = prev + " " + line
        else:
            out.append(line)
    return "\n".join(out)


def clean_pdf_text(text: str) -> str:
    return clean_whitespace(rejoin_pdf_lines(clean_whitespace(text)))


def clean_markdown_text(text: str) -> str:
    return clean_whitespace(text)

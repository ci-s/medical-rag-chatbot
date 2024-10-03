from pypdf import PdfReader
import re


header_pattern = r"Klinik\sund\sPoliklinik\sfür\sNeurologie"
footer_pattern_page = r"Seite\s\d+\svon\s\d+"
title_pattern = r"Handbuch\sVaskuläre\sNeurologie\s/\sStroke\sUnit"
version_pattern = r"Version\s\d{4}"

alt1_header_pattern = r"Technische\sUniversität\sMünchen"
alt2_header_pattern = r"Klinikum\srechts\sder\sIsar"

page_pattern = r"Seite\s\d+\s(v\s?on|von)\s\d+"

REMOVE_LIST = [
    header_pattern,
    footer_pattern_page,
    title_pattern,
    version_pattern,
    alt1_header_pattern,
    alt2_header_pattern,
    page_pattern,
]


def preprocess_content(content: str) -> str:
    for pattern in REMOVE_LIST:
        content = re.sub(pattern, " ", content)
    return content


def remove_string(content: str, string: str, case_sensitive=True) -> str:
    pattern = re.sub(r" ", r"\\s+", string)

    if not case_sensitive:
        return re.sub(pattern, " ", content, flags=re.IGNORECASE)

    return re.sub(pattern, " ", content)


def read_pdf(file_path, pages: list[int] = []) -> str:
    """
    Reads the content of a PDF file, excluding specified pages.

    Args:
        file_path (str): The path to the PDF file.
        exclude_pages (list[int], optional): The list of page numbers to include. Defaults to [] and reads all pages.

    Returns:
        str: The extracted content from the PDF file.
    """
    reader = PdfReader(file_path)
    doc = ""
    for i, page in enumerate(reader.pages):
        if pages and i + 1 not in pages:
            continue
        page_content = page.extract_text()

        page_content = preprocess_content(page_content)

        doc = doc + f"\nSeite {i+1}\n" + page_content
    return doc

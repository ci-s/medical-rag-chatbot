from pypdf import PdfReader
import re

from structures.page import PageInfo, Chunk

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

        doc = doc + page_content  # f"\nSeite {i+1}\n" +

    return doc.replace("4.3 Blutdrucktherapie", "")  # TODO: Temprorary fix for duplicate heading 4.3


def get_page_info(file_path, pages: list[int] = []) -> dict[int, PageInfo]:
    """
    TODO: Add docstring
    """
    reader = PdfReader(file_path)

    page_content_dict = {}
    for i, page in enumerate(reader.pages):
        if pages and i + 1 not in pages:
            continue
        raw_page_content = page.extract_text()

        if i + 1 == 24:  # TODO: Temprorary fix for duplicate heading 4.3
            raw_page_content.replace("4.3 Blutdrucktherapie", "")

        page_content = preprocess_content(raw_page_content)
        page_content_dict[i + 1] = PageInfo(
            page_number=i + 1, token_count=None, raw_content=raw_page_content, processed_content=page_content
        )

    return page_content_dict


# TODO: Move or modify
def chunk_pdf_with_page_metadata(file_path, chunk_size: int, pages: list[int] = []) -> list[dict]:
    """
    Reads the content of a PDF file, chunking it and preserving page number metadata.

    Args:
        file_path (str): The path to the PDF file.
        pages (list[int], optional): The list of page numbers to include. Defaults to [] (reads all pages).

    Returns:
        list[dict]: A list of chunks with associated text and page metadata.
    """
    reader = PdfReader(file_path)
    chunks = []
    current_chunk = ""
    start_page = None  # Start page of the current chunk
    current_page = None  # Keep track of the current page number

    for i, page in enumerate(reader.pages):
        if pages and i + 1 not in pages:
            continue

        page_content = page.extract_text()
        page_content = preprocess_content(page_content)  # Apply your custom preprocessing

        if not page_content:
            continue

        current_page = i + 1  # Update the current page number

        # If we haven't started a chunk yet, mark the start page
        if start_page is None:
            start_page = current_page

        # Add content from this page to the chunk, handling chunk sizes
        while page_content:
            remaining_space = chunk_size - len(current_chunk)

            # If the remaining content fits into the current chunk
            if len(page_content) <= remaining_space:
                current_chunk += page_content
                page_content = ""  # All content consumed from this page
            else:
                # Split the content and finalize the current chunk
                current_chunk += page_content[:remaining_space]
                page_content = page_content[remaining_space:]  # Remaining part of the content

                # Append the chunk with metadata (spanning pages, if necessary)
                chunks.append(Chunk(text=current_chunk, start_page=start_page, end_page=current_page))

                # Reset for a new chunk, and mark the start of the next one
                current_chunk = ""
                start_page = current_page

    # Handle any remaining text in the last chunk
    if current_chunk:
        chunks.append(Chunk(text=current_chunk, start_page=start_page, end_page=current_page))

    return chunks

from pypdf import PdfReader
import re
import time

from domain.document import Document, Page
from core.utils import levenshteinDistance, normalize_text, replace_abbreviations
from core.model import generate_response
from parsing import get_format_instructions, WhitespaceInjectionResponse, parse_with_retry

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


def calculate_num_changes(original, modified):
    normalized_original = normalize_text(original)
    normalized_modified = normalize_text(modified)

    distance = levenshteinDistance(normalized_original, normalized_modified)

    return distance


def inject_whitespace_w_llm(text: str) -> str:
    system_prompt = """
        You're a helpful AI assistant that has been asked to fix a block of text where some words are concatenated due to errors in PDF extraction. Your task is to identify where spaces are missing between words and add them where necessary. Please only adjust spacing—do not modify punctuation, capitalization, or any part of the text otherwise. Also there is one exception, do not modify headings and subheadings. Return the modified text with spaces added where needed in a json format as follows: {"processed_text": "whitespace injected text here"}. Say nothing else and don't change anything else.
    """
    user_prompt = """
     Text: {text}
    """
    user_prompt = user_prompt.format(text=text)
    response = generate_response(system_prompt, user_prompt, max_new_tokens=2048)
    parsed_response = parse_with_retry(WhitespaceInjectionResponse, response)
    return parsed_response.processed_text


def inject_whitespace(content: str, num_changes_threshold: int = 50) -> str:
    injected_content = inject_whitespace_w_llm(content)
    num_changes = calculate_num_changes(content, injected_content)
    print(f"Number of changes: {num_changes}")
    if num_changes > num_changes_threshold:
        print(f"Significant changes detected. Number of changes: {num_changes}")
        print("Original content: ", content)
        print("Modified content: ", injected_content)

    return injected_content


def preprocess_content(content: str, whitespace_injection: bool = False, is_replace_abbreviations: bool = False) -> str:
    for pattern in REMOVE_LIST:
        content = re.sub(pattern, " ", content)

    if whitespace_injection:
        content = inject_whitespace(content)

    if is_replace_abbreviations:
        return replace_abbreviations(content)
    else:
        return content, 0


def remove_string(content: str, string: str, case_sensitive=True) -> str:
    pattern = re.sub(r" ", r"\\s+", string)

    if not case_sensitive:
        return re.sub(pattern, " ", content, flags=re.IGNORECASE)

    return re.sub(pattern, " ", content)


def filter_document(document: Document, pages: list[int]) -> Document:
    document.pages = [page for page in document.pages if page.page_number in pages]
    return document


def load_document(file_path: str, pages: list[int] = []) -> Document:
    """
    TODO: Add docstring
    """
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        doc = Document(path=file_path)

        for i, page in enumerate(reader.pages):
            if pages and i + 1 not in pages:
                continue
            page_content = page.extract_text()

            if i + 1 == 24:  # TODO: Temprorary fix for duplicate heading 4.3
                page_content = page_content.replace("4.3 Blutdrucktherapie", "")

            doc.add_page(
                Page(page_number=i + 1, token_count=None, raw_content=page_content, processed_content=page_content)
            )
    elif file_path.endswith(".pkl"):
        doc = Document.load(file_path)
        if pages:
            doc = filter_document(doc, pages)
            print("Document pages are filtered. The number of pages: ", len(doc.pages))
            for page in pages:
                if page not in [page.page_number for page in doc.pages]:
                    raise ValueError(f"Page {page} not found in the document.")
        else:
            print("All pages are loaded. The number of pages: ", len(doc.pages))
    else:
        raise ValueError("Unsupported file format. Only PDF and pickle files are supported.")

    return doc


def process_document(
    document: Document, whitespace_injection: bool = False, is_replace_abbreviations: bool = True
) -> Document:
    total_replaced_abbrev_count = 0
    for page in document.pages:
        processed_content, replaced_count = preprocess_content(
            page.processed_content, whitespace_injection, is_replace_abbreviations
        )
        page.processed_content = processed_content
        total_replaced_abbrev_count += replaced_count

    if whitespace_injection:
        document.save(
            document.path.replace(".pdf", f"_processed_with_whitespace_{int(time.time())}.pkl")
        )  # TODO: artifacts folder
    print(f"Number of abbreviations replaced total: {total_replaced_abbrev_count}")
    return document


def get_document(
    file_path: str | Document, pages: list[int] = [], whitespace_injection=False, is_replace_abbreviations=False
) -> Document:
    document = load_document(file_path, pages=pages)
    return process_document(
        document, whitespace_injection=whitespace_injection, is_replace_abbreviations=is_replace_abbreviations
    )


def merge_document(documents: list[Document]) -> Document:
    merged_document = Document()

    for document in documents:
        merged_document.pages.extend(document.pages)

    page_numbers = [page.page_number for page in merged_document.pages]
    duplicates = [page for page in set(page_numbers) if page_numbers.count(page) > 1]

    if duplicates:
        raise ValueError(f"Duplicate page numbers found: {duplicates}")

    merged_document.pages.sort(key=lambda page: page.page_number)

    return merged_document

from pypdf import PdfReader
import re
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from structures.page import Document, Page
from core.utils import levenshteinDistance, normalize_text

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


abbreviations = pd.read_csv("../data/Abbreviations - Revised_by_clinicians.csv")
abbreviation_dict = dict(zip(abbreviations["Abbreviation"], abbreviations["Meaning"]))

llm = ChatOllama(model="llama3.1:8b-instruct-q4_0", temperature=0, format="json")
chain = llm | JsonOutputParser()  # TODO: move to init


def replace_abbreviations(text: str, abbreviation_dict: dict) -> str:
    global REPLACED_ABB_COUNT

    pattern = re.compile(r"\b(" + "|".join(re.escape(abbr) for abbr in abbreviation_dict.keys()) + r")\b")

    count = 0

    def replacer(match):
        nonlocal count
        count += 1
        return abbreviation_dict[match.group()]

    result_text = pattern.sub(replacer, text)

    print(f"Number of abbreviations replaced: {count}")
    REPLACED_ABB_COUNT += count
    print(f"Number of abbreviations replaced total: {count}")
    return result_text


def has_significant_changes(original, modified, threshold=20):
    normalized_original = normalize_text(original)
    normalized_modified = normalize_text(modified)

    distance = levenshteinDistance(normalized_original, normalized_modified)

    return distance


def inject_whitespace_w_llm(text: str):
    messages = [
        (
            "system",
            """I have a block of text where some words are concatenated due to errors in PDF extraction. Your task is to identify where spaces are missing between words and add them where necessary. Please only adjust spacing—do not modify punctuation, capitalization, or any part of the text otherwise. Return the modified text with spaces added where needed in a json format as follows: {"output": "modified text here"} Say nothing else and don't change anything else. Be specifically careful about not modifying headings and subheadings' numberings.""",
        ),
        ("human", text),
    ]

    return chain.invoke(messages)["output"]


def inject_whitespace(content: str, num_changes_threshold: int = 20) -> str:
    injected_content = inject_whitespace_w_llm(content)

    num_changes = has_significant_changes(content, injected_content)
    if num_changes > num_changes_threshold:
        # Create a logger and artifact saver to save the significantly changed content
        # significantly_changed[id] = {
        #     "num_changes": num_changes,
        #     "original": content,
        #     "modified": injected_content,
        # }
        return content
    else:
        return injected_content


def preprocess_content(content: str, whitespace_injection: bool = False, is_replace_abbreviations: bool = False) -> str:
    for pattern in REMOVE_LIST:
        content = re.sub(pattern, " ", content)

    if whitespace_injection:
        content = inject_whitespace(content)

    if is_replace_abbreviations:
        return replace_abbreviations(content, abbreviation_dict)
    else:
        return content


def remove_string(content: str, string: str, case_sensitive=True) -> str:
    pattern = re.sub(r" ", r"\\s+", string)

    if not case_sensitive:
        return re.sub(pattern, " ", content, flags=re.IGNORECASE)

    return re.sub(pattern, " ", content)


# Rewrite using get_document
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

        doc = doc + "\f" + page_content

    return doc.replace(
        "4.3 Blutdrucktherapie\nBei hypotonen", "\nBei hypotonen"
    )  # TODO: Temprorary fix for duplicate heading 4.3


def load_document(file_path, pages: list[int] = []) -> Document:
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

            doc.add_page(Page(page_number=i + 1, token_count=None, raw_content=page_content, processed_content=None))
    elif file_path.endswith(".pkl"):
        doc = Document.load(file_path)
    else:
        raise ValueError("Unsupported file format. Only PDF and pickle files are supported.")

    return doc


def process_document(
    document: Document, whitespace_injection: bool = True, is_replace_abbreviations: bool = True
) -> Document:
    global REPLACED_ABB_COUNT
    REPLACED_ABB_COUNT = 0
    for page in document.pages:
        page.processed_content = preprocess_content(page.raw_content, whitespace_injection, is_replace_abbreviations)

    if whitespace_injection:
        document.save(document.path.replace(".pdf", "_processed_with_whitespace.pkl"))  # TODO: artifacts folder
    print(f"Number of abbreviations replaced total: {REPLACED_ABB_COUNT}")
    return document


def get_document(
    file_path: str | Document, pages: list[int] = [], whitespace_injection=False, is_replace_abbreviations=False
) -> Document:
    document = load_document(file_path, pages=pages)
    return process_document(
        document, whitespace_injection=whitespace_injection, is_replace_abbreviations=is_replace_abbreviations
    )

import json
import re
import time
from typing import Literal
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_nomic import NomicEmbeddings

from thefuzz import fuzz
from core.model import generate_response
from core.utils import merge_document
from domain.document import Document, Chunk
from prompts import HEADINGS_PROMPT


def chunk_by_size(document: Document, pages: list[int], chunk_size: int = 512, overlap: int = 0) -> list[Chunk]:
    document_str = merge_document(document, pages)
    print(f"Chunking by size {chunk_size} with overlap {overlap}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    docs = text_splitter.create_documents([document_str])

    chunks = [Chunk(document.page_content, None, None) for document in docs]
    chunks, problem_count = match_chunks_with_pages(chunks, document, pages)
    if problem_count > 0:
        raise ValueError("Problems encountered during matchinkg chunks with pages")
    return chunks


def chunk_semantic(document: Document, pages: list[int]) -> list[Chunk]:
    document_str = merge_document(document, pages=pages)
    text_splitter = SemanticChunker(NomicEmbeddings(model="nomic-embed-text-v1.5"))
    chunks = text_splitter.split_text(document_str)  # or create_documents using my get_document
    chunks = [Chunk(chunk, None, None) for chunk in chunks]
    chunks, problem_count = match_chunks_with_pages(chunks, document, pages)
    if problem_count > 0:
        raise ValueError("Problems encountered during matchinkg chunks with pages")
    return chunks


def convert_output_to_json(output: str):
    """
    Process the LLM output string to extract and return a JSON dictionary.

    Args:
        output (str): The raw output from the LLM.

    Returns:
        dict: The parsed JSON dictionary.
    """
    try:
        # Clean up the string to remove unnecessary newlines and leading/trailing whitespace
        cleaned_output = output.strip()
        cleaned_output = cleaned_output.replace("\n", "")

        # Remove outer quotes if the entire content is wrapped in quotes
        if cleaned_output.startswith('"') and cleaned_output.endswith('"'):
            cleaned_output = cleaned_output[1:-1]

        cleaned_output = cleaned_output.strip()
        # Decode any escaped characters
        decoded_output = cleaned_output.encode("utf-8").decode("unicode_escape")

        # Convert the string into JSON
        json_dict = json.loads(decoded_output)

        return json_dict
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")


def get_headings(document: Document) -> list[str]:
    headings = []

    for page in document.pages:
        response = generate_response(HEADINGS_PROMPT + page.processed_content)
        print(response)
        if isinstance(response, list):
            heading_dict_list = response
        else:
            heading_dict_list = convert_output_to_json(response)
        headings.extend([d["number"] + ". " + d["heading"] for d in heading_dict_list])
        print("page is done")

    return headings


def build_heading_hierarchy(headings: list[str]) -> dict[str, str]:
    """
    Constructs a hierarchy of headings such that each heading includes its parent headings.
    Handles inconsistent separator formatting.
    """
    hierarchy = {}
    current_path = []

    for heading in headings:
        match = re.match(r"(\d+(?:\.\d+)*)\s*(.*)", heading)
        if not match:
            continue  # Skip invalid headings

        number, title = match.groups()
        levels = number.split(".")

        # Adjust the current path to match the heading level
        while len(current_path) > len(levels):
            current_path.pop()
        current_path = current_path[: len(levels) - 1]
        current_path.append(heading)

        hierarchy[number] = " ".join([f"{h}" for h in current_path])

    print("Heading hierarchy:\n", hierarchy)
    return hierarchy


def split_by_headings(doc: str, headings: list[str], hierarchy: dict[str, str]) -> dict:
    """
    Splits the document into sections by headings, including the full hierarchy.
    """
    sections = {}

    headings_pattern = "|".join(
        [re.escape(heading).replace(r"\ ", r"\s*").replace(r"\.", r"\.?\s*") for heading in headings]
    )
    split_content = re.split(f"({headings_pattern})", doc)

    for i in range(1, len(split_content), 2):
        heading = split_content[i].strip()
        content = split_content[i + 1].strip() if i + 1 < len(split_content) else ""
        heading_number_match = re.match(r"^([\d.]+)", heading)  # Extract just the number part
        if heading_number_match:
            heading_number = heading_number_match.group(1)  # Full numeric structure
        else:
            heading_number = heading

        if heading_number.endswith("."):
            heading_number = heading_number[:-1]
        print("Detected heading numbr: ", heading_number)
        full_hierarchy = hierarchy.get(heading_number, "NOT RETURNED")
        sections[heading] = {"full_hierarchy": full_hierarchy, "content": content}

    return sections


def postprocess_sections(sections: dict[str, str], lean=True) -> dict[str, str]:
    if lean:
        return [
            heading + "\n" + hr_content_dict["content"]
            for heading, hr_content_dict in sections.items()
            if hr_content_dict["content"].strip()
        ]
    else:
        return [
            hr_content_dict["full_hierarchy"] + "\n" + hr_content_dict["content"]
            for _, hr_content_dict in sections.items()
            if hr_content_dict["content"].strip()
        ]


def chunk_by_section(
    document: Document,
    toc: Document,
    pages: list[int],
) -> list[Chunk]:
    """
    Splits a document into chunks based on its sections and headings hierarchy.
    """
    # headings = get_headings(toc)
    # print("Extracted headings:\n")
    # for heading in headings:
    #     print("\n" + heading)
    with open(
        "/Users/cisemaltan/workspace/thesis/medical-rag-chatbot/data/headings.json", "r", encoding="utf-8"
    ) as file:
        data = json.load(file)
    headings = data["headings"]

    hierarchy = build_heading_hierarchy(headings)
    document_str = merge_document(document, pages=pages)
    sections = split_by_headings(document_str, headings, hierarchy)

    # This is required because adding higher level titles interfere with matching with pages
    lean_heading_sections = postprocess_sections(sections, lean=True)
    chunks = [Chunk(section, None, None) for section in lean_heading_sections]

    chunks, problem_count = match_chunks_with_pages(chunks, document, pages)
    sections_with_full_hieararchy_titles = postprocess_sections(sections, lean=False)

    for i in range(len(chunks)):
        chunks[i].text = sections_with_full_hieararchy_titles[i]
    if problem_count > 0:
        raise ValueError("Problems encountered during matchinkg chunks with pages")
    return chunks


def match_chunks_with_pages(
    chunks: list[Chunk], document: Document, pages: list[int], similarity_threshold: int = 97, overlap: bool = False
):
    """Works with consecutive pages only

    Args:
        chunks (list[Chunk]): _description_
        document (Document): _description_
        start_page (int): _description_
        overlap (bool, optional): _description_. Defaults to False.
    """
    problem_counter = 0

    start_pointer = 0  # index
    current_pointer = start_pointer

    for n, chunk in enumerate(chunks):
        matched = False
        start = time.time()

        while not matched:
            interest_part = " ".join(
                document.get_page(pages[i]).processed_content for i in range(start_pointer, current_pointer + 1)
            )

            if len(interest_part) < len(chunk.text):
                current_pointer += 1
                continue

            rat = fuzz.partial_ratio(chunk.text, interest_part)

            if rat > similarity_threshold:
                matched = True
                chunk.start_page = pages[start_pointer]
                chunk.end_page = pages[current_pointer]

                trailing_part = document.get_page(pages[current_pointer]).processed_content[-300:]
                if overlap and fuzz.partial_ratio(trailing_part, chunk.text) > similarity_threshold:
                    start_pointer = current_pointer + 1
                else:
                    start_pointer = current_pointer
            else:
                current_pointer += 1

            if current_pointer >= len(pages):
                print("Chunk not found in document")
                problem_counter += 1
                break

            if current_pointer - start_pointer == 15:
                print("Pointers grew apart")
                problem_counter += 1
                break

            if time.time() - start > 120:
                print("Timeout")
                problem_counter += 1
                break
    print(f"Problems encountered while matching chunks with pages: {problem_counter}")
    return chunks, problem_counter


def chunk_document(
    method: Literal["size", "semantic", "section"], document: Document, pages: list[int], **kwargs
) -> list[Chunk]:
    if method == "size":
        print("Chunking by size")
        chunks = chunk_by_size(document, pages, **kwargs)
    elif method == "semantic":
        print("Chunking by semantic")
        chunks = chunk_semantic(document, pages, **kwargs)
    elif method == "section":
        print("Chunking by section")
        toc = kwargs.pop("toc", None)
        if toc is None:
            ValueError("toc document must be provided for section method")
        chunks = chunk_by_section(document, toc, pages, **kwargs)
    else:
        ValueError("Invalid method")

    print("Number of chunks created: ", len(chunks))

    return chunks

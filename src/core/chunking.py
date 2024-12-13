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

    return [Chunk(document.page_content, None, None) for document in docs]


def chunk_semantic(document: Document, pages: list[int]) -> list[Chunk]:
    document_str = merge_document(document, pages=pages)
    text_splitter = SemanticChunker(NomicEmbeddings(model="nomic-embed-text-v1.5"))
    chunks = text_splitter.split_text(document_str)  # or create_documents using my get_document
    return [Chunk(chunk, None, None) for chunk in chunks]


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

    return headings


def split_by_headings(doc: str, headings: list[str]) -> dict:
    sections = {}

    headings_pattern = "|".join(
        [re.escape(heading).replace(r"\ ", r"\s*").replace(r"\.", r"\.?\s*") for heading in headings]
    )
    split_content = re.split(f"({headings_pattern})", doc)

    for i in range(1, len(split_content), 2):
        heading = split_content[i].strip()
        content = split_content[i + 1].strip() if i + 1 < len(split_content) else ""
        sections[heading] = content

    return sections


def postprocess_sections(sections: dict[str, str]) -> dict[str, str]:
    # TODO: Maybe add higher level headings to the content of the lower level headings
    # 4. Allgemeine Behandlungsmaßnahmen / Basistherapie 4.1 Oxygenierung content blabla
    return [heading + "\n" + content for heading, content in sections.items() if content.strip()]


def chunk_by_section(
    document: Document,
    toc: Document,
    pages: list[int],
) -> list[Chunk]:
    headings = get_headings(toc)
    print("Extracted headings:\n")
    for heading in headings:
        print("\n" + heading)

    document_str = merge_document(document, pages=pages)
    sections = postprocess_sections(split_by_headings(document_str, headings))
    return [Chunk(section, None, None) for section in sections]


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
        chunks = chunk_by_size(document, pages, **kwargs)
    elif method == "semantic":
        chunks = chunk_semantic(document, pages, **kwargs)
    elif method == "section":
        toc = kwargs.pop("toc", None)
        if toc is None:
            ValueError("toc document must be provided for section method")
        chunks = chunk_by_section(document, toc, pages, **kwargs)
    else:
        ValueError("Invalid method")

    print("Number of chunks created: ", len(chunks))

    chunks, problem_count = match_chunks_with_pages(chunks, document, pages)
    if problem_count > 0:
        raise ValueError("Problems encountered during matchinkg chunks with pages")
    return chunks

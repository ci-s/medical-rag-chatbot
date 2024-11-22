import json
import re
import time
from typing import Literal
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_nomic import NomicEmbeddings

from thefuzz import fuzz
from core.ollama import generate_response
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


def get_headings(document: Document) -> list[str]:
    headings = []

    for page in document.pages:
        # TODO: Add error handling for invalid JSON format
        response = generate_response(HEADINGS_PROMPT + page.processed_content)
        # print(response)
        # print("______________________")
        heading_dict_list = json.loads(response)
        headings.extend([d["number"] + ". " + d["heading"] for d in heading_dict_list])

        # Only include the headings that are within the table of contents (i.e. NOT "Inhalt") and only the headings and subheadings up to the second level i.e. 4.1. Don't include subheadings BEYOND the second level i.e. 4.1.1.
        # Do not say anything else and don't use markdown. Make sure the response is a valid JSON.\n

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

    start_pointer = pages[0]
    current_pointer = start_pointer

    for n, chunk in enumerate(chunks):
        print("*" * 50)
        print(f"Processing chunk {n}")
        matched = False
        start = time.time()

        while not matched:
            interest_part = " ".join(
                document.get_page(i).processed_content for i in range(start_pointer, current_pointer + 1)
            )  # processed or raw content?

            if len(interest_part) < len(chunk.text):
                current_pointer += 1
                continue

            rat = fuzz.partial_ratio(chunk.text, interest_part)

            print(f"Partial ratio similarity score: {rat}")
            if rat > similarity_threshold:
                matched = True
                chunk.start_page = start_pointer
                chunk.end_page = current_pointer

                trailing_part = document.get_page(current_pointer).processed_content[-300:]
                if overlap and fuzz.partial_ratio(trailing_part, chunk.text) > similarity_threshold:
                    start_pointer = current_pointer + 1
                else:
                    start_pointer = current_pointer
            else:
                current_pointer += 1

            if current_pointer > pages[-1]:
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
    print(f"Problems encountered: {problem_counter}")
    return chunks, problem_counter


def chunk_document(
    method: Literal["size", "semantic", "section"], document: Document, pages: list[int], **kwargs
) -> list[Chunk]:
    if method == "size":
        chunks = chunk_by_size(document, pages, **kwargs)
    elif method == "semantic":
        chunks = chunk_semantic(document, pages, **kwargs)
    elif method == "section":
        toc_pages = kwargs.pop("toc_pages", None)
        if toc_pages is None:
            ValueError("toc_pages must be provided for section method")
        chunks = chunk_by_section(document, toc_pages, pages, **kwargs)
    else:
        ValueError("Invalid method")

    print("Number of chunks created: ", len(chunks))

    chunks, problem_count = match_chunks_with_pages(chunks, document, pages)
    if problem_count > 0:
        raise ValueError("Problems encountered during matchinkg chunks with pages")
    return chunks

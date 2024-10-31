import json
import re
import time

from thefuzz import fuzz
from core.pdf_utils import read_pdf
from core.ollama import generate_response
from structures.page import Document, Chunk


def get_headings(file_path: str, pages: list[int]) -> list[str]:
    headings = []

    for page in pages:
        content = read_pdf(file_path, pages=[page])

        prompt = """
            Read the PDF page containing table of content below and provide the list of headings in JSON format as follows:
            [
            {
                "number": "4",
                "heading": "Allgemeine Behandlungsmaßnahmen / Basistherapie"
            },
            {
                "number": "4.1",
                "heading": "Oxygenierung"
            }
            ]

            for a content as follows:
                "Inhalt
                4. Allgemeine Behandlungsmaßnahmen / Basistherapie ………………………………………....
                4.1 Oxygenierung ….............................
                88
                92"

            Please ensure the list starts from the very first heading in the content and continues up to the second level of subheadings. Only include headings from the table of contents, excluding "Inhalt." Do not include subheadings beyond the second level (e.g., no 4.1.1). Do not say anything else and don't use markdown. Make sure the response is a valid JSON.\n

        """
        # TODO: Add error handling for invalid JSON format
        response = generate_response(prompt + content)
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
    file_path,
    toc_pages: list[int],
    content_pages: list[int],
) -> list[Chunk]:
    doc = read_pdf(file_path, pages=content_pages)

    headings = get_headings(file_path, pages=toc_pages)
    print("Extracted headings:\n")
    for heading in headings:
        print("\n" + heading)

    sections = postprocess_sections(split_by_headings(doc, headings))
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

        print(chunk.start_page, chunk.end_page)
        print(chunk.text[:50])

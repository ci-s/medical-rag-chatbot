import json
import re

from core.pdf_utils import read_pdf
from core.ollama import generate_response


def get_headings(file_path: str, pages: list[int]) -> list[str]:
    headings = []

    for page in pages:
        content = read_pdf(file_path, pages=[page])
        prompt = """
            Read the PDF page containing table of content below and provide the list of headings in JSON format as follows:
            [
            {
                "id": "1",
                "heading": "Einleitung"
            },
            {
                
            },
            ...
            ]
            Do not say anything else and don't use markdown. Remember "Inhalt" should not be included to the heading list.\n
        """

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
            
            Only include the headings that are within the table of contents (i.e. NOT "Inhalt") and only the headings and subheadings up to the second level i.e. 4.1. Don't include subheadings BEYOND the second level i.e. 4.1.1.
            Do not say anything else and don't use markdown.\n
        """
        # TODO: Add error handling for invalid JSON format
        response = generate_response(prompt + content)
        print(response)
        print("______________________")
        heading_dict_list = json.loads(response)
        headings.extend([d["number"] + ". " + d["heading"] for d in heading_dict_list])

    return headings


def split_by_headings(doc: str, headings: list[str]) -> dict:
    # Prepare a dictionary to hold the sectioned content
    sections = {}

    # Create a regex pattern to match any of the headings in the document
    headings_pattern = "|".join(
        [re.escape(heading).replace(r"\ ", r"\s*").replace(r"\.", r"\.?\s*") for heading in headings]
    )
    # Split the document using the headings, keeping the headings in the split
    split_content = re.split(f"({headings_pattern})", doc)

    # Iterate through the split content to pair each heading with its content
    for i in range(1, len(split_content), 2):  # Step through the split content
        heading = split_content[i].strip()  # The heading itself
        content = (
            split_content[i + 1].strip() if i + 1 < len(split_content) else ""
        )  # The content following the heading
        sections[heading] = content  # Add the heading and its content to the sections

    return sections


def postprocess_sections(sections: dict[str, str]) -> dict[str, str]:
    # TODO: Maybe add higher level headings to the content of the lower level headings
    # 4. Allgemeine Behandlungsmaßnahmen / Basistherapie 4.1 Oxygenierung content blabla
    return [heading + "\n" + content for heading, content in sections.items() if content.strip()]


def chunk_by_section(
    file_path,
    toc_pages: list[int],
    content_pages: list[int],
) -> list[str]:
    doc = read_pdf(file_path, pages=content_pages)

    headings = get_headings(file_path, pages=toc_pages)

    return postprocess_sections(split_by_headings(doc, headings))

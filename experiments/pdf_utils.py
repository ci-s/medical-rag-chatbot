from pypdf import PdfReader
import re

# header_footer_pattern = r"Klinik\sund\sPoliklinik\sfür\sNeurologie\sSeite\s\d+\svon\s\d+Handbuch\sVaskuläre\sNeurologie\s/\sStroke\sUnit\sVersion\s\d{4}"

header_pattern = r"Klinik\sund\sPoliklinik\sfür\sNeurologie"
footer_pattern_page = r"Seite\s\d+\svon\s\d+"
title_pattern = r"Handbuch\sVaskuläre\sNeurologie\s/\sStroke\sUnit"
version_pattern = r"Version\s\d{4}"

alt1_header_pattern = r"Technische\sUniversität\sMünchen"
alt2_header_pattern = r"Klinikum\srechts\sder\sIsar"

page_pattern = r"Seite\s\d+\s(v\s?on|von)\s\d+"

remove_list = [
    header_pattern,
    footer_pattern_page,
    title_pattern,
    version_pattern,
    alt1_header_pattern,
    alt2_header_pattern,
    page_pattern,
]


def read_pdf(file_path):
    reader = PdfReader(file_path)
    doc = ""
    for i, page in enumerate(reader.pages):
        page_content = page.extract_text()

        for pattern in remove_list:
            page_content = re.sub(pattern, " ", page_content)

        doc = doc + f"\nSeite {i}\n" + page_content
    return doc

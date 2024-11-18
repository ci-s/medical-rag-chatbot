import re
from domain.document import Document


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def normalize_text(text):
    return re.sub(r"[^a-zA-Z0-9]", "", text)


def merge_document(document: Document, pages: list[int], raw: bool = False) -> str:
    doc = ""
    for page in document.pages:
        if page.page_number not in pages:
            continue
        if raw:
            doc = doc + "\f" + page.raw_content
        else:
            doc = doc + "\f" + page.processed_content

    return doc

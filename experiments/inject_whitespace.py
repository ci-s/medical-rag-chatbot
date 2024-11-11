import re
import os
import sys

import pandas as pd
import pickle

from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)
from core.document import load_document, process_document  # noqa: E402


def report_replaced_abbreviation_count(document):
    abbreviations = pd.read_csv("../data/Abbreviations - Revised_by_clinicians.csv")
    abbreviation_dict = dict(zip(abbreviations["Abbreviation"], abbreviations["Meaning"]))

    def replace_abbreviations(text: str, abbreviation_dict: dict):
        pattern = re.compile(r"\b(" + "|".join(re.escape(abbr) for abbr in abbreviation_dict.keys()) + r")\b")

        count = 0

        def replacer(match):
            nonlocal count
            count += 1
            return abbreviation_dict[match.group()]

        result_text = pattern.sub(replacer, text)

        return result_text, count

    total_counts = {}
    for page in document.pages:
        if not page.processed_content:
            continue
        page.processed_content, count = replace_abbreviations(page.processed_content, abbreviation_dict)
        print(f"For page: {page.page_number} Number of abbreviations replaced: {count}")
        total_counts[page.page_number] = count

    print(f"Total number of abbreviations replaced: {sum(total_counts.values())}")
    return total_counts


result_path = "../results"
data_path = "../data"
file_name = "MNL_VA_Handbuch_vaskulaere_Neurologie_221230.pdf"

file_path = os.path.join(data_path, file_name)
pages = list(range(7, 109))

document = load_document(file_path, pages=pages)
document = process_document(document, is_replace_abbreviations=False)
original_abb_count = report_replaced_abbreviation_count(document)
pickle.dump(original_abb_count, open("original_abb_count.pkl", "wb"))
print("Original number of abbreviations replaced saved")

document = load_document(file_path, pages=pages)
document = process_document(document, is_replace_abbreviations=False)
print("Reprocessing with whitespace injection")
print("Document reloaded, number of pages: ", len(document.pages))

llm = ChatOllama(model="llama3.1:8b-instruct-q4_0", temperature=0, format="json")

chain = llm | JsonOutputParser()


def improve_pdf_output(text: str):
    messages = [
        (
            "system",
            """I have a block of text where some words are concatenated due to errors in PDF extraction. Your task is to identify where spaces are missing between words and add them where necessary. Please only adjust spacing—do not modify punctuation, capitalization, or any part of the text otherwise. Return the modified text with spaces added where needed in a json format as follows: {"output": "modified text here"} Say nothing else and don't change anything else.""",
        ),
        ("human", text),
    ]

    return chain.invoke(messages)


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
    # Remove all whitespace and punctuation, but keep numbers and letters
    return re.sub(r"[^a-zA-Z0-9]", "", text)


def has_significant_changes(original, modified, threshold=20):
    # Normalize texts by removing whitespace and punctuation
    normalized_original = normalize_text(original)
    normalized_modified = normalize_text(modified)

    # Calculate Levenshtein distance on normalized texts
    distance = levenshteinDistance(normalized_original, normalized_modified)

    # Check if distance is within an acceptable threshold
    return distance  # distance > threshold


significantly_changed = {}
NUM_CHANGES_THRESHOLD = 20  # Number of changes (on top of whitespaces) that are considered significant

for page in document.pages:
    print(f"Processing page {page.page_number}")

    content_before = page.processed_content
    if page.processed_content:
        if page.page_number == 67:
            continue
        try:
            output = improve_pdf_output(page.processed_content)
        except KeyboardInterrupt:
            print(f"Error processing page {page.page_number}. Skipping")
        page.processed_content = output["output"]
        num_changes = has_significant_changes(content_before, page.processed_content)
        print(f"Number of changes made apart from whitespace injecting: {num_changes}")
        if num_changes > NUM_CHANGES_THRESHOLD:
            print(f"Significant changes detected on page {page.page_number}. Won't save.")
            significantly_changed[page.page_number] = {
                "num_changes": num_changes,
                "original": content_before,
                "modified": page.processed_content,
            }
            page.processed_content = content_before
        else:
            print(f"Saving page {page.page_number}")
    else:
        print(f"Empty page content. Skipping {page.page_number}")

document.save(os.path.join(result_path, file_name).replace(".pdf", "_processed_2.pkl"))

pickle.dump(significantly_changed, open("whitespace_injection_log_2.pkl", "wb"))

injected_abb_count = report_replaced_abbreviation_count(document)
pickle.dump(injected_abb_count, open("injected_abb_count.pkl", "wb"))
print("Injected number of abbreviations replaced saved")

import sys
import os
import json
import pickle
import pandas as pd

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from core.document import read_pdf
from core.ollama import generate_response

data_path = "../data"
file_name = "MNL_VA_Handbuch_vaskulaere_Neurologie_221230.pdf"

file_path = os.path.join(data_path, file_name)
doc = read_pdf(file_path, pages=list(range(7, 109)))  # First 6 pages = table of contents etc.


def get_abbreviations_from_pages(pages: list[int]) -> list:
    abbreviations = []
    pages_with_errors = []

    for page in pages:
        print("Processing page ", str(page))
        content = read_pdf(file_path, pages=[page])
        prompt = """
            Read the PDF page and provide the list of abbreviations and their meanings in JSON format as follows:
            [
            {
                "abbreviation": "SOP",
                "meaning": "Standard Operating Procedure"
            },
            ...
            ]

            Don't include names of people or units of measurement. The meaning should be in the language of the abbreviation. Most probably English or German in this case. If you are in doubt, I prefer German, but don't try to translate words that are not abbreviations.
            
            Do not say anything else and don't use markdown.\n
        """
        response = generate_response(prompt + content)

        try:
            dict_list = json.loads(response)

            for d in dict_list:
                abbreviations.append((d["abbreviation"], d["meaning"]))
        except:
            print("PROBLEM WITH PAGE")
            pages_with_errors.append(page)

        print("Length of abbreviations: ", len(abbreviations))

    return abbreviations, pages_with_errors


pages = list(range(1, 109))
abbreviations, pages_with_errors = get_abbreviations_from_pages(pages)
print("Number of pages with errors: ", len(pages_with_errors))

# Second try
print("*" * 20, "Second try", "*" * 20)
additional_abbreviations, pages_with_errors = get_abbreviations_from_pages(pages_with_errors)
abbreviations.extend(additional_abbreviations)
print("Number of pages with errors: ", len(pages_with_errors))

# Third try
print("*" * 20, "Third try", "*" * 20)
additional_abbreviations, pages_with_errors = get_abbreviations_from_pages(pages_with_errors)
abbreviations.extend(additional_abbreviations)
print("Number of pages with errors: ", len(pages_with_errors))


with open("abbreviations_3.pkl", "wb") as fp:
    pickle.dump(abbreviations, fp)

pd.DataFrame(abbreviations, columns=["Abbreviation", "Meaning"]).to_csv("abbreviations_3.csv", index=False)

counts = abbreviations.groupby("Abbreviation").size().reset_index(name="Count").sort_values("Count", ascending=False)

abbreviations.drop_duplicates(inplace=True)
abbreviations.merge(counts, on="Abbreviation", how="left").sort_values(["Count", "Meaning"], ascending=False)

pd.DataFrame(abbreviations, columns=["Abbreviation", "Meaning"]).to_csv("abbreviations_3_dedup.csv", index=False)
print("Left pages with errors: ", pages_with_errors)
print("Done")

# This script is intended to work with a different environment than the rest of the project.
# It is used to identify non-text pages in a PDF file by using the LlavaNext model.
# The name of the environment is "vlm" in the translatum server.

import os
import json
import pickle

from PIL import Image
import pymupdf

import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

DATA_PATH = "../../data"
FILE_NAME = "MNL_VA_Handbuch_vaskulaere_Neurologie_221230.pdf"
model_cache_dir = "/home/guests/cisem_altan/vlm_cache"
folder_name = "non_text_pages_handbuch_v2"

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", cache_dir=model_cache_dir)
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    cache_dir=model_cache_dir,
    load_in_4bit=True,
    device_map="auto",
)

prompt = """
[INST] <image>\nAnalyze the page and determine if it contains a flow chart or a table.

**Decision tree**: A diagram representing a process, workflow (flowchart), typically using shapes (e.g., rectangles, diamonds, or ovals) connected by arrows to indicate sequence or decisions. It can cover the entire page or part of it and may include rotated diagrams. Containing only one or two arrows within a text line does not qualify as a flow chart.

**Table**: A structured set of data with at least two columns and two rows. Having a bounding box around the bullet points does not make the page a table.

A page may contain both a flow chart and a table. 

Respond strictly using the following JSON format:
    {"decision_tree": <bool>, "table": <bool>} 

Do not generate any additional text or include the prompt in your response. [/INST]
"""
# Number of wrongly detected pages: 0
# Number of undetected pages: 22
# Number of correctly detected pages: 14
# Number of wrongly detected pages: 18
# Number of undetected pages: 3
# Number of correctly detected pages: 32

# """
# [INST] <image>\nTell me if the page contains a flow chart or a table. Flow chart defined as a diagram that represents a process with steps having arrow between. It might also be a decision tree. It might fit to the whole page or a part of it. Only containing one or two arrow within a text line does not make the page a flow chart. It can also be rotated 90 degrees.
# Table category contains a structured set of data with at least two columns and two rows. Having a bounding box around the bullet points does not make the page a table.

# A page can contain both a flow chart and a table.
# Consider the output template when you respond. Do not generate anything else. Here is the output template:
#     {"flow_chart": <bool>, "table": <bool>} Take a deep breath and answer only with a JSON. Do not include the prompt in your response. [/INST]
# """
# Number of wrongly detected pages: 0
# Number of undetected pages: 32
# Number of correctly detected pages: 4
# Number of wrongly detected pages: 7
# Number of undetected pages: 6
# Number of correctly detected pages: 29
# """
# [INST] <image>\nTell me if the page contains a flow chart or a table. Consider the output template when you respond. Do not generate anything else. Here is the output template:
#     {"flow_chart": "bool", "table":"bool"} Take a deep breath and answer only with a JSON. Do not include the prompt in your response. [/INST]
# """
# Number of wrongly detected pages: 1
# Number of undetected pages: 25
# Number of correctly detected pages: 12
# Number of wrongly detected pages: 25
# Number of undetected pages: 5
# Number of correctly detected pages: 27

prompt_len = len(prompt)
fname = os.path.join(DATA_PATH, FILE_NAME)
doc = pymupdf.open(fname)


# flow_chart_pages = []
# table_pages = []
# problematic_pages = {}

# for i, page in enumerate(doc):
#     pix = page.get_pixmap()  # render page to an image
#     image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#     #pix.save("page-%i.png" % page.number)  # store image as a PNG
#     inputs = processor(prompt, image, return_tensors="pt", ).to("cuda:0")

#     output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

#     response = processor.decode(output[0], skip_special_tokens=True, return_prompt=False)

#     try:
#         start_idx = response.rfind('{')
#         end_idx = response.rfind('}') + 1
#         if start_idx != -1 and end_idx != -1:
#             json_output = response[start_idx:end_idx]
#             json_output = json.loads(json_output)
#             print(json_output)

#         if json_output["flow_chart"]:
#             flow_chart_pages.append(i)
#         if json_output["table"]:
#             table_pages.append(i)
#     except KeyError:
#         print(f"KeyError for page {i}")
#         print("Response: ", response)
#         problematic_pages[i] = response
#     except json.JSONDecodeError:
#         print(f"JSONDecodeError for page {i}")
#         print("Response: ", response)
#         problematic_pages[i] = response

#     # Write flow_chart_pages and table_pages to a pickle file
#     output_file = "/home/guests/cisem_altan/medical-rag-chatbot/results/non_text_pages.pkl"
#     with open(output_file, "wb") as f:
#         pickle.dump((flow_chart_pages, table_pages, problematic_pages), f)

# TODO: do it recursive? so that it calls again if the response is not in the correct format


def process_page(page, prompt, processor, model, retries=3):
    """
    Processes a single page and retries if the response is not in the correct format.

    Args:
        page: The page object from pymupdf.
        prompt: The prompt string to be used.
        processor: The processor object for tokenizing inputs.
        model: The model object for generating outputs.
        retries: The maximum number of retries allowed.

    Returns:
        A tuple containing:
            - flow_chart (bool): Whether the page contains a flow chart.
            - table (bool): Whether the page contains a table.
            - problematic_response (str): The problematic response, if any.
    """
    pix = page.get_pixmap()  # Render page to an image
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    response = processor.decode(output[0], skip_special_tokens=True, return_prompt=False)

    try:
        start_idx = response.rfind("{")
        end_idx = response.rfind("}") + 1
        if start_idx != -1 and end_idx != -1:
            json_output = response[start_idx:end_idx]
            json_output = json.loads(json_output)
            return json_output["decision_tree"], json_output["table"], None
        else:
            raise json.JSONDecodeError("Response is not a valid JSON", response, start_idx)
    except (KeyError, json.JSONDecodeError) as e:
        if retries > 0:
            print(f"Retrying for page with response: {response}. Retries left: {retries}")
            return process_page(page, prompt, processor, model, retries - 1)
        else:
            print(f"Failed to process page after retries. Error: {e}")
            return None, None, response


flow_chart_pages = []
table_pages = []
problematic_pages = {}

for i, page in enumerate(doc):
    flow_chart, table, problematic_response = process_page(page, prompt, processor, model)

    if flow_chart:
        flow_chart_pages.append(i + 1)
    if table:
        table_pages.append(i + 1)
    if problematic_response:
        problematic_pages[i + 1] = problematic_response

    # Write flow_chart_pages and table_pages to a pickle file
    output_file = f"/home/guests/cisem_altan/medical-rag-chatbot/results/{folder_name}/non_text_pages.pkl"
    with open(output_file, "wb") as f:
        pickle.dump((prompt, flow_chart_pages, table_pages, problematic_pages), f)

##### ANALYSIS #####
# Labelled by me
true_flow_chart_pages = [
    8,
    10,
    11,
    12,
    13,
    14,
    18,
    22,
    26,
    28,
    29,
    30,
    33,
    36,
    38,
    41,
    43,
    50,
    54,
    55,
    57,
    60,
    61,
    64,
    67,
    69,
    71,
    72,
    76,
    79,
    80,
    81,
    83,
    85,
    88,
    94,
]

true_table_pages = [
    2,
    3,
    4,
    5,
    6,
    15,
    19,
    21,
    23,
    24,
    25,
    31,
    32,
    37,
    39,
    40,
    42,
    52,
    58,
    62,
    63,
    65,
    69,
    70,
    77,
    78,
    83,
    87,
    88,
    89,
    91,
    92,
    104,
    105,
    107,
]

difference = {
    "existing_flow_chart_not_detected": [],
    "non_existing_flow_chart_detected": [],
    "existing_table_not_detected": [],
    "non_existing_table_detected": [],
}

for p in true_flow_chart_pages:
    if not p in flow_chart_pages:
        difference["existing_flow_chart_not_detected"].append(p)

for p in flow_chart_pages:
    if not p in true_flow_chart_pages:
        difference["non_existing_flow_chart_detected"].append(p)

for p in true_table_pages:
    if not p in table_pages:
        difference["existing_table_not_detected"].append(p)

for p in table_pages:
    if not p in true_table_pages:
        difference["non_existing_table_detected"].append(p)

print("Difference: ", difference)
output_file = f"/home/guests/cisem_altan/medical-rag-chatbot/results/{folder_name}/non_text_pages_analysis.pkl"
with open(output_file, "wb") as f:
    pickle.dump(difference, f)


def get_overview(true_pages, detected_pages):
    overview = {}
    for p in set(true_pages).union(set(detected_pages)):
        overview = {}
        for p in set(true_pages).union(set(detected_pages)):
            if p in true_pages and p in detected_pages:
                overview[p] = "detected"
            elif p in true_pages and p not in detected_pages:
                overview[p] = "undetected"
            elif p not in true_pages and p in detected_pages:
                overview[p] = "wrongly detected"
            else:
                overview[p] = "not applicable"
    print("Number of wrongly detected pages:", len([p for p, v in overview.items() if v == "wrongly detected"]))
    print("Number of undetected pages:", len([p for p, v in overview.items() if v == "undetected"]))
    print("Number of correctly detected pages:", len([p for p, v in overview.items() if v == "detected"]))
    return overview


flow_chart_overview = get_overview(true_flow_chart_pages, flow_chart_pages)
table_overview = get_overview(true_table_pages, table_pages)

print("FC Overview: ", flow_chart_overview)
print("Table Overview: ", table_overview)
output_file = f"/home/guests/cisem_altan/medical-rag-chatbot/results/{folder_name}/non_text_pages_overview.pkl"
with open(output_file, "wb") as f:
    pickle.dump((flow_chart_overview, table_overview), f)

print("Problematic pages: ", problematic_pages)

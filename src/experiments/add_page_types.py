# Handbuch pages!

import sys
import os
import json

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from settings.settings import settings

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

visual_pages = [
    40,
    51,
    91,
    92,
    100,
    104,
    105,
]

# TODO: Another layer in the dict if antibiotika pages are added

page_data = {
    "flowchart": true_flow_chart_pages,
    "table": true_table_pages,
    "visual": visual_pages,
}

with open(os.path.join(settings.data_path, "page_data.json"), "w") as file:
    json.dump(page_data, file, indent=4)

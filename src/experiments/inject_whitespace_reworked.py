from enum import Enum
import sys
import os
from collections import Counter
import json
import time
import pickle

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from core.document import load_document, process_document, merge_document, get_text_in_table_pages

from settings.settings import settings
from settings import get_page_types


class Mode(Enum):
    TEXTPAGE = "TextPages"
    TABLEPAGE = "TablePages"
    # FLOWCHART = "FlowchartPages"


mode = Mode.TABLEPAGE
print("Mode is set as:", mode)

if mode == Mode.TEXTPAGE:
    file_path = os.path.join(settings.data_path, settings.raw_file_name)
    # pages, _, _, _ = get_page_types()
    pages = [8, 26, 32, 89]  # Revised table and flowchart pages
    document = load_document(file_path, pages=pages)
elif mode == Mode.TABLEPAGE:
    document = get_text_in_table_pages()

document = process_document(document, is_replace_abbreviations=True, whitespace_injection=True)

existing_document = load_document(os.path.join(settings.data_path, settings.file_name))

merged_document = merge_document([existing_document, document])

merged_document.save(
    os.path.join(settings.data_path, settings.raw_file_name).replace(".pdf", f"_processed_{int(time.time())}.pkl")
)

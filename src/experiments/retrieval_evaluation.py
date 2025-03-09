import sys
import os
import json
import time

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from core.document import get_document
from services.retrieval import FaissService
from core.chunking import chunk_document
from eval.retrieval import evaluate_source

from settings.settings import settings
from settings import get_page_types, config

file_path = os.path.join(settings.data_path, settings.file_name)

# Added new, never tried before
if config.filter_questions:
    if config.filter_questions == ["Text"]:
        pages, _, _, _ = get_page_types()
    elif config.filter_questions == ["Table"]:
        _, _, pages, _ = get_page_types()
    elif config.filter_questions == ["Flowchart"]:
        _, pages, _, _ = get_page_types()
    else:
        raise ValueError("Multiple filter_questions value is not configured for page types yet")
else:
    pages = list(range(7, 109))

# pages = list(range(7, 109))
# toc_pages = [2, 3]

document = get_document(file_path, pages)
# TODO: integrate mlflow
method_args = {
    # "semantic": {},  # set NOMIC_API_KEY
    # "section": {"toc": None},
    "section_and_size": {"toc": None, "chunk_size": config.chunk_size},
    "size": {"chunk_size": config.chunk_size},
}


result_dicts = []
for method, args in method_args.items():
    for optim_method in [
        None,
        "hypothetical_document",
        "decomposing",
        "paraphrasing",
        "stepback",
    ]:  # , "hypothetical_document", "decomposing", "paraphrasing", "stepback"
        if optim_method:
            config.optimization_method = optim_method
            config.use_original_query_only = False
        else:
            config.optimization_method = None
            config.use_original_query_only = True

        config.chunk_method = method
        subdict = {}
        subdict["method"] = method
        subdict["optim_method"] = str(optim_method)
        subdict["config"] = config.model_dump()
        subdict["settings"] = settings.model_dump(mode="json")

        print(f"Method: {method}")
        chunks = chunk_document(method=method, document=document, pages=pages, **args)

        faiss_service = FaissService()
        faiss_service.create_index(chunks)

        stats = evaluate_source("Handbuch", faiss_service)
        print(stats)

        subdict["result"] = stats.to_dict()
        result_dicts.append(subdict)


output_file = f"retrieval_eval_{int(time.time())}.json"
output_path = os.path.join(settings.results_path, output_file)
with open(output_path, "w") as file:
    json.dump(result_dicts, file, indent=4, ensure_ascii=False)

print("Results are saved in: ", output_path)

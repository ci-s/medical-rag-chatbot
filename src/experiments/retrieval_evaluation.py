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
pages, _, _, _ = get_page_types()
# pages = list(range(7, 109))
# toc_pages = [2, 3]

document = get_document(file_path, pages)
# toc = get_document(
#     "/Users/cisemaltan/workspace/thesis/medical-rag-chatbot/data/MNL_VA_Handbuch_vaskulaere_Neurologie_221230.pdf",
#     toc_pages,
#     is_replace_abbreviations=True,
# )

method_args = {
    # "semantic": {},  # set NOMIC_API_KEY
    "size": {"chunk_size": 512},
    # "section": {"toc": toc},
}


result_dicts = []
for method, args in method_args.items():
    for optim_method in [None, "hypothetical_document", "decomposing", "paraphrasing", "stepback"]:
        if optim_method:
            config.optimization_method = optim_method
            config.use_original_query_only = False
        else:
            config.optimization_method = None
            config.use_original_query_only = True

        subdict = {}
        subdict["method"] = method
        subdict["optim_method"] = str(optim_method)
        subdict["config"] = config.model_dump()

        print(f"Method: {method}")
        chunks = chunk_document(method=method, document=document, pages=pages, **args)

        faiss_service = FaissService()
        faiss_service.create_index(chunks)

        stats = evaluate_source("Handbuch", faiss_service, config.text_questions_only)
        print(stats)

        subdict["result"] = stats.to_dict()
        result_dicts.append(subdict)


output_file = f"retrieval_eval_{int(time.time())}.json"
output_path = os.path.join(settings.results_path, output_file)
with open(output_path, "w") as file:
    json.dump(result_dicts, file, indent=4)

print("Results are saved in: ", output_path)

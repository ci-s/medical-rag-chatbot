import sys
import os
from collections import Counter
import statistics
import json
import time

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from core.document import get_document
from services.retrieval import FaissService
from core.chunking import chunk_document

from settings.settings import settings
from settings import get_page_types, config
from eval.generation import evaluate_single, evaluate_source, evaluate_ragas

file_path = os.path.join(settings.data_path, settings.file_name)

if config.text_questions_only:
    pages, _, _, _ = get_page_types()
else:
    pages = list(range(7, 109))
print(f"Number of pages: {len(pages)}")
toc_pages = [2, 3]

document = get_document(file_path, pages)
# toc = get_document(file_path, toc_pages)
chunks = chunk_document(method="size", document=document, pages=pages, chunk_size=config.chunk_size)


faiss_service = FaissService()
faiss_service.create_index(chunks)


result_dicts = []
for optim_method in [
    None,
]:  # None, "hypothetical_document", "decomposing", "paraphrasing", "stepback"
    if optim_method:
        config.optimization_method = optim_method
        config.use_original_query_only = False
    else:
        config.optimization_method = None
        config.use_original_query_only = True

    # avg_score, all_feedbacks = evaluate_ragas("Handbuch", faiss_service, text_only=text_only)
    avg_score, all_feedbacks = evaluate_source("Handbuch", faiss_service, text_only=config.text_questions_only)
    tim = int(time.time())

    result_dict = {
        "config": config.model_dump(),
        "settings": settings.model_dump(mode="json"),
        "all_feedbacks": [fb.to_dict() for fb in all_feedbacks],
        "avg_score": avg_score,
        # "std_dev": std_dev,
        # "counted_values": counted_values,
    }

    # output_file = f"generation_eval_{tim}_{config.experiment_name}_{str(optim_method)}.json"
    # output_path = os.path.join(settings.results_path, output_file)
    # with open(output_path, "w") as file:
    #     json.dump(result_dict, file, indent=4)

    result_dicts.append(result_dict)


output_file = f"generation_eval_{tim}.json"
output_path = os.path.join(settings.results_path, output_file)
with open(output_path, "w") as file:
    json.dump(result_dicts, file, indent=4)

print("Results are saved in: ", output_path)

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
from eval.generation import evaluate_ragas_qids

file_path = os.path.join(settings.data_path, settings.file_name)

pages, _, _, _ = get_page_types()
print(f"Number of pages: {len(pages)}")

document = get_document(file_path, pages)
chunks = chunk_document(method="size", document=document, pages=pages, chunk_size=512)

faiss_service = FaissService()
faiss_service.create_index(chunks)


# No retrieval optimization method
question_ids_w_good_retrieval = [9, 14, 18, 23, 24, 30, 34, 35, 42, 43, 48, 51, 52, 65, 72, 81]  #
# question_ids_w_good_retrieval_good_generation = 23
# 26 = good retrieval, good generation but answer and context relevance = 0

result_dicts = []
for optim_method in [None]:  # "stepback", None, "hypothetical_document", "decomposing", "paraphrasing",
    if optim_method:
        config.optimization_method = optim_method
        config.use_original_query_only = False
    else:
        config.optimization_method = None
        config.use_original_query_only = True

    avg_score, all_feedbacks = evaluate_ragas_qids("Handbuch", faiss_service, question_ids_w_good_retrieval)

    result_dict = {
        "config": config.model_dump(),
        "all_feedbacks": [fb.to_dict() for fb in all_feedbacks],
        "avg_score": avg_score,
    }

    # output_file = f"generation_eval_{config.experiment_name}_{str(optim_method)}_{int(time.time())}.json"
    # output_path = os.path.join(settings.results_path, output_file)
    # with open(output_path, "w") as file:
    #     json.dump(result_dict, file, indent=4)

    result_dicts.append(result_dict)


output_file = f"generation_eval_{int(time.time())}.json"
output_path = os.path.join(settings.results_path, output_file)
with open(output_path, "w") as file:
    json.dump(result_dicts, file, indent=4)

print("Results are saved in: ", output_path)

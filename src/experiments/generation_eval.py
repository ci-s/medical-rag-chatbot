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
from settings import get_page_types
from eval.generation import evaluate_single, evaluate_source

file_path = os.path.join(settings.data_path, settings.file_name)

pages, _, _, _ = get_page_types()
print(f"Number of pages: {len(pages)}")
# toc_pages=[2,3]

document = get_document(file_path, pages)
# toc = get_document(file_path, toc_pages)
chunks = chunk_document(method="size", document=document, pages=pages, chunk_size=512)

faiss_service = FaissService()
faiss_service.create_index(chunks)

top_k = 3
model = "8B"
text_only = True
if text_only:
    questions = "text-only"
else:
    questions = "all"

# f = evaluate_single(
#     vignette_id=0,
#     question_id=1,
#     faiss_service=faiss_service,
# )
# f.to_dict()

avg_score, all_feedbacks = evaluate_source("Handbuch", faiss_service, top_k=top_k, text_only=text_only)

scores = []
for feedback in all_feedbacks:
    scores.append(int(feedback.score))

counted_values = Counter(scores)
std_dev = statistics.stdev(scores)

result = {
    "avg_score": avg_score,
    "std_dev": std_dev,
    "counted_values": counted_values,
    "questions": questions,
    "topk": top_k,
    "model": model,
    "all_feedbacks": [fb.to_dict() for fb in all_feedbacks],
}

output_file = f"generation_eval_{int(time.time())}.json"
with open(os.path.join(settings.results_path, output_file), "w") as file:
    json.dump(result, file, indent=4)

print(result)
print("Results are saved in the data folder: ", output_file)

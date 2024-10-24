import yaml
from statistics import mean

from structures.page import Chunk
from core.embedding import embed_chunks
from services.retrieval import FaissService


class Stats:
    pct: float
    total: int

    def __init__(self, pct: float, total: int):
        self.pct = pct
        self.total = total


ypath = "../data/fallvignetten.yaml"  # TODO: move to config

with open(ypath, "r") as file:
    vignette_yaml = yaml.safe_load(file)


def get_references_w_id(vignette_id, question_id) -> list[int]:
    return vignette_yaml["vignettes"][vignette_id]["questions"][question_id]["reference"]


def get_references(query: str) -> list[int]:
    # Assuming there are no duplicate questions
    for vignette in vignette_yaml["vignettes"]:
        for question in vignette["questions"]:
            if query == question["question"]:
                return question["reference"]


def evaluate_single(query: str, retrieved_passages: list[Chunk]) -> Stats:
    reference_pages = get_references(query)
    print("Reference pages are: ", reference_pages)
    covered_reference_count = 0
    total_reference_count = len(reference_pages)

    for reference_page in reference_pages:
        for retrieved_passage in retrieved_passages:
            if reference_page in list(range(retrieved_passage.start_page, retrieved_passage.end_page + 1)):
                covered_reference_count += 1
                break

    return Stats(pct=covered_reference_count / total_reference_count, total=total_reference_count)


def evaluate_all(chunks: list[Chunk], faiss_service: FaissService, top_k: int = 3) -> list[Stats]:
    all_stats = []

    for vignette in vignette_yaml["vignettes"]:
        context = vignette["context"]
        background = vignette["background"]

        for question in vignette["questions"]:
            query_embedding = embed_chunks(question["question"], task_type="search_query")

            similarity, i = faiss_service.search_index(query_embedding, top_k)
            retrieved_documents = [chunks[idx] for idx in i]

            all_stats.append(evaluate_single(question["question"], retrieved_documents))

    return mean([stat.pct for stat in all_stats])

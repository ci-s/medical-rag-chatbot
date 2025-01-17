from typing import Literal
from typing import Literal
import faiss

from core.embedding import embed_chunks
from settings.settings import config
from core.utils import replace_abbreviations
from domain.vignette import Vignette, Question
from core.model import create_user_question_prompt, generate_response
from prompts import HYPOTHETICAL_DOCUMENT_PROMPT, STEPBACK_PROMPT, DECOMPOSING_PROMPT, PARAPHRASING_PROMPT

from langchain_core.output_parsers import BaseOutputParser


class FaissService:
    def __init__(self):
        self.index = None
        self.chunks = None

    def create_index(self, chunks: list[str]):
        if chunks is None:
            raise ValueError("Image representations cannot be None")

        embeddings = embed_chunks([chunk.text for chunk in chunks], task_type="search_document")

        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)  # TODO: parametrize
        index.add(embeddings)
        print("Index created with {} chunks".format(index.ntotal))
        self.index = index
        self.chunks = chunks

    def search_index(self, query, k):
        if isinstance(query, str):
            query_embedding = self._encode_chunks([query])
        else:
            query_embedding = query

        D, I = self.index.search(query_embedding, k)

        return D[0], [self.chunks[idx] for idx in I[0]]


def _retrieve(query: str, faiss_service: FaissService) -> list[str]:
def _retrieve(query: str, faiss_service: FaissService) -> list[str]:
    query, _ = replace_abbreviations(query)
    query_embedding = embed_chunks(query, task_type="search_query")

    similarities, retrieved_documents = faiss_service.search_index(query_embedding, config.top_k)

    print("Retrieved documents:")
    for i, retrieved_document in enumerate(retrieved_documents):
        print("*" * 20, f"Retrieval {i + 1} with similarity score {similarities[i]}", "*" * 20)
        print(retrieved_document)

    return retrieved_documents


class LineListOutputParser(BaseOutputParser[list[str]]):
    """Robust output parser for a list of lines."""

    def parse(self, text: str) -> list[str]:
        if "\\n" in text:
            text = text.replace("\\n", "\n")
        text = text.replace("\n\n", "\n")
        lines = text.strip().split("\n")
        cleaned_lines = [line.strip().strip('"').strip("'") for line in lines if line.strip()]

        return list(filter(None, cleaned_lines))


output_parser = LineListOutputParser()  # TODO: Move


def parse_optimized_query(response: str) -> str:
    if config.optimization_method is None:
        return response
    elif config.optimization_method == "hypothetical_document":
        return response
    elif config.optimization_method == "stepback":
        return response

    try:
        return output_parser.parse(response)
    except Exception as e:
        raise ValueError(f"Failed to parse optimized query: {e}") from e


def get_optimization_prompt() -> str:
    method = config.optimization_method

    if method == "hypothetical_document":
        return HYPOTHETICAL_DOCUMENT_PROMPT
    elif method == "stepback":
        return STEPBACK_PROMPT
    elif method == "decomposing":
        return DECOMPOSING_PROMPT
    elif method == "paraphrasing":
        return PARAPHRASING_PROMPT
    else:
        raise ValueError("Invalid optimization method")


def retrieve_and_rank(queries: list[str], faiss_service: FaissService):
    # Current implementation returns top k documents with the highest scores from all queries
    # Not exactly sure if this is the best way to combine results from multiple queries
    # Because of the length of the queries, does it make sense to compare scores?
    all_retrieved_documents = []

    for query in queries:
        query, _ = replace_abbreviations(query)
        query_embedding = embed_chunks(query, task_type="search_query")

        similarities, retrieved_documents = faiss_service.search_index(query_embedding, config.top_k)
        all_retrieved_documents.extend(zip(retrieved_documents, similarities))

    all_retrieved_documents.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in all_retrieved_documents[: config.top_k]]


def retrieve(
    vignette: Vignette,
    question: Question,
    faiss_service: FaissService,
) -> tuple[str, str]:
    if config.use_original_query_only:
        print("Using original query only")
        return _retrieve(question.get_question(), faiss_service)

    print("Using optimized query with method: ", config.optimization_method)

    system_prompt = get_optimization_prompt()
    user_prompt = create_user_question_prompt(vignette, question)
    response = generate_response(user_prompt, system_prompt)
    print("Response:", response)
    new_query = parse_optimized_query(response)
    print("New query:", new_query)
    # add parser because decompose will return two queries
    if isinstance(new_query, list):
        retrieved_documents = retrieve_and_rank(new_query, faiss_service)
    else:
        retrieved_documents = _retrieve(new_query, faiss_service)

    # if config.use_original_along_with_optimized:
    #     orig_retrieved_documents = _retrieve(question.get_question(), faiss_service)
    #     retrieved_documents.extend(orig_retrieved_documents)
    #     retrieved_documents = list(set(retrieved_documents))
    #     # TODO: Maybe sort by similarity score?

    return retrieved_documents

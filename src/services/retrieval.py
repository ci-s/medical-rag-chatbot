import faiss
from typing import Callable

from core.embedding import embed_chunks
from settings.settings import config
from core.utils import replace_abbreviations
from domain.vignette import Vignette, Question
from domain.document import Chunk, ChunkType
from core.model import generate_response
from core.question_answering import create_user_question_prompt
from prompts import HYPOTHETICAL_DOCUMENT_PROMPT, STEPBACK_PROMPT, DECOMPOSING_PROMPT, PARAPHRASING_PROMPT

from langchain_core.output_parsers import BaseOutputParser


class FaissService:
    def __init__(self):
        self.index = None
        self.chunks: list[Chunk] = None
        self.retrieval_strings: list[str] = None

    def create_index(
        self,
        chunks: list[Chunk],
        retrieve_by: Callable[[Chunk], str] = lambda chunk: chunk.section_heading + " " + chunk.text
        if chunk.section_heading
        else chunk.text,
    ):
        if chunks is None:
            raise ValueError("Image representations cannot be None")

        self.retrieval_strings = [retrieve_by(chunk) for chunk in chunks]
        embeddings = embed_chunks(
            self.retrieval_strings,
            task_type="search_document",
        )

        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)  # TODO: parametrize
        index.add(embeddings)
        print("Index created with {} chunks".format(index.ntotal))
        self.index = index
        self.chunks = chunks
        self.set_chunk_indices()

    def search_index(self, query_embedding, k: int) -> tuple[list[float], list[Chunk]]:
        D, I = self.index.search(query_embedding, k)

        retrieved_indices = {idx: sim for idx, sim in zip(I[0], D[0])}

        if config.surrounding_chunk_length > 0:
            print("Expanding retrieved indices with surrounding chunks")
            original_indices = list(retrieved_indices.items())
            print("Original indices: ", original_indices)
            for idx, similarity_score in original_indices:
                original_section = self.chunks[idx].section_heading
                for offset in range(1, config.surrounding_chunk_length + 1):
                    for new_idx in (idx - offset, idx + offset):
                        if 0 <= new_idx < len(self.chunks) and self.chunks[new_idx].section_heading == original_section:
                            # Add only if index doesn't exist or has a lower similarity score
                            if new_idx not in retrieved_indices:
                                retrieved_indices[new_idx] = 0
                            elif retrieved_indices[new_idx] < similarity_score:
                                retrieved_indices[new_idx] = similarity_score

            print("Expanded indices: ", retrieved_indices)
        expanded_indices = sorted(retrieved_indices.items())
        expanded_indices.sort(key=lambda tup: tup[0])
        sorted_retrieved_chunks = [(self.chunks[idx], sim) for idx, sim in expanded_indices]
        # print("Sorted retrieved chunks as str: ", [(score, str(c)) for c, score in sorted_retrieved_chunks])
        scores, returned_chunks = self.merge_chunks_if_consecutive(sorted_retrieved_chunks)
        return scores, returned_chunks

    def set_chunk_indices(self):
        for i, chunk in enumerate(self.chunks):
            chunk.index = i

    def merge_chunks_if_consecutive(self, sorted_chunks: list[tuple[Chunk, float]]) -> list[Chunk]:
        merged_chunks = []

        scores = [score for _, score in sorted_chunks]
        sorted_chunks = [chunk for chunk, _ in sorted_chunks]
        current_chunk = sorted_chunks[0].copy()
        current_scores = [scores[0]]

        for i in range(1, len(sorted_chunks)):
            next_chunk = sorted_chunks[i].copy()
            next_score = scores[i]

            # No merge for tables and flowcharts
            # If type is none, might be problematic
            if (
                next_chunk.type not in (ChunkType.TABLE, ChunkType.FLOWCHART)
                and next_chunk.index == (current_chunk.index + 1)
                and next_chunk.section_heading == current_chunk.section_heading
            ):
                current_chunk.text += " " + next_chunk.text
                current_chunk.end_page = max(current_chunk.end_page, next_chunk.end_page)
                current_chunk.index = next_chunk.index
                current_scores.append(next_score)
            else:
                merged_chunks.append((current_chunk, max(current_scores)))
                current_chunk = next_chunk
                current_scores = [next_score]

        merged_chunks.append((current_chunk, max(current_scores)))

        # Sort by similarity score
        merged_chunks.sort(key=lambda x: x[1], reverse=True)
        retrieved_chunks = [chunk for chunk, _ in merged_chunks]
        scores = [score for _, score in merged_chunks]

        for chunk in retrieved_chunks:
            chunk.index = None

        return scores, retrieved_chunks

    # def add_chunks(
    #     self,
    #     new_chunks: list[Chunk],
    #     retrieve_by: Callable[[Chunk], str] = lambda chunk: chunk.section_heading + " " + chunk.text
    #     if chunk.section_heading
    #     else chunk.text,
    # ):
    #     """Adds new chunks to the existing FAISS index."""
    #     if new_chunks is None or len(new_chunks) == 0:
    #         raise ValueError("New chunks cannot be None or empty.")

    #     if self.index is None:
    #         print("Index not found. Creating a new index...")
    #         self.create_index(new_chunks, retrieve_by)
    #         return

    #     # Process new chunks using `retrieve_by`
    #     new_retrieval_strings = [retrieve_by(chunk) for chunk in new_chunks]
    #     new_embeddings = embed_chunks(new_retrieval_strings, task_type="search_document")

    #     self.index.add(new_embeddings)

    #     current_length = len(self.chunks)
    #     # TODO: fix order?
    #     self.chunks.extend(new_chunks)
    #     self.retrieval_strings.extend(new_retrieval_strings)

    #     for i, chunk in enumerate(new_chunks, start=current_length):
    #         chunk.index = i

    #     print(f"Added {len(new_chunks)} new chunks. Total chunks: {len(self.chunks)}")


def _retrieve(query: str, faiss_service: FaissService) -> list[Chunk]:
    query, _ = replace_abbreviations(query)
    query_embedding = embed_chunks(query, task_type="search_query")

    _, retrieved_documents = faiss_service.search_index(query_embedding, config.top_k)
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


def _retrieve_and_rank(queries: list[str], faiss_service: FaissService) -> list[Chunk]:
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
    vignette: Vignette | None,
    question: Question | str,
    faiss_service: FaissService,
    production: bool = False,
) -> list[Chunk]:
    if production:
        if config.use_original_query_only:
            return _retrieve(question, faiss_service)
    else:
        if config.use_original_query_only:
            return _retrieve(question.get_question(), faiss_service)

        print("Using optimized query with method: ", config.optimization_method)

        system_prompt = get_optimization_prompt()
        user_prompt = create_user_question_prompt(vignette, question)
        response = generate_response(user_prompt, system_prompt)
        new_query = parse_optimized_query(response)
        # add parser because decompose will return two queries
        if isinstance(new_query, list):
            retrieved_documents = _retrieve_and_rank(new_query, faiss_service)
        else:
            retrieved_documents = _retrieve(new_query, faiss_service)

        return retrieved_documents


def tables_to_chunks(tables_dict: dict[int, list[str]]):
    return [
        Chunk(
            text=table,
            start_page=int(page_number),
            end_page=int(page_number),
            type=ChunkType.TABLE,
        )
        for page_number, tables in tables_dict.items()
        for table in tables
    ]


def retrieve_by_summarization():
    
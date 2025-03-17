import faiss
from typing import Callable

from core.embedding import embed_chunks
from settings.settings import config
from core.utils import replace_abbreviations
from domain.vignette import Vignette, Question
from domain.document import Chunk, ChunkType, Document
from core.model import generate_response
from core.question_answering import create_user_question_prompt
from prompts import HYPOTHETICAL_DOCUMENT_PROMPT, STEPBACK_PROMPT, DECOMPOSING_PROMPT, PARAPHRASING_PROMPT
from parsing import parse_with_retry, TableDescription, TableMarkdown

from langchain_core.output_parsers import BaseOutputParser


class FaissService:
    def __init__(self):
        self.index = None
        self.chunks: list[Chunk] = None
        self.retrieval_strings: list[str] = None

    def create_index(
        self,
        chunks: list[Chunk] | list[tuple[str, Chunk]],
        retrieve_by: Callable[[Chunk], str] = lambda chunk: chunk.section_heading + " " + chunk.text
        if chunk.section_heading
        else chunk.text,
    ):
        if chunks is None:
            raise ValueError("Image representations cannot be None")

        if isinstance(chunks[0], Chunk):
            self.retrieval_strings = [retrieve_by(chunk) for chunk in chunks]
        elif isinstance(chunks[0], tuple):
            print("Ignoring retrieve_by function for tuple input")
            self.retrieval_strings = [text for text, _ in chunks]
            chunks = [chunk for _, chunk in chunks]
        else:
            raise ValueError("Invalid input type for chunks")

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

    sims, retrieved_documents = faiss_service.search_index(query_embedding, config.top_k)
    print(sims)
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


def tables_to_chunks(tables_dict: dict[int, dict]):
    return [
        Chunk(
            text=table,
            start_page=int(page_number),
            end_page=int(page_number),
            section_heading=table_dict["section_heading"],
            type=ChunkType.TABLE,
        )
        for page_number, table_dict in tables_dict.items()
        for table in table_dict["content"]
    ]

    # You'll be given a table along with the context from a medical document that clinicians use to make decisions.

    # Given the table in text format and its context, provide a detailed description of the table in German. Then, include the table in markdown format to the description.

    # Do not deviate from the specified format and respond strictly in the following JSON format:

    # {
    #     "description": "<Your description and markdown here in German>"
    # }

    # Do not say anything else. Make sure the response is a valid JSON.\n

    # You'll be given a table along with the context from a medical document that clinicians use to make decisions.

    # Given the table in text format and its context, you'll write a detailed description in German. Description requires:
    # - provide a summary first
    # - then convert the table into a paragraph

    # Summary should provide an general idea what the table is about and the paragraph should cover all the information in the table.

    # Do not deviate from the specified format and respond strictly in the following JSON format:

    # {
    #     "description": "<Your summary and table in text paragraph here in German>"
    # }

    # Do not say anything else. Make sure the response is a valid JSON.\n


def retrieve_table_by_summarization(table: Chunk, document: Document):
    system_prompt = """
        You'll be given a table along with the context from a medical document that clinicians use to make decisions.

        Your task is to generate a detailed, structured, and information-rich description in German that maximizes retrieval effectiveness. Your response should include:

        1. A Clear Summary (2-3 sentences):  
        - Provide a concise yet informative overview of what the table represents.  
        - Include key medical concepts and terms clinicians might search for.  
        - Use synonyms and alternative phrasing to capture diverse query formulations.  

        2. A Detailed Table-to-Text Description:  
        - Convert the table into a well-structured, coherent paragraph.
        - Group related information together logically instead of listing data row by row.  
        - Include clear relationships between values (e.g., comparisons, trends, categories).  
        - Avoid overly mechanical repetition; use descriptive wording and natural transitions. 

        Your response must follow this JSON format strictly:  

        {
            "description": "<Your summary and table-to-text conversion in German>"
        }
        
        Do not say anything else. Make sure the response is a valid JSON.\n
    """

    user_prompt = f"""
        The context:\n{
        "\n".join(
            [
                document.get_processed_content(page_number)
                for page_number in range(table.start_page - 1, table.end_page + 1)
                if document.get_processed_content(page_number) is not None
            ]
        )
    }
        
        The table content:\n{table.text}
        """  ## start and end page are the same for tables
    response = generate_response(system_prompt, user_prompt)
    try:
        response = parse_with_retry(TableDescription, response)
        print("Response within summarization: ", response)
        return response.description
    except Exception as e:
        print("Problematic parsing:", e)
        raise e


def gather_chunks_orderly(sorted_text_chunks: list[Chunk], sorted_table_chunks: list[Chunk]) -> list[Chunk]:
    """Expects already sorted text and table chunks and merges them in the right order.

    Args:
        sorted_text_chunks (list[Chunk]): _description_
        sorted_table_chunks (list[Chunk]): _description_

    Returns:
        list[Chunk]: _description_
    """
    merged_chunks = []
    table_index = 0  # Track position in table_chunks

    for text_chunk in sorted_text_chunks:
        # Insert all table chunks that belong *before* this text chunk
        while (
            table_index < len(sorted_table_chunks)
            and sorted_table_chunks[table_index].start_page <= text_chunk.start_page
        ):
            merged_chunks.append(sorted_table_chunks[table_index])
            table_index += 1

        # Insert the text chunk (maintaining its order)
        merged_chunks.append(text_chunk)

    # Add any remaining table chunks at the end
    while table_index < len(sorted_table_chunks):
        merged_chunks.append(sorted_table_chunks[table_index])
        table_index += 1

    return merged_chunks


def describe_table_for_generation(table: Chunk, document: Document):
    system_prompt = """
    You'll be given a table along with the context from a medical document that clinicians use to make decisions.

    Given the table in text format and its context, you'll write a detailed description in German. Description requires:
    - provide a summary first
    - then convert the table into a paragraph

    Summary should provide an general idea what the table is about and the paragraph should cover all the information in the table.

    Do not deviate from the specified format and respond strictly in the following JSON format:

    {
        "description": "<Your summary and table in text paragraph here in German>"
    }

    Do not say anything else. Make sure the response is a valid JSON.\n
    """

    user_prompt = f"""
        The context:\n{
        "\n".join(
            [
                document.get_processed_content(page_number)
                for page_number in range(table.start_page - 1, table.end_page + 1)
                if document.get_processed_content(page_number) is not None
            ]
        )
    }
        
        The table content:\n{table.text}
        """  ## start and end page are the same for tables
    response = generate_response(system_prompt, user_prompt)
    try:
        response = parse_with_retry(TableDescription, response)
        print("Response within summarization: ", response)
        return response.description
    except Exception as e:
        print("Problematic parsing:", e)
        raise e


def markdown_table_for_generation(table: Chunk, document: Document):
    system_prompt = """
    You'll be given a table from a medical document that clinicians use to make decisions. The table can contain footer notes, headers, and other formatting elements.

    Given the table in text format, you'll convert it into markdown format so that it is easier to read and understand. Don't change anything in the table, just convert it into markdown format. Keep the footer notes if there are any.

    Do not deviate from the specified format and respond strictly in the following JSON format:

    {
        "markdown": "<Table in markdown format here along with footer notes if there are any>"
    }

    Do not say anything else. Make sure the response is a valid JSON.\n
    """

    user_prompt = f"""
        The table content:\n{table.text}
        """  ## start and end page are the same for tables
    response = generate_response(system_prompt, user_prompt)
    try:
        response = parse_with_retry(TableMarkdown, response)
        print("Response within summarization: ", response)
        return response.markdown
    except Exception as e:
        print("Problematic parsing:", e)
        raise e

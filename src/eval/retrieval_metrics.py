from domain.vignette import Question, Vignette
from domain.document import Chunk
from domain.evaluation import Stats, ContextRelevanceResult
from core.model import generate_response
from parsing import parse_with_retry, ContextRelevanceResultResponse
from prompts import CONTEXT_RELEVANCE_PROMPT


def does_retrieved_passage_overlap(retrieved_passage: Chunk, reference_pages: list[int]) -> bool:
    return any(
        reference_page in range(retrieved_passage.start_page, retrieved_passage.end_page + 1)
        for reference_page in reference_pages
    )


def recall(retrieved_passages: list[Chunk], reference_pages: list[int]) -> Stats:
    matched_reference_pages = set()  # To avoid double-counting matched reference pages
    total_reference_count = len(reference_pages)

    if total_reference_count == 0:
        raise ValueError("Reference pages cannot be empty")

    for retrieved_passage in retrieved_passages:
        for reference_page in reference_pages:
            if does_retrieved_passage_overlap(retrieved_passage, [reference_page]):
                matched_reference_pages.add(reference_page)

    covered_reference_count = len(matched_reference_pages)
    return covered_reference_count / total_reference_count


def precision(retrieved_passages: list[Chunk], reference_pages: list[int]) -> Stats | None:
    if not retrieved_passages:
        return None

    relevant_retrieved_count = 0
    total_retrieved_count = len(retrieved_passages)

    for retrieved_passage in retrieved_passages:
        if does_retrieved_passage_overlap(retrieved_passage, reference_pages):
            relevant_retrieved_count += 1

    return relevant_retrieved_count / total_retrieved_count


def context_relevance(
    vignette: Vignette, question: Question, generated_answer: str, retrieved_documents: list[Chunk]
) -> ContextRelevanceResult:
    context = ". ".join([doc.text for doc in retrieved_documents])
    user_prompt = f"Context: {context}\nQuestion: {question.get_question()}"

    response = generate_response(user_prompt, CONTEXT_RELEVANCE_PROMPT, max_new_tokens=4096)

    try:
        response = parse_with_retry(ContextRelevanceResultResponse, response)
        relevant_sentences = response.relevant_sentences
        irrelevant_sentences = response.irrelevant_sentences

        relevant_sentence_count = len(relevant_sentences)

        total_sentence_count = relevant_sentence_count + len(irrelevant_sentences)

        score = relevant_sentence_count / total_sentence_count if total_sentence_count > 0 else 0.0
    except Exception as e:
        print(f"Error parsing response: {e}")
        relevant_sentences = []
        irrelevant_sentences = []
        score = 0.0

    return ContextRelevanceResult(
        relevant_sentences=relevant_sentences,
        irrelevant_sentences=irrelevant_sentences,
        question_id=question.get_id(),
        score=score,
        generated_answer=generated_answer,
    )

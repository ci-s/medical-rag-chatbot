from statistics import mean
from typing import Literal
import json

from services.retrieval import FaissService, retrieve, retrieve_and_return_optimized_query
from core.model import generate_response
from core.generation import create_question_prompt_w_docs
from domain.evaluation import Feedback
from domain.document import Chunk, Document
from settings import VIGNETTE_COLLECTION, get_page_types
from .generation_metrics import llm_as_a_judge, faithfulness, answer_relevance, get_generated_flowchart_page_description
from .retrieval_metrics import context_relevance, recall, precision
from parsing import Answer, parse_with_retry, ReasoningAnswer, ThinkingAnswer
from settings.settings import config, settings
from core.chunking import tables_to_chunks

with open(settings.table_texts_path, "r") as file:
    tables = json.load(file)
table_chunks = tables_to_chunks(tables)


class RAGASResult:
    def __init__(
        self,
        question_id: int,
        retrieved_documents: list[Chunk],
        generated_answer: dict | str,
        llm_as_a_judge_param: Feedback | None,
        faithfulness_param: Feedback | None,
        answer_relevance_param: Feedback | None,
        context_relevance_param: Feedback | None,
        optimized_query: str = "",
        reference_pages: list[int] = [],
        retrieval_recall: float = None,
        retrieval_precision: float = None,
    ):
        self.question_id = question_id
        self.retrieved_documents = retrieved_documents
        self.generated_answer = generated_answer
        self.llm_as_judge = llm_as_a_judge_param
        self.faithfulness = faithfulness_param
        self.answer_relevance = answer_relevance_param
        self.context_relevance = context_relevance_param
        self.optimized_query = optimized_query
        self.reference_pages = reference_pages
        self.retrieval_recall = retrieval_recall
        self.retrieval_precision = retrieval_precision

    def to_dict(self):
        return {
            "question_id": self.question_id,
            "retrieved_documents": [doc.to_dict() for doc in self.retrieved_documents],
            "generated_answer": self.generated_answer,
            "optimized_query": self.optimized_query,
            "reference_pages": list(self.reference_pages),
            "llm_as_judge": self.llm_as_judge.to_dict() if self.llm_as_judge else None,
            "faithfulness": self.faithfulness.to_dict() if self.faithfulness else None,
            "answer_relevance": self.answer_relevance.to_dict() if self.answer_relevance else None,
            "context_relevance": self.context_relevance.to_dict() if self.context_relevance else None,
            "retrieval_recall": self.retrieval_recall,
            "retrieval_precision": self.retrieval_precision,
        }


class FeedbackResult(Feedback):
    def __init__(
        self,
        feedback: str,
        question_id: int,
        score: float,
        generated_answer: str = "",
        reference_pages: list[int] = [],
        retrieved_documents: list[Chunk] = [],
    ):
        super().__init__(question_id=question_id, feedback=feedback, score=score, generated_answer=generated_answer)
        self.retrieved_documents = retrieved_documents
        self.reference_pages = reference_pages

    def to_dict(self):
        return {
            "question_id": self.question_id,
            "feedback": self.text,
            "score": self.score,
            "generated_answer": self.generated_answer,
            "reference_pages": list(self.reference_pages),
            "retrieved_documents": [doc.to_dict() for doc in self.retrieved_documents],
        }


def evaluate_single(
    vignette_id: int,
    question_id: int,
    faiss_service: FaissService,
    document: Document,
    use_references_directly: bool = False,
) -> FeedbackResult:
    vignette = VIGNETTE_COLLECTION.get_vignette_by_id(vignette_id)
    questions = vignette.get_questions()
    print(f"Questions in vignette {vignette_id}: {len(questions)}")
    if questions is None:
        print(f"Vignette with id {vignette_id} not found")
        return None

    question = vignette.get_question(question_id)
    if not use_references_directly:
        retrieved_documents = retrieve(vignette, question, faiss_service)
    else:
        retrieved_documents = [
            Chunk(
                text=document.get_processed_content(page_number),
                start_page=page_number,
                end_page=page_number,
                index=None,
            )
            if document.get_processed_content(page_number)
            else Chunk(
                text=get_generated_flowchart_page_description(page_number),
                start_page=page_number,
                end_page=page_number,
                index=None,
            )
            for page_number in question.get_reference_pages()
        ]

        _, _, table_pages, _ = get_page_types()
        for page_number in question.get_reference_pages():
            if page_number in table_pages:
                print(f"Page {page_number} is a table page")
                for chunk in table_chunks:
                    if chunk.start_page == page_number:
                        retrieved_documents.append(chunk)
                        break

    print(f"# of Retrieved documents: {len(retrieved_documents)}")
    print(f"Retrieved docs: {[doc.__str__() for doc in retrieved_documents]}")
    system_prompt, user_prompt = create_question_prompt_w_docs(retrieved_documents, vignette, question)
    generated_answer = generate_response(user_prompt, system_prompt)
    if config.reasoning:
        generated_answer = parse_with_retry(ReasoningAnswer, generated_answer)
    elif config.thinking:
        generated_answer = parse_with_retry(ThinkingAnswer, generated_answer)
    else:
        generated_answer = parse_with_retry(Answer, generated_answer)

    feedback = llm_as_a_judge(vignette, question, generated_answer.answer, document)
    return FeedbackResult(
        feedback=feedback.text,
        question_id=question_id,
        reference_pages=question.get_reference_pages(),
        score=feedback.score,
        generated_answer=generated_answer.answer,
        retrieved_documents=retrieved_documents,
    )


def evaluate_source(
    source: Literal["Handbuch", "Antibiotika"],
    faiss_service: FaissService,
    document: Document,
    use_references_directly: bool = False,
) -> tuple[int, list[FeedbackResult]]:
    all_feedbacks = []

    for vignette in VIGNETTE_COLLECTION.get_vignettes():
        for question in vignette.get_questions():
            if question.get_source() != source:
                continue

            all_feedbacks.append(
                evaluate_single(vignette.get_id(), question.get_id(), faiss_service, document, use_references_directly)
            )

    print(f"Questions from {source}: {len([all_feedbacks for s in all_feedbacks if s is not None])}")

    try:
        avg_score = mean(
            [float(feedback_result.score) for feedback_result in all_feedbacks if feedback_result.score is not None]
        )
    except Exception as e:
        print("Trouble calculating average score: ", e)
        avg_score = 0
    # Add validation to score for integer between 1 and 5
    return avg_score, all_feedbacks


def evaluate_single_w_ragas(
    vignette_id: int,
    question_id: int,
    faiss_service: FaissService,
    document: Document,
    use_references_directly: bool = False,
) -> RAGASResult:
    vignette = VIGNETTE_COLLECTION.get_vignette_by_id(vignette_id)
    questions = vignette.get_questions()
    print(f"Questions in vignette {vignette_id}: {len(questions)}")
    if questions is None:
        print(f"Vignette with id {vignette_id} not found")
        return None

    question = vignette.get_question(question_id)

    if not use_references_directly:
        retrieved_documents, optimized_query = retrieve_and_return_optimized_query(vignette, question, faiss_service)
    else:
        retrieved_documents = [
            Chunk(
                text=document.get_processed_content(page_number),
                start_page=page_number,
                end_page=page_number,
                index=None,
            )
            if document.get_processed_content(page_number)
            else Chunk(
                text=get_generated_flowchart_page_description(page_number),
                start_page=page_number,
                end_page=page_number,
                index=None,
            )
            for page_number in question.get_reference_pages()
        ]
        optimized_query = None

    retrieval_recall = recall(retrieved_documents, question.get_reference_pages())
    retrieval_precision = precision(retrieved_documents, question.get_reference_pages())

    system_prompt, user_prompt = create_question_prompt_w_docs(retrieved_documents, vignette, question)

    generated_answer = generate_response(user_prompt, system_prompt)
    if config.reasoning:
        generated_answer = parse_with_retry(ReasoningAnswer, generated_answer)
        answer_dict = {
            "answer": generated_answer.answer,
            "reasoning": generated_answer.reasoning,
        }
    elif config.thinking:
        generated_answer = parse_with_retry(ThinkingAnswer, generated_answer)
        answer_dict = {
            "answer": generated_answer.answer,
            "reasoning": generated_answer.thinking,
        }
    else:
        generated_answer = parse_with_retry(Answer, generated_answer)
        answer_dict = {
            "answer": generated_answer.answer,
        }

    return RAGASResult(
        question_id=question_id,
        generated_answer=answer_dict,
        optimized_query=optimized_query,
        reference_pages=question.get_reference_pages(),
        llm_as_a_judge_param=llm_as_a_judge(
            vignette, question, generated_answer.answer, document
        ),  # this requires real document
        faithfulness_param=faithfulness(vignette, question, generated_answer.answer, retrieved_documents),
        answer_relevance_param=answer_relevance(vignette, question, generated_answer.answer, retrieved_documents),
        context_relevance_param=None,  # context_relevance(vignette, question, generated_answer.answer, retrieved_documents),
        retrieved_documents=retrieved_documents,
        retrieval_recall=retrieval_recall,
        retrieval_precision=retrieval_precision,
    )


def compute_average_scores(all_feedbacks: list[RAGASResult], score_keys: list[str]) -> dict:
    """
    Computes average scores for given score keys in all_feedbacks.

    Parameters:
        all_feedbacks (list): List of feedback objects containing different score categories.
        score_keys (list): List of score keys to compute averages for.

    Returns:
        dict: Dictionary with average scores for each key.
    """
    avg_scores = {}

    for key in score_keys:
        scores = [
            float(getattr(ragas_result, key).score)
            for ragas_result in all_feedbacks
            if getattr(ragas_result, key) is not None
        ]
        avg_scores[key] = mean(scores) if scores else 0.0  # Avoid StatisticsError

    return avg_scores


def evaluate_ragas(
    source: Literal["Handbuch", "Antibiotika"],
    faiss_service: FaissService,
    document: Document,
    use_references_directly: bool = False,
) -> tuple[dict, int, int, list[RAGASResult]]:
    all_feedbacks = []

    try:
        for vignette in VIGNETTE_COLLECTION.get_vignettes():
            for question in vignette.get_questions():
                if question.get_source() != source:
                    continue
                all_feedbacks.append(
                    evaluate_single_w_ragas(
                        vignette.get_id(), question.get_id(), faiss_service, document, use_references_directly
                    )
                )

        print(f"Questions from {source}: {len([all_feedbacks for s in all_feedbacks if s is not None])}")

        score_keys = ["llm_as_judge", "faithfulness", "answer_relevance", "context_relevance"]
        avg_scores = compute_average_scores(all_feedbacks, score_keys)
        avg_recall = mean(
            [
                float(eval_result.retrieval_recall)
                for eval_result in all_feedbacks
                if eval_result.retrieval_recall is not None
            ]
        )
        avg_precision = mean(
            [
                float(eval_result.retrieval_precision)
                for eval_result in all_feedbacks
                if eval_result.retrieval_precision is not None
            ]
        )
    except Exception as e:
        print("Trouble calculating average scores: ", e)
        avg_recall = 0.0
        avg_precision = 0.0
        avg_scores = None
    return avg_scores, avg_recall, avg_precision, all_feedbacks


def evaluate_ragas_qids(
    source: Literal["Handbuch", "Antibiotika"],
    faiss_service: FaissService,
    question_ids: list[int],
) -> tuple[dict, int, int, list[RAGASResult]]:
    all_feedbacks = []

    for vignette in VIGNETTE_COLLECTION.get_vignettes():
        for question in vignette.get_questions():
            if question.get_id() not in question_ids:
                continue

            all_feedbacks.append(evaluate_single_w_ragas(vignette.get_id(), question.get_id(), faiss_service))

    print(f"Questions from {source}: {len([all_feedbacks for s in all_feedbacks if s is not None])}")

    score_keys = ["llm_as_judge", "faithfulness", "answer_relevance", "context_relevance"]
    try:
        avg_scores = compute_average_scores(all_feedbacks, score_keys)
        avg_recall = mean(
            [
                float(eval_result.retrieval_recall)
                for eval_result in all_feedbacks
                if eval_result.retrieval_recall is not None
            ]
        )
        avg_precision = mean(
            [
                float(eval_result.retrieval_precision)
                for eval_result in all_feedbacks
                if eval_result.retrieval_precision is not None
            ]
        )
    except Exception as e:
        print("Trouble calculating average scores: ", e)
        avg_recall = 0.0
        avg_precision = 0.0
        avg_scores = None
    return avg_scores, avg_recall, avg_precision, all_feedbacks

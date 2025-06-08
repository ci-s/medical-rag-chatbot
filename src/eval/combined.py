from .retrieval_metrics import recall, precision
from services.retrieval import FaissService, retrieve_and_return_optimized_query
from domain.document import Document
from core.model import generate_response
from core.generation import create_question_prompt_w_docs
from .generation_metrics import llm_as_a_judge
from domain.vignette import Vignette, Question
from domain.document import Chunk
from parsing import Answer, parse_with_retry, ReasoningAnswer, ThinkingAnswer
from settings.settings import config


class EvalResult:
    """
    Class to hold the evaluation results.
    """

    def __init__(
        self,
        feedback: str,
        question_id: int,
        score: float,
        generated_answer: str,
        optimized_query: str | list[str],
        reference_pages: list,
        retrieved_documents: list[Chunk],
        retrieval_recall: float,
        retrieval_precision: float,
    ):
        self.feedback = feedback
        self.question_id = question_id
        self.score = score
        self.generated_answer = generated_answer
        self.optimized_query = optimized_query
        self.reference_pages = reference_pages
        self.retrieved_documents = retrieved_documents
        self.retrieval_recall = retrieval_recall
        self.retrieval_precision = retrieval_precision

    def to_dict(self):
        """
        Convert the evaluation result to a dictionary format.
        """
        return {
            "feedback": self.feedback,
            "question_id": self.question_id,
            "score": self.score,
            "generated_answer": self.generated_answer,
            "optimized_query": self.optimized_query,
            "reference_pages": list(self.reference_pages),
            "retrieved_documents": [doc.to_dict() for doc in self.retrieved_documents],
            "retrieval_recall": self.retrieval_recall,
            "retrieval_precision": self.retrieval_precision,
        }


def evaluate_single_combined(
    vignette: Vignette,
    question: Question,
    faiss_service: FaissService,
    document: Document,
) -> EvalResult:
    """
    Evaluate both retrieval and generation for a single question.
    """

    retrieved_documents, optimized_query = retrieve_and_return_optimized_query(vignette, question, faiss_service)
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
    generation_feedback = llm_as_a_judge(vignette, question, generated_answer.answer, document)

    if generation_feedback.score < 1 or generation_feedback.score > 5:
        print(f"Invalid score: {generation_feedback.score} for question {question.get_id()}. Trying again.")
        generation_feedback = llm_as_a_judge(vignette, question, generated_answer.answer, document)

    return EvalResult(
        feedback=generation_feedback.text,
        question_id=question.get_id(),
        score=generation_feedback.score,
        generated_answer=answer_dict,
        optimized_query=optimized_query,
        reference_pages=question.get_reference_pages(),
        retrieved_documents=retrieved_documents,
        retrieval_recall=retrieval_recall,
        retrieval_precision=retrieval_precision,
    )

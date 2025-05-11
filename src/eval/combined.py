from .retrieval_metrics import recall, precision
from services.retrieval import FaissService, retrieve
from domain.document import Document
from core.model import generate_response
from core.generation import create_question_prompt_w_docs
from .generation_metrics import llm_as_a_judge
from domain.vignette import Vignette, Question


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
        reference_pages: list,
        retrieved_documents: list,
        retrieval_recall: float,
        retrieval_precision: float,
    ):
        self.feedback = feedback
        self.question_id = question_id
        self.score = score
        self.generated_answer = generated_answer
        self.reference_pages = reference_pages
        self.retrieved_documents = retrieved_documents
        self.retrieval_recall = retrieval_recall
        self.retrieval_precision = retrieval_precision


def evaluate_single_combined(
    vignette: Vignette,
    question: Question,
    faiss_service: FaissService,
    document: Document,
) -> EvalResult:
    """
    Evaluate both retrieval and generation for a single question.
    """

    retrieved_documents = retrieve(vignette, question, faiss_service)
    retrieval_recall = recall(retrieved_documents, question.get_reference_pages())
    retrieval_precision = precision(retrieved_documents, question.get_reference_pages())

    system_prompt, user_prompt = create_question_prompt_w_docs(retrieved_documents, vignette, question)
    generated_answer = generate_response(system_prompt, user_prompt)
    generation_feedback = llm_as_a_judge(vignette, question, generated_answer, document)

    return EvalResult(
        feedback=generation_feedback.text,
        question_id=question.get_id(),
        score=generation_feedback.score,
        generated_answer=generated_answer,
        reference_pages=question.get_reference_pages(),
        retrieved_documents=retrieved_documents,
        retrieval_recall=retrieval_recall,
        retrieval_precision=retrieval_precision,
    )

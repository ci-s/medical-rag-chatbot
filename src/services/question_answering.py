from typing import NamedTuple

from core.generation import create_question_prompt_w_docs_prod
from parsing import parse_with_retry, Answer
from services.retrieval import FaissService, _retrieve
from core.model import generate_response
from domain.document import Chunk


class RAGAnswer(NamedTuple):
    generated_answer: str
    references: dict[int, str] # page number, chunk type i.e. 15: "table"

def generate_rag_answer(question: str, faiss_service: FaissService) -> Answer:    
    retrieved_documents: list[Chunk] = _retrieve(question, faiss_service)
    system_prompt, user_prompt = create_question_prompt_w_docs_prod(retrieved_documents, question)
    generated_answer = generate_response(system_prompt, user_prompt)
    generated_answer = parse_with_retry(Answer, generated_answer)
    
    references = {retrieved_doc.start_page: retrieved_doc.type for retrieved_doc in retrieved_documents}
    
    return RAGAnswer(generated_answer.answer, references)

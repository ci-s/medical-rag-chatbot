from typing import NamedTuple
from enum import Enum

from core.generation import create_question_prompt_w_docs_prod
from parsing import parse_with_retry, Answer, FollowUpQuestion
from services.retrieval import FaissService, _retrieve
from core.model import generate_response
from domain.document import Chunk


class RAGAnswer(NamedTuple):
    generated_answer: str
    references: dict[int, str]  # page number, chunk type i.e. 15: "table"


def generate_rag_answer(question: str, faiss_service: FaissService) -> RAGAnswer:
    retrieved_documents: list[Chunk] = _retrieve(question, faiss_service)
    system_prompt, user_prompt = create_question_prompt_w_docs_prod(retrieved_documents, question)
    generated_answer = generate_response(system_prompt, user_prompt)
    generated_answer = parse_with_retry(Answer, generated_answer)

    references = {retrieved_doc.start_page: retrieved_doc.type for retrieved_doc in retrieved_documents}
    return RAGAnswer(generated_answer.answer, references)


def generate_followup_questions_if_needed(question: str, faiss_service: FaissService) -> FollowUpQuestion | None:
    retrieved_documents: list[Chunk] = _retrieve(question, faiss_service)

    system_prompt = """
        You are a medical assistant specialized in supporting clinicians during decision making.

        You are given a user’s clinical question and the relevant background information about the patient.

        Your task is:
            1.	Determine if further information is needed to safely and accurately answer the question.
            2.	If additional clarification is needed, generate:
            •	A clear and specific follow-up question in German.
            •	Dropdown options (3–5 options) that the clinician can choose from (also in German).
            3.	If no additional clarification is needed, respond that no follow-up is required.
            4.  Only ask for information that is not already provided in the background.
            5.  Focus on decision-relevant missing variables like vital signs, contraindications, therapy windows, imaging findings.

        **Example Format:**
        Background: {{background}}
        Question: {{question}}

        {
        "follow_up_required": true/false,
        "follow_up_question": "string, only if follow_up_required is true otherwise empty",
        "options": ["option 1", "option 2", "option 3", "..."] // only if follow_up_required is true otherwise empty
        }
        
        Do not say anything else. Make sure the response is a valid JSON.
    """

    _, user_prompt = create_question_prompt_w_docs_prod(retrieved_documents, question)
    response = generate_response(system_prompt, user_prompt)
    followup_question: FollowUpQuestion = parse_with_retry(FollowUpQuestion, response)
    if followup_question.follow_up_required:
        return followup_question
    else:
        None


class ConversationState(Enum):
    GENERATING = 0
    WAITING_FOR_USER = 1
    FINISHED = 2


class Conversation:
    def __init__(self, id: str, state: ConversationState, text: str, details: list, answers: list):
        self.id = id
        self.state = state
        self.text = text
        self.details = details
        self.answers = answers

    def to_dict(self):
        return {
            "id": self.id,
            "state": self.state.value,
            "text": self.text,
            "details": self.details,
            "answers": self.answers,  # [answer.to_dict() for answer in self.answers]
        }


class ConversationService:
    last_conversation_id = 0
    last_detail_id = 0
    conversations = dict()
    document_index: FaissService

    async def create_conversation(self, question: str, previous_question: str) -> tuple[str, str | None]:
        self.last_conversation_id += 1
        id = self.last_conversation_id

        quick_answer = {
            "type": "quick",
            "text": "",
            # "reference": {"type": "pdf", "document": "handbuch.pdf", "page": 12},
        }

        conversation = Conversation(
            id=id, state=ConversationState.GENERATING, text=question, details=[], answers=[quick_answer]
        )
        conversation = conversation.to_dict()
        self.conversations[str(id)] = conversation

        def on_update(generated_answer: str | None, references: dict | None, form_detail: dict = None):
            if generated_answer:
                quick_answer["text"] = generated_answer
            if references:
                quick_answer["references"] = [
                    {"type": "pdf", "document": "handbuch.pdf", "page": page} for page, chunk_type in references.items()
                ]
            if form_detail:
                conversation["details"] = [form_detail]

            if not generated_answer and not form_detail:
                generated_answer = "Sorry there has been a problem with your request."
                quick_answer["text"] = generated_answer

        followup_question = generate_followup_questions_if_needed(question, self.document_index)

        if followup_question is not None:
            self.last_detail_id += 1
            form_detail = {
                "id": self.last_detail_id,  # Ideally make this dynamic later
                "type": "form",
                "template": {
                    "text": followup_question.follow_up_question,
                    "fields": {"art": {"type": "select", "label": "Select", "options": followup_question.options}},
                },
            }
            on_update(None, None, form_detail)
        else:
            # No follow-up needed → generate final answer immediately
            generated_answer, references = generate_rag_answer(question, self.document_index)
            on_update(generated_answer, references)

        if conversation["details"]:
            conversation["state"] = ConversationState.WAITING_FOR_USER.value
        else:
            conversation["state"] = ConversationState.FINISHED.value
        return id, None

    def get_conversation(self, id: str) -> dict | None:
        if id in self.conversations:
            return self.conversations[id]
        else:
            return None

    def update_conversation_detail(self, conversation_id: str, detail_id: str, value: any) -> dict | None:
        conversation = self.get_conversation(conversation_id)
        for detail in conversation["details"]:
            if "template" in detail:
                if detail["id"] == int(detail_id):
                    # Update the value in the details
                    detail["template"]["selected_value"] = value
                    break

        original_question = conversation["text"]
        augmented_question = self.augment_question_with_details(original_question, conversation["details"])
        generated_answer, references = generate_rag_answer(augmented_question, self.document_index)

        if generated_answer:
            print("will update answer text")
            conversation["answers"][0]["text"] = generated_answer
        if references:
            conversation["answers"][0]["references"] = [
                {"type": "pdf", "document": "handbuch.pdf", "page": page} for page, chunk_type in references.items()
            ]
        conversation["state"] = ConversationState.FINISHED.value
        return conversation

    def augment_question_with_details(self, original_question: str, details: list) -> str:
        """
        Insert details into the original question text to enrich it.
        """
        detail_texts = []
        for detail in details:
            print("detail = ", detail)
            if "template" in detail:
                if "selected_value" in detail["template"]:
                    selected_value = detail["template"]["selected_value"]
                    if selected_value:
                        print("will update selected value")
                        # E.g., "Art: Kompression"
                        detail_texts.append(f"{detail['template']['text']} {selected_value}/n")

        if detail_texts:
            # E.g., "Original Question (Art: Kompression)"
            augmented = f"{original_question} ({'; '.join(detail_texts)})"
            print(f"Augmented question: {augmented}")
            return augmented
        else:
            print("No details to augment the question: {detail_texts}")
            return original_question

    def to_json(self):
        return {"last_conversation_id": self.last_conversation_id, "conversations": self.conversations}

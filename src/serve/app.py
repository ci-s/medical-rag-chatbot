from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import uvicorn

from pydantic import BaseModel
import sys
import os
from typing import Any

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from settings import settings, config

# Setup
from core.chunking import load_saved_chunks
from services.retrieval import FaissService, _retrieve, reorder_flowchart_chunks
from enum import Enum
from services.question_answering import generate_rag_answer

_all_chunks = load_saved_chunks(config.saved_chunks_path)
all_chunks = reorder_flowchart_chunks(_all_chunks)
faiss_service = FaissService()
faiss_service.create_index(all_chunks)


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
            "answers": self.answers # [answer.to_dict() for answer in self.answers]
        }
        
class DetailUpdatePayload(BaseModel):
    value: Any # Use Any to allow different types of detail values


async def generate_quick_answer(question: str, on_update, on_finish):
    generated_answer, references = generate_rag_answer(question, faiss_service)
    on_update(generated_answer, references)

    on_finish()

class ConversationService:
    last_conversation_id = 0
    conversations = dict()
    
    async def create_conversation(self, question: str, previous_question: str) -> tuple[str, str | None]:
        self.last_conversation_id += 1
        id = self.last_conversation_id

        quick_answer = {
            "type": "quick",
            "text": "",
            "reference": {
            "type": "pdf",
            "document": "handbuch.pdf",
            "page": 12
            }
        }
        
        conversation = Conversation(
            id=id,
            state=ConversationState.GENERATING,
            text=question,
            details=[],
            answers=[quick_answer]
        )
        conversation = conversation.to_dict()
        self.conversations[str(id)] = conversation
        
        
        def on_update(generated_answer: str, references: dict):
            quick_answer["text"] = generated_answer
            quick_answer["references"] = [
                {
                    "type": "pdf",
                    "document": "handbuch.pdf",
                    "page": page
                }
                for page, chunk_type in references.items()
            ]
        
        def on_finish():
            conversation["state"] = ConversationState.FINISHED.value
            
        await generate_quick_answer(question, on_update, on_finish)
        return id, None
    
    def get_conversation(self, id: str) -> dict | None:
        if id in self.conversations:
            return self.conversations[id]
        else:
            return None
        
    async def update_conversation_detail(self, conversation_id: str, detail_id: str, value: Any) -> dict:
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Find the required detail by id
        detail_index = -1
        for i, detail in enumerate(conversation.required_details):
            if detail.get("id") == detail_id:
                detail_index = i
                break
        
        if detail_index == -1:
             raise HTTPException(status_code=404, detail=f"Detail with id '{detail_id}' not found or not required for this conversation")

        # Store the submitted value (optional, depends on your logic)
        conversation.details.append({"id": detail_id, "value": value})

        # Remove the detail from required_details as it's now fulfilled
        fulfilled_detail = conversation.required_details.pop(detail_index)

        # --- Application Logic Placeholder ---
        # TODO: Add logic here to process the submitted value.
        # This might involve:
        # 1. Calling another service (e.g., RAG again with the new info).
        # 2. Updating the conversation's answers.
        # 3. Adding new required_details if more input is needed.
        # 4. Changing the conversation state.
        
        # Example: If no more details are required, finish the conversation
        if not conversation.required_details:
             conversation.state = ConversationState.FINISHED
        else:
             conversation.state = ConversationState.WAITING_FOR_USER # Or keep waiting if more details needed

        # --- End Placeholder ---

        return conversation.to_dict()
    def to_json(self):
        return {
            "last_conversation_id": self.last_conversation_id,
            "conversations": self.conversations
        }

conversation_service = ConversationService()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # allow POST, GET, OPTIONS, etc.
    allow_headers=["*"],  # allow all headers
)


@app.get("/")
def read_root():
    return {"message": "HellooooWorld"}


class Question(BaseModel):
    text: str


@app.post("/retrieve")
async def retrieve_documents(request: Request):
    question = await request.json()
    question = question["text"]
    docs = _retrieve(question, faiss_service)
    return JSONResponse({"message": [doc.to_dict() for doc in docs]})


@app.post("/answer")
async def answer_question(request: Request):
    question = await request.json()
    question = question["text"]
    return {"answer": generate_rag_answer(question, faiss_service)}

@app.post("/conversations")
async def create_conversation(request: Request):
    question = await request.json()
    question = question["text"]
    print(f"question = {question}")
    if question:
        id, _ = await conversation_service.create_conversation(question, "")

        return JSONResponse(content=[id, None], status_code=200)

    return JSONResponse(content=[], status_code=400)

@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id):
    conversation = conversation_service.get_conversation(conversation_id)

    if conversation is not None:
        return JSONResponse(content=conversation if not None else {}, status_code=200)
    else:
        return JSONResponse(content={}, status_code=404)

@app.post("/conversations/{conversation_id}/detail/{detail_id}")
async def update_detail(conversation_id: str, detail_id: str, payload: DetailUpdatePayload):
    try:
        updated_conversation_state = await conversation_service.update_conversation_detail(
            conversation_id, detail_id, payload.value
        )
        return JSONResponse(content=updated_conversation_state, status_code=200)
    except HTTPException as e:
        # Forward HTTP exceptions (like 404 Not Found) from the service
        return JSONResponse(content={"error": e.detail}, status_code=e.status_code)
    except Exception as e:
        # Catch unexpected errors
        print(f"Error updating detail: {e}") # Log the error
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)
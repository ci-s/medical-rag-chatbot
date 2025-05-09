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
from services.question_answering import generate_rag_answer, generate_followup_questions_if_needed, ConversationService

_all_chunks = load_saved_chunks(config.saved_chunks_path)
all_chunks = reorder_flowchart_chunks(_all_chunks)
faiss_service = FaissService()
faiss_service.create_index(all_chunks)


class DetailUpdatePayload(BaseModel):
    data: Any  # Use Any to allow different types of detail values


conversation_service = ConversationService()
ConversationService.document_index = faiss_service

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


@app.post("/conversations/{conversation_id}/details/{detail_id}")
async def update_detail(conversation_id: str, detail_id: str, payload: DetailUpdatePayload):
    selected_value = payload.data["art"]
    print(f"Conversation ID: {conversation_id}, Detail ID: {detail_id}, Selected Value: {selected_value}")
    try:
        updated_conversation_state = conversation_service.update_conversation_detail(
            conversation_id, detail_id, selected_value
        )
        return JSONResponse(content=updated_conversation_state, status_code=200)
    except HTTPException as e:
        # Forward HTTP exceptions (like 404 Not Found) from the service
        return JSONResponse(content={"error": e.detail}, status_code=e.status_code)
    except Exception as e:
        # Catch unexpected errors
        print(f"Error updating detail: {e}")  # Log the error
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)

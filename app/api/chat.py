from fastapi import APIRouter
from app.models.schemas import ChatRequest, ChatResponse
from app.services import llm

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    return await llm.chat(request)

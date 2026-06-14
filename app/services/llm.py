from app.config import get_settings
from app.models.schemas import ChatRequest, ChatResponse

MOCK_REPLIES = [
    "Отличный выбор стиля! Для {style} я рекомендую использовать нейтральные тона и натуральные материалы.",
    "Хороший вопрос! При бюджете {budget} сомов советую начать с покраски стен — это даст максимальный эффект за минимальные деньги.",
    "Для небольшой комнаты важно использовать многофункциональную мебель: диваны с ящиками для хранения, откидные столы.",
    "Свет играет ключевую роль в дизайне. Слоёное освещение (общее + акцентное + декоративное) преобразит любое пространство.",
    "Зеркала визуально увеличивают комнату. Разместите большое зеркало напротив окна — это удвоит естественный свет.",
]

_SYSTEM_PROMPT = (
    "You are roomi.ai — an expert AI interior design assistant. "
    "Help users redesign their living spaces with practical, affordable advice. "
    "Focus on Central Asian market (Kyrgyzstan, Kazakhstan, Uzbekistan). "
    "Recommend locally available furniture and materials. "
    "Keep answers concise, friendly, and actionable. "
    "Respond in the same language the user writes in (Russian or Kyrgyz preferred)."
)


async def chat(request: ChatRequest) -> ChatResponse:
    """Groq first (fast), then any OpenAI-compatible endpoint, then mock."""
    settings = get_settings()

    if settings.use_groq:
        try:
            return await _groq_reply(request, settings)
        except Exception as e:
            print(f"[chat] groq error: {e}")

    if settings.use_openai:
        try:
            return await _openai_reply(request, settings)
        except Exception as e:
            print(f"[chat] openai error: {e}")

    return _mock_reply(request)


def _build_messages(request: ChatRequest) -> list[dict]:
    system_prompt = _SYSTEM_PROMPT
    if request.context:
        system_prompt += f"\n\nUser context: {request.context}"

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for msg in request.history[-6:]:  # last 3 turns
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": request.message})
    return messages


def _mock_reply(request: ChatRequest) -> ChatResponse:
    import random
    reply = random.choice(MOCK_REPLIES)
    reply = reply.replace("{style}", "минимализм").replace("{budget}", "50,000")
    return ChatResponse(reply=reply, is_mock=True)


async def _groq_reply(request: ChatRequest, settings) -> ChatResponse:
    from groq import AsyncGroq

    client = AsyncGroq(api_key=settings.groq_api_key)
    response = await client.chat.completions.create(
        model=settings.groq_model,
        messages=_build_messages(request),
        max_tokens=400,
        temperature=0.7,
    )
    return ChatResponse(reply=response.choices[0].message.content or "", is_mock=False)


async def _openai_reply(request: ChatRequest, settings) -> ChatResponse:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
    )
    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=_build_messages(request),
        max_tokens=400,
        temperature=0.7,
    )
    return ChatResponse(reply=response.choices[0].message.content or "", is_mock=False)

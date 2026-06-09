from pydantic import BaseModel
from typing import Optional


# ── Redesign ──────────────────────────────────────

class RedesignRequest(BaseModel):
    style: str = "minimal"
    budget: Optional[int] = None   # KGS
    room_type: str = "bedroom"     # bedroom | living | kitchen | office


class RedesignResponse(BaseModel):
    design_id: str
    original_url: str
    result_url: str
    style: str
    is_mock: bool = False


# ── Products ──────────────────────────────────────

class Product(BaseModel):
    id: str
    name: str
    category: str
    price: int
    currency: str
    image: str
    shop: str
    url: str
    priority: int


class ProductsResponse(BaseModel):
    style: str
    budget: Optional[int]
    products: list[Product]
    total_price: int
    currency: str
    is_mock: bool = True


# ── Plan ──────────────────────────────────────────

class PlanWeek(BaseModel):
    week: int
    title: str
    tasks: list[str]
    estimated_cost: int
    currency: str
    icon: str


class PlanResponse(BaseModel):
    design_id: str
    total_weeks: int
    total_cost: int
    currency: str
    weeks: list[PlanWeek]
    is_mock: bool = True


# ── Chat ──────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str   # user | assistant
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    context: Optional[str] = None   # e.g. "minimal style, 50000 KGS budget"


class ChatResponse(BaseModel):
    reply: str
    is_mock: bool = False

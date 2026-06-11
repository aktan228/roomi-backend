from pydantic import BaseModel
from typing import Optional


# ── Redesign ──────────────────────────────────────

class RedesignRequest(BaseModel):
    style: str = "minimal"
    budget: Optional[int] = None   # KGS
    room_type: str = "bedroom"     # bedroom | living | kitchen | office
    preferences: Optional[str] = None   # free-text user wishes


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


# ── Perception (room analysis) ────────────────────

class RoomObject(BaseModel):
    label: str                 # sofa | bed | window | door | radiator | ...
    confidence: float
    bbox: list[int]            # [x1, y1, x2, y2] in pixels
    area_ratio: float          # fraction of the image area
    suggestion: str = "keep"   # keep | replace | remove


class RoomSurfaces(BaseModel):
    wall_m2: float
    floor_m2: float
    ceiling_m2: float


class RoomAnalysis(BaseModel):
    room_type: str
    est_area_m2: float
    ceiling_height_m: float
    objects: list[RoomObject] = []
    surfaces: RoomSurfaces
    palette: list[str] = []    # hex colors of the current room
    lighting: str = "bright"   # dim | bright
    is_stub: bool = False       # True when models unavailable → heuristic only


# ── Design brief (LLM output that drives everything) ─

class ShoppingItem(BaseModel):
    item: str
    category: str
    why: str
    est_price: int
    priority: int = 2


class PlanPhase(BaseModel):
    phase: int
    title: str
    tasks: list[str]
    est_cost: int


class DesignBrief(BaseModel):
    sd_prompt: str
    negative_prompt: str = ""
    inpaint_targets: list[str] = []   # e.g. ["wall", "floor"]
    shopping_list: list[ShoppingItem] = []
    reno_plan: list[PlanPhase] = []
    is_stub: bool = False


# ── Full redesign response (composed at the route) ──

class RedesignFullResponse(BaseModel):
    design_id: str
    original_url: str
    result_url: str
    style: str
    is_mock: bool = False
    analysis: Optional[RoomAnalysis] = None
    products: Optional[ProductsResponse] = None
    plan: Optional[PlanResponse] = None

import asyncio
import uuid

from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Any

from app.models.schemas import RedesignFullResponse, ProductsResponse, PlanResponse, PlanWeek
from app.services import image_gen, job_store
from app.services.perception import analyze_room
from app.services.design_brief import build_brief

router = APIRouter()


# ── Start job ─────────────────────────────────────────────────────────────────

class JobStarted(BaseModel):
    job_id: str
    status: str = "queued"


@router.post("/redesign", response_model=JobStarted)
async def redesign_room(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    style: str = Form("minimal"),
    room_type: str = Form("bedroom"),
    budget: int | None = Form(None),
    preferences: str | None = Form(None),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

    image_bytes = await file.read()
    job_id = str(uuid.uuid4())[:8]
    job_store.create(job_id)

    background_tasks.add_task(_run_pipeline, job_id, image_bytes, style, room_type, budget, preferences)
    return JobStarted(job_id=job_id)


# ── Poll status ───────────────────────────────────────────────────────────────

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    label: str
    result: Any = None
    error: str | None = None


@router.get("/redesign/job/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(job_id=job_id, **{k: job[k] for k in ("status","progress","label","result","error")})


# ── Pipeline (runs in background) ────────────────────────────────────────────

async def _run_pipeline(
    job_id: str,
    image_bytes: bytes,
    style: str,
    room_type: str,
    budget: int | None,
    preferences: str | None,
):
    try:
        job_store.update(job_id, "analyzing")
        analysis = await asyncio.get_event_loop().run_in_executor(
            None, analyze_room, image_bytes, room_type
        )

        job_store.update(job_id, "briefing")
        brief = await asyncio.get_event_loop().run_in_executor(
            None, build_brief, analysis, style, budget, preferences
        )

        job_store.update(job_id, "generating")
        result = await image_gen.generate_redesign(
            image_bytes, style, room_type, sd_prompt=brief.sd_prompt
        )

        products = _brief_to_products(brief, style, budget)
        plan = _brief_to_plan(brief, result.design_id)

        full = RedesignFullResponse(
            design_id=result.design_id,
            original_url=result.original_url,
            result_url=result.result_url,
            style=result.style,
            is_mock=result.is_mock,
            analysis=analysis,
            products=products,
            plan=plan,
        )
        job_store.finish(job_id, full.model_dump())

    except Exception as e:
        job_store.fail(job_id, str(e))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _brief_to_products(brief, style: str, budget: int | None) -> ProductsResponse:
    from app.models.schemas import Product
    import uuid as _uuid

    products = [
        Product(
            id=str(_uuid.uuid4())[:8],
            name=item.item, category=item.category,
            price=item.est_price, currency="KGS",
            image="", shop="—", url="#", priority=item.priority,
        )
        for item in brief.shopping_list
    ]
    return ProductsResponse(
        style=style, budget=budget,
        products=products, total_price=sum(p.price for p in products),
        currency="KGS", is_mock=brief.is_stub,
    )


def _brief_to_plan(brief, design_id: str) -> PlanResponse:
    weeks = [
        PlanWeek(
            week=p.phase, title=p.title, tasks=p.tasks,
            estimated_cost=p.est_cost, currency="KGS", icon="",
        )
        for p in brief.reno_plan
    ]
    return PlanResponse(
        design_id=design_id, total_weeks=len(weeks),
        total_cost=sum(w.estimated_cost for w in weeks),
        currency="KGS", weeks=weeks, is_mock=brief.is_stub,
    )

import json
from pathlib import Path
from fastapi import APIRouter
from app.models.schemas import PlanResponse, PlanWeek

router = APIRouter()

DATA_FILE = Path(__file__).parent.parent / "data" / "plans_mock.json"


@router.get("/plan/{design_id}", response_model=PlanResponse)
def get_plan(design_id: str):
    with open(DATA_FILE, encoding="utf-8") as f:
        data = json.load(f)

    plan = data["default"]
    weeks = [PlanWeek(**w) for w in plan["weeks"]]
    total_cost = sum(w.estimated_cost for w in weeks)

    return PlanResponse(
        design_id=design_id,
        total_weeks=plan["total_weeks"],
        total_cost=total_cost,
        currency="KGS",
        weeks=weeks,
        is_mock=True,
    )

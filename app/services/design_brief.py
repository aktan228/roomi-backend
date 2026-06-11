"""Design brief: the reasoning layer between perception and generation.

Takes the structured RoomAnalysis + the user's style/budget/preferences and
produces ONE brief that drives everything downstream — the SD prompt for the
image, the shopping list, and the phased reno plan. Grounding all three in the
same analysis keeps them consistent (the plan paints exactly the m² we measured).

Runs on Groq (Llama). Falls back to a heuristic brief synthesized from the
analysis when Groq is unavailable, so the pipeline never hard-fails.
"""
from __future__ import annotations

import json

from app.config import get_settings
from app.models.schemas import (
    RoomAnalysis, DesignBrief, ShoppingItem, PlanPhase,
)
from app.services.image_gen import STYLE_PROMPTS

_SYSTEM = (
    "You are roomi.ai's lead interior designer for the Central Asian market "
    "(Kyrgyzstan, Kazakhstan, Uzbekistan) — small Soviet-era apartments, tight "
    "budgets, locally available furniture and materials. "
    "You receive a JSON room analysis (detected objects, measured surfaces in m², "
    "palette) plus the user's chosen style, budget (KGS) and free-text wishes. "
    "Return ONLY a JSON object with these exact keys:\n"
    "  sd_prompt (string): a vivid Stable Diffusion prompt for the redesigned room\n"
    "  negative_prompt (string)\n"
    "  inpaint_targets (string[]): subset of [\"wall\",\"floor\",\"ceiling\"] to regenerate\n"
    "  shopping_list (array of {item, category, why, est_price (int KGS), priority (1-3)})\n"
    "  reno_plan (array of {phase (int), title, tasks (string[]), est_cost (int KGS)})\n"
    "Respect the budget. Prefer replacing objects the analysis marked 'replace'. "
    "Size furniture to the measured area. Keep prices realistic for the region."
)


def build_brief(
    analysis: RoomAnalysis,
    style: str,
    budget: int | None,
    preferences: str | None,
) -> DesignBrief:
    settings = get_settings()
    if not settings.use_groq:
        return _stub(analysis, style, budget)

    try:
        from groq import Groq
        client = Groq(api_key=settings.groq_api_key)

        user_payload = {
            "analysis": analysis.model_dump(),
            "style": style,
            "budget_kgs": budget,
            "preferences": preferences or "",
        }
        resp = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=0.6,
            max_tokens=1500,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        return DesignBrief(
            sd_prompt=data.get("sd_prompt") or _fallback_prompt(analysis, style),
            negative_prompt=data.get("negative_prompt", _NEG),
            inpaint_targets=data.get("inpaint_targets", []),
            shopping_list=[ShoppingItem(**s) for s in data.get("shopping_list", [])],
            reno_plan=[PlanPhase(**p) for p in data.get("reno_plan", [])],
            is_stub=False,
        )
    except Exception:
        return _stub(analysis, style, budget)


# ── Heuristic fallback (coherent with the analysis, not random) ──

_NEG = ("lowres, watermark, text, deformed, blurry, out of focus, "
        "cluttered, distorted perspective, ugly")


def _fallback_prompt(analysis: RoomAnalysis, style: str) -> str:
    return (
        f"{analysis.room_type.replace('_', ' ')}, "
        f"{STYLE_PROMPTS.get(style, STYLE_PROMPTS['minimal'])}, "
        f"{analysis.lighting} natural lighting, photorealistic, 8k, interior photography"
    )


def _stub(analysis: RoomAnalysis, style: str, budget: int | None) -> DesignBrief:
    cap = budget or 120_000
    wall = analysis.surfaces.wall_m2

    shopping: list[ShoppingItem] = [
        ShoppingItem(item=f"Краска для стен (~{wall:.0f} м²)", category="materials",
                     why="Освежить стены — максимальный эффект за минимум денег",
                     est_price=int(wall * 250), priority=1),
        ShoppingItem(item="Подвесное освещение", category="lighting",
                     why="Слоёный свет преображает любое пространство",
                     est_price=6000, priority=2),
        ShoppingItem(item="Шторы и текстиль", category="textile",
                     why="Тёплые тона и мягкость по запросу пользователя",
                     est_price=7000, priority=2),
    ]
    # Suggest a replacement for each big item the analysis flagged.
    for obj in analysis.objects:
        if obj.suggestion == "replace":
            shopping.append(ShoppingItem(
                item=f"Новый {obj.label} в стиле {style}", category="furniture",
                why=f"Заменить устаревший {obj.label}, найденный на фото",
                est_price=25000, priority=1,
            ))
    shopping = [s for s in shopping if s.priority == 1 or s.est_price <= cap * 0.3]

    plan = [
        PlanPhase(phase=1, title="Подготовка и стены",
                  tasks=["Освободить комнату", "Зашпаклевать неровности",
                         f"Покрасить ~{wall:.0f} м² стен", "Покрасить потолок"],
                  est_cost=int(wall * 250) + 3000),
        PlanPhase(phase=2, title="Мебель и освещение",
                  tasks=["Собрать и расставить мебель", "Установить освещение",
                         "Повесить полки и зеркала"],
                  est_cost=min(cap // 2, 60000)),
        PlanPhase(phase=3, title="Декор и финал",
                  tasks=["Повесить шторы", "Расстелить ковёр",
                         "Расставить декор и растения", "Финальная уборка"],
                  est_cost=12000),
    ]

    return DesignBrief(
        sd_prompt=_fallback_prompt(analysis, style),
        negative_prompt=_NEG,
        inpaint_targets=["wall", "floor"],
        shopping_list=shopping,
        reno_plan=plan,
        is_stub=True,
    )

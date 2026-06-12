"""In-memory job store for async design generation.

Each job tracks: status, progress (0-100), human-readable label, result/error.
Jobs are cleaned up after 2 hours to avoid memory leaks.
"""
from __future__ import annotations
import time
from typing import Any

# job_id → state dict
_jobs: dict[str, dict] = {}

STAGES = {
    "queued":     (0,   "В очереди..."),
    "analyzing":  (15,  "Анализируем комнату..."),
    "briefing":   (40,  "Составляем дизайн..."),
    "generating": (60,  "Генерируем изображение..."),
    "done":       (100, "Готово!"),
    "error":      (0,   "Ошибка"),
}


def create(job_id: str) -> None:
    _jobs[job_id] = {
        "status": "queued", "progress": 0,
        "label": STAGES["queued"][1],
        "result": None, "error": None,
        "created_at": time.time(),
    }


def update(job_id: str, stage: str) -> None:
    if job_id not in _jobs:
        return
    progress, label = STAGES.get(stage, (0, stage))
    _jobs[job_id].update({"status": stage, "progress": progress, "label": label})


def finish(job_id: str, result: Any) -> None:
    if job_id not in _jobs:
        return
    progress, label = STAGES["done"]
    _jobs[job_id].update({
        "status": "done", "progress": progress,
        "label": label, "result": result,
    })


def fail(job_id: str, error: str) -> None:
    if job_id not in _jobs:
        return
    _jobs[job_id].update({"status": "error", "error": error})


def get(job_id: str) -> dict | None:
    _cleanup()
    return _jobs.get(job_id)


def _cleanup() -> None:
    cutoff = time.time() - 7200  # 2 hours
    stale = [k for k, v in _jobs.items() if v["created_at"] < cutoff]
    for k in stale:
        del _jobs[k]

"""Room perception: turn a single photo into a structured RoomAnalysis.

Hybrid tier: light CV models run locally (YOLOv8 detection + MiDaS depth +
OpenCV palette). Each stage is wrapped — if a model/weight is missing the
stage degrades to a heuristic and the whole call still returns a valid
RoomAnalysis (is_stub=True), so the pipeline never hard-fails in a demo.

Note: FastSAM segmentation is intentionally NOT here yet — its masks are only
consumed by the SDXL inpainting step, so it lands in image_gen alongside that.
"""
from __future__ import annotations

import io

from app.config import get_settings
from app.models.schemas import RoomAnalysis, RoomObject, RoomSurfaces

# Typical living-space sizes (m²) for Central Asian / Soviet apartments —
# used as the metric prior, then nudged by measured depth spread.
_AREA_PRIOR: dict[str, float] = {
    "bedroom": 12, "living_room": 18, "kitchen": 8, "bathroom": 4,
    "office": 9, "dining_room": 12, "nursery": 10, "basement": 14,
    "balcony": 5, "gaming_room": 12,
}

# COCO labels (YOLOv8 default) → room vocabulary.
_LABEL_MAP: dict[str, str] = {
    "couch": "sofa", "bed": "bed", "chair": "chair", "dining table": "table",
    "tv": "tv", "potted plant": "plant", "refrigerator": "fridge",
    "oven": "oven", "sink": "sink", "toilet": "toilet", "microwave": "microwave",
    "vase": "decor", "book": "decor", "clock": "decor", "laptop": "electronics",
}
# Big furniture we'd suggest swapping in a redesign; everything else we keep.
_REPLACE = {"sofa", "bed", "chair", "table", "tv"}

# Lazy module-level singletons (loaded once, reused).
_yolo = None
_midas = None
_midas_tf = None


def analyze_room(image_bytes: bytes, room_type: str) -> RoomAnalysis:
    settings = get_settings()

    try:
        import numpy as np
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)
    except Exception:
        return _stub(room_type, settings.ceiling_height_m)

    if not settings.enable_perception:
        return _stub(room_type, settings.ceiling_height_m)

    objects = _detect(arr)
    depth_factor = _depth_factor(arr)          # ~0.8..1.4, 1.0 if depth unavailable
    palette = _palette(arr)
    lighting = _lighting(arr)

    area = round(_AREA_PRIOR.get(room_type, 12) * depth_factor, 1)
    surfaces = _surfaces(area, settings.ceiling_height_m)

    # If detection produced nothing AND depth was flat, we're effectively guessing.
    is_stub = not objects and depth_factor == 1.0

    return RoomAnalysis(
        room_type=room_type,
        est_area_m2=area,
        ceiling_height_m=settings.ceiling_height_m,
        objects=objects,
        surfaces=surfaces,
        palette=palette,
        lighting=lighting,
        is_stub=is_stub,
    )


# ── Stages ────────────────────────────────────────

def _detect(arr) -> list[RoomObject]:
    global _yolo
    try:
        if _yolo is None:
            from ultralytics import YOLO
            _yolo = YOLO(get_settings().yolo_model)

        h, w = arr.shape[:2]
        img_area = float(h * w)
        result = _yolo(arr, verbose=False)[0]

        objects: list[RoomObject] = []
        for box in result.boxes:
            name = result.names[int(box.cls)]
            label = _LABEL_MAP.get(name)
            if not label:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            area_ratio = round(((x2 - x1) * (y2 - y1)) / img_area, 3)
            objects.append(RoomObject(
                label=label,
                confidence=round(float(box.conf), 2),
                bbox=[x1, y1, x2, y2],
                area_ratio=area_ratio,
                suggestion="replace" if label in _REPLACE else "keep",
            ))
        return objects
    except Exception:
        return []


def _depth_factor(arr) -> float:
    """Run MiDaS, return a room-size multiplier from depth spread.

    Monocular depth is relative, not metric — so we don't read absolute size
    off it. Instead the spread (how much near/far variation the scene has)
    scales the room-type prior: a deep, open room reads larger than a shallow
    one shot from the doorway.
    """
    global _midas, _midas_tf
    try:
        import torch
        import numpy as np

        if _midas is None:
            model_name = get_settings().midas_hub_model
            _midas = torch.hub.load("intel-isl/MiDaS", model_name)
            _midas.eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            _midas_tf = (transforms.small_transform
                         if "small" in model_name.lower()
                         else transforms.dpt_transform)

        batch = _midas_tf(arr)
        with torch.no_grad():
            pred = _midas(batch)
        depth = pred.squeeze().cpu().numpy()

        d = depth.astype("float32")
        d = (d - d.min()) / (d.max() - d.min() + 1e-6)
        spread = float(np.percentile(d, 90) - np.percentile(d, 10))  # 0..1
        return round(min(1.4, max(0.8, 0.8 + spread * 0.6)), 3)
    except Exception:
        return 1.0


def _surfaces(area_m2: float, ceiling_h: float) -> RoomSurfaces:
    import math
    # Assume a roughly square footprint; drop ~15% of wall for openings.
    perimeter = 4 * math.sqrt(area_m2)
    wall_gross = perimeter * ceiling_h
    return RoomSurfaces(
        wall_m2=round(wall_gross * 0.85, 1),
        floor_m2=round(area_m2, 1),
        ceiling_m2=round(area_m2, 1),
    )


def _palette(arr, k: int = 4) -> list[str]:
    try:
        import cv2
        import numpy as np
        small = cv2.resize(arr, (64, 64))
        z = small.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(z, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        order = np.argsort(-np.bincount(labels.flatten(), minlength=k))
        return ["#%02x%02x%02x" % (int(c[0]), int(c[1]), int(c[2]))
                for c in centers[order]]
    except Exception:
        return []


def _lighting(arr) -> str:
    try:
        return "bright" if float(arr.mean()) > 110 else "dim"
    except Exception:
        return "bright"


def _stub(room_type: str, ceiling_h: float) -> RoomAnalysis:
    area = _AREA_PRIOR.get(room_type, 12)
    return RoomAnalysis(
        room_type=room_type,
        est_area_m2=area,
        ceiling_height_m=ceiling_h,
        objects=[],
        surfaces=_surfaces(area, ceiling_h),
        palette=[],
        lighting="bright",
        is_stub=True,
    )

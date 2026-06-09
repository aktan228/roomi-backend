from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.models.schemas import RedesignResponse
from app.services import image_gen

router = APIRouter()


@router.post("/redesign", response_model=RedesignResponse)
async def redesign_room(
    file: UploadFile = File(...),
    style: str = Form("minimal"),
    room_type: str = Form("bedroom"),
    budget: int | None = Form(None),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

    image_bytes = await file.read()
    result = await image_gen.generate_redesign(image_bytes, style, room_type)
    return result

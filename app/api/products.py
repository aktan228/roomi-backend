from fastapi import APIRouter, Query
from app.models.schemas import ProductsResponse
from app.services import products as products_service

router = APIRouter()


@router.get("/products", response_model=ProductsResponse)
def get_products(
    style: str = Query("minimal"),
    budget: int | None = Query(None, description="Budget in KGS"),
):
    return products_service.get_products(style, budget)

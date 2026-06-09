import json
from pathlib import Path
from app.models.schemas import Product, ProductsResponse

DATA_FILE = Path(__file__).parent.parent / "data" / "products_mock.json"

BUDGET_LIMITS = {
    "economy": 50_000,
    "medium":  120_000,
    "premium": 999_999,
}


def get_products(style: str, budget: int | None) -> ProductsResponse:
    with open(DATA_FILE, encoding="utf-8") as f:
        all_products: dict = json.load(f)

    # Fall back to minimal if style not found
    items = all_products.get(style) or all_products.get("minimal", [])

    products = [Product(**p) for p in items]

    # Filter by budget — keep priority-1 items always, drop luxury if tight budget
    if budget:
        products = [
            p for p in products
            if p.priority == 1 or p.price <= budget * 0.25
        ]

    # Sort by priority
    products.sort(key=lambda p: p.priority)

    total = sum(p.price for p in products)

    return ProductsResponse(
        style=style,
        budget=budget,
        products=products,
        total_price=total,
        currency="KGS",
        is_mock=True,
    )

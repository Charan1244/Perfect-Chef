@@ -1,254 +0,0 @@
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
import sqlite3
import json
import os
import base64
from datetime import datetime

# ── App Init ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Cart to Cook API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBbcjPJuBxZCjuocJvd4-ZnQC-UlVTwT88")
genai.configure(api_key=GEMINI_API_KEY)

# ── Database Setup ─────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect("carttocook.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            dish_name   TEXT NOT NULL,
            ingredients TEXT NOT NULL,   -- JSON array
            total_items INTEGER NOT NULL,
            status      TEXT DEFAULT 'placed',
            created_at  TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recipe_cache (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            dish_name   TEXT UNIQUE NOT NULL,
            recipe_json TEXT NOT NULL,
            created_at  TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

init_db()

# ── Pydantic Models ────────────────────────────────────────────────────────────
class DishRequest(BaseModel):
    dish_name: str

class Ingredient(BaseModel):
    name: str
    quantity: str

class RecipeResponse(BaseModel):
    dish_name: str
    instructions: List[str]
    ingredients: List[Ingredient]
    prep_time: str
    serves: int

class OrderRequest(BaseModel):
    dish_name: str
    ingredients: List[Ingredient]

class OrderResponse(BaseModel):
    order_id: int
    dish_name: str
    status: str
    total_items: int
    created_at: str

# ── Helper: DB Connection ──────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect("carttocook.db")
    conn.row_factory = sqlite3.Row
    return conn

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Cart to Cook API"}


@app.post("/api/recipe", response_model=RecipeResponse)
def get_recipe(req: DishRequest):
    """
    Fetch AI-generated recipe for a dish using Gemini.
    Caches results in SQLite to avoid redundant API calls.
    """
    dish = req.dish_name.strip().lower()

    # Check cache first
    conn = get_db()
    cached = conn.execute(
        "SELECT recipe_json FROM recipe_cache WHERE dish_name = ?", (dish,)
    ).fetchone()

    if cached:
        conn.close()
        return json.loads(cached["recipe_json"])

    # Call Gemini
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
        Generate a detailed recipe for "{dish}". 
        Return ONLY a valid JSON object with these exact keys:
        {{
          "dish_name": "{dish}",
          "instructions": ["step 1", "step 2", ...],
          "ingredients": [{{"name": "ingredient", "quantity": "amount"}}],
          "prep_time": "X minutes",
          "serves": number
        }}
        """
        response = model.generate_content(prompt)
        text = response.text.strip().strip("```json").strip("```").strip()
        recipe_data = json.loads(text)

        # Cache it
        conn.execute(
            "INSERT INTO recipe_cache (dish_name, recipe_json, created_at) VALUES (?, ?, ?)",
            (dish, json.dumps(recipe_data), datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()

        return recipe_data

    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")


@app.post("/api/generate-image")
def generate_dish_image(req: DishRequest):
    """
    Generate a dish image using Gemini's image generation model.
    Returns base64-encoded image.
    """
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = model.generate_content(
            f"Generate a realistic, appetizing food photo of {req.dish_name}, "
            f"professional food photography, bright lighting, top-down view.",
        )

        # Extract image from response
        for part in response.parts:
            if hasattr(part, "inline_data"):
                image_data = base64.b64encode(part.inline_data.data).decode("utf-8")
                return {
                    "image_base64": image_data,
                    "mime_type": part.inline_data.mime_type
                }

        raise HTTPException(status_code=500, detail="No image generated")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/orders", response_model=OrderResponse)
def place_order(req: OrderRequest):
    """
    Place a new order and persist it to SQLite.
    """
    conn = get_db()
    try:
        cursor = conn.execute(
            """INSERT INTO orders (dish_name, ingredients, total_items, status, created_at)
               VALUES (?, ?, ?, 'placed', ?)""",
            (
                req.dish_name,
                json.dumps([i.dict() for i in req.ingredients]),
                len(req.ingredients),
                datetime.utcnow().isoformat()
            )
        )
        conn.commit()
        order_id = cursor.lastrowid

        order = conn.execute(
            "SELECT * FROM orders WHERE id = ?", (order_id,)
        ).fetchone()

        return OrderResponse(
            order_id=order["id"],
            dish_name=order["dish_name"],
            status=order["status"],
            total_items=order["total_items"],
            created_at=order["created_at"]
        )
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@app.get("/api/orders", response_model=List[OrderResponse])
def get_all_orders():
    """
    Retrieve all placed orders from the database.
    """
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM orders ORDER BY created_at DESC"
    ).fetchall()
    conn.close()

    return [
        OrderResponse(
            order_id=r["id"],
            dish_name=r["dish_name"],
            status=r["status"],
            total_items=r["total_items"],
            created_at=r["created_at"]
        )
        for r in rows
    ]


@app.get("/api/orders/{order_id}", response_model=OrderResponse)
def get_order(order_id: int):
    """
    Retrieve a specific order by ID.
    """
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM orders WHERE id = ?", (order_id,)
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Order not found")

    return OrderResponse(
        order_id=row["id"],
        dish_name=row["dish_name"],
        status=row["status"],
        total_items=row["total_items"],
        created_at=row["created_at"]
    )

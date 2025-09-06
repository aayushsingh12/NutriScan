from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import requests
import json
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NutriScanner API")


@app.get("/")
def root():
    return {"message": "NutriScanner API running"}


@app.get("/openfoodfacts")
def get_open_food_facts(brand: str = Query(..., description="Brand name to search for")):
    """
    Query Open Food Facts API exactly like:
    https://world.openfoodfacts.net/api/v2/search?brands_tags=sting

    After fetching, save top product tags (ingredients + nova_group) and then POST
    that JSON to the RAG service (/analyze_ingredients). The response will only
    include the simplified top_product JSON and the rag_analysis.
    """
    url = "https://world.openfoodfacts.net/api/v2/search"
    params = {"brands_tags": brand}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # --- Extract and save top product ---
        top_product_json = None
        if "products" in data and len(data["products"]) > 0:
            top_product = data["products"][0]
            ingredients_text_en = top_product.get("ingredients_text_en", "")
            nova_group = top_product.get("nova_group", None)

            ingredients_list = [item.strip() for item in ingredients_text_en.split(",") if item.strip()]

            top_product_json = {
                "ingredients": ingredients_list,
                "nova_group": nova_group
            }

            with open("top_products.json", "w", encoding="utf-8") as f:
                json.dump(top_product_json, f, ensure_ascii=False, indent=4)

        # --- POST to RAG ---
        rag_output = {"error": "No payload prepared"}
        if top_product_json:
            rag_endpoints = [
                "http://localhost:9000/analyze_ingredients",
                "http://localhost:8000/analyze_ingredients"
            ]
            for rag_url in rag_endpoints:
                try:
                    r = requests.post(rag_url, json=top_product_json, timeout=30)
                    r.raise_for_status()
                    rag_output = r.json()
                    break
                except requests.exceptions.RequestException as e:
                    rag_output = {"error": f"Failed to reach RAG at {rag_url}: {e}"}

        # --- Final combined response (only simplified product + rag_analysis) ---
        combined = {
            "top_product": top_product_json,
            "rag_analysis": rag_output
        }

        with open("combined_response.json", "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)

        return JSONResponse(content=combined)

    except requests.exceptions.RequestException as e:
        return JSONResponse(content={"error": str(e)}, status_code=502)

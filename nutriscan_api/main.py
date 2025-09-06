from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import requests

app = FastAPI(title="NutriScanner API")


@app.get("/")
def root():
    return {"message": "NutriScanner API running"}

@app.get("/openfoodfacts")
def get_open_food_facts(brand: str = Query(..., description="Brand name to search for")):
    """
    Query Open Food Facts API exactly like:
    https://world.openfoodfacts.net/api/v2/search?brands_tags=sting
    """
    url = "https://world.openfoodfacts.net/api/v2/search"
    params = {
        "brands_tags": brand
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # --- NEW: store ingredients_text_en and nova_group ---
        if "products" in data and len(data["products"]) > 0:
            top_product = data["products"][0]
            ingredients_text_en = top_product.get("ingredients_text_en", None)
            nova_group = top_product.get("nova_group", None)

            # Save to a JSON file locally
            save_data = {
                "ingredients_text_en": ingredients_text_en,
                "nova_group": nova_group
            }
            with open("top_product_tags.json", "w", encoding="utf-8") as f:
                import json
                json.dump(save_data, f, ensure_ascii=False, indent=4)

        # Return the full API response exactly
        return JSONResponse(content=data)

    except requests.exceptions.RequestException as e:
        return JSONResponse(content={"error": str(e)})

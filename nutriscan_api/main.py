from fastapi import FastAPI, Query, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
import os
from pathlib import Path
import logging
import shutil
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gem import extract_ingredients_gemini

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NutriScanner API")

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "NutriScanner API running"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Create a temporary file to save the uploaded image
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract ingredients using the Gemini model
        ingredients_result = extract_ingredients_gemini(temp_path)
        
        # Clean up the temporary file
        os.remove(temp_path)
        
        # If there are ingredients, send them to the RAG system
        if ingredients_result and "ingredients" in ingredients_result:
            try:
                # Prepare payload for RAG
                rag_payload = {
                    "ingredients": ingredients_result["ingredients"]
                }
                
                # Try different RAG endpoints
                rag_endpoints = [
                    "http://localhost:9000/analyze_ingredients",
                    "http://localhost:8000/analyze_ingredients"
                ]
                
                rag_response = None
                for rag_url in rag_endpoints:
                    try:
                        r = requests.post(rag_url, json=rag_payload, timeout=30)
                        r.raise_for_status()
                        rag_response = r.json()
                        break
                    except requests.exceptions.RequestException:
                        continue
                
                # Combine results
                result = {
                    "extracted_ingredients": ingredients_result,
                    "rag_analysis": rag_response if rag_response else {"error": "RAG analysis failed"}
                }
                
                return JSONResponse(content=result)
            
            except Exception as e:
                return JSONResponse(
                    content={"error": f"Error during RAG analysis: {str(e)}"},
                    status_code=500
                )
        
        return JSONResponse(content=ingredients_result)
    
    except Exception as e:
        return JSONResponse(
            content={"error": f"Error processing image: {str(e)}"},
            status_code=500
        )


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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import re
import asyncio
from contextlib import asynccontextmanager

# Import the RAG system components
import subprocess
import time
import requests
import os
import psutil
from pathlib import Path
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store RAG system
rag_chain = None
retriever = None

# Import statements for RAG system (will be imported after dependency installation)
def import_rag_dependencies():
    """Import RAG dependencies after installation"""
    global PyPDFLoader, RecursiveCharacterTextSplitter, FAISS, Ollama, OllamaEmbeddings
    global create_stuff_documents_chain, create_retrieval_chain
    global ChatPromptTemplate, Document
    
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.llms import Ollama
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.documents import Document

# Pydantic models for request/response
class IngredientListRequest(BaseModel):
    ingredients: List[str]

class Warning(BaseModel):
    type: str
    message: str
    item: str

class IngredientAnalysisResponse(BaseModel):
    product_name: str
    ingredients: List[str]
    warnings: List[Warning]
    nova_group: int
    nova_description: str
    health_notes: str

# Initialize RAG system on startup
async def initialize_rag_system():
    """Initialize the RAG system asynchronously"""
    global rag_chain, retriever
    
    logger.info("Starting RAG system initialization...")
    
    try:
        # Install dependencies
        logger.info("Installing dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q",
            "langchain", "langchain_community", "langchain-huggingface", "faiss-cpu",
            "sentence-transformers", "pypdf", "requests", "psutil",
            "beautifulsoup4", "lxml", "transformers", "torch", "uvicorn[standard]", "fastapi"
        ], check=True)
        
        # Import dependencies
        import_rag_dependencies()
        
        # Create enhanced knowledge base with structured data
        knowledge_base = [
            Document(
                page_content="""NOVA Group 1: Unprocessed or Minimally Processed Foods
Definition: Natural foods altered only by removal of inedible parts, drying, crushing, grinding, fractioning, filtering, roasting, boiling, non-alcoholic fermentation, pasteurization, chilling, freezing, placing in containers and vacuum-packaging.
Examples: Fresh, dried, ground, chilled, frozen, pasteurized fruits and vegetables; grains like brown rice, corn kernels, wheat berries; legumes like beans, lentils, chickpeas; nuts and seeds; meat, poultry, fish and seafood; eggs; milk.
Health Impact: Generally healthiest option, rich in nutrients, fiber, and beneficial compounds.""",
                metadata={"category": "nova_classification", "group": "1"}
            ),
            Document(
                page_content="""NOVA Group 2: Processed Culinary Ingredients
Definition: Substances derived from Group 1 foods or from nature by processes that include pressing, refining, grinding, milling, and drying.
Examples: Oils, butter, sugar, salt, and other substances derived from Group 1 foods and used in kitchens to prepare, season and cook Group 1 foods.
Health Impact: Should be used in small amounts to season and prepare foods from Group 1.""",
                metadata={"category": "nova_classification", "group": "2"}
            ),
            Document(
                page_content="""NOVA Group 3: Processed Foods
Definition: Products made by adding salt, oil, sugar or other substances from Group 2 to Group 1 foods.
Examples: Bottled vegetables, canned fish, fruits in syrup, cheeses and freshly made breads.
Health Impact: Most processed foods have two or three ingredients, and can be consumed in moderation as part of a balanced diet.""",
                metadata={"category": "nova_classification", "group": "3"}
            ),
            Document(
                page_content="""NOVA Group 4: Ultra-Processed Foods
Definition: Industrial formulations made entirely or mostly from substances extracted from foods (oils, fats, sugar, starch, and proteins), derived from food constituents (hydrogenated fats and modified starch), or synthesized in laboratories from food substrates or other organic sources (flavor enhancers, colors, and emulsifiers).
Examples: Carbonated soft drinks, sweet or savory packaged snacks, ice-cream, chocolate, candies, mass-produced packaged breads and buns, margarines, breakfast cereals, cereal and energy bars, instant soups, many ready-to-heat products.
Indicators: Contains 5+ ingredients, includes additives like preservatives, colors, flavors, emulsifiers, artificial sweeteners.
Health Concerns: Linked to obesity, type 2 diabetes, cardiovascular disease, and some cancers. High in calories, sugar, unhealthy fats, and sodium while being low in protein, fiber, and micronutrients.""",
                metadata={"category": "nova_classification", "group": "4"}
            ),
            Document(
                page_content="""Hidden Sugars and Alternative Names: Complete List
Common hidden sugars include: Sugar, Maltodextrin, Corn Syrup, High Fructose Corn Syrup, Dextrose, Sucrose, Fructose, Glucose Syrup, Cane Sugar, Brown Sugar, Coconut Sugar, Agave Nectar, Honey, Maple Syrup, Molasses, Date Sugar, Rice Syrup, Barley Malt, Fruit Juice Concentrate.
Health Impact: All contribute to daily sugar intake and can lead to blood sugar spikes, weight gain, and increased risk of diabetes and heart disease.""",
                metadata={"category": "hidden_sugars"}
            ),
            Document(
                page_content="""Major Food Allergens (Top 14 List)
1. Cereals containing gluten: wheat, rye, barley, oats, spelt, kamut, triticale
2. Crustaceans: prawns, crabs, lobster, crayfish, shrimp
3. Eggs: all forms including powdered, liquid
4. Fish: all fish species and fish-derived products
5. Peanuts: groundnuts and all peanut products
6. Soybeans: soy, soya, and all soy derivatives
7. Milk: all dairy products, lactose, casein, whey
8. Tree nuts: almonds, hazelnuts, walnuts, cashews, pecans, Brazil nuts, pistachios, macadamia nuts
9. Celery: including celeriac and celery seed
10. Mustard: seeds, leaves, and mustard preparations
11. Sesame seeds: tahini, sesame oil, halva
12. Sulphur dioxide and sulphites: when >10mg/kg or 10mg/L
13. Lupin: legume used in some flours and pastries
14. Molluscs: mussels, oysters, snails, squid, octopus""",
                metadata={"category": "allergens"}
            ),
            Document(
                page_content="""E-Number Food Additives Classification
E100-199: Colors (natural and artificial dyes) - Examples: E102 Tartrazine, E110 Sunset Yellow, E129 Allura Red, E160c Paprika Extract
E200-299: Preservatives - Examples: E200 Sorbic Acid, E211 Sodium Benzoate, E250 Sodium Nitrite
E300-399: Antioxidants and Acidity Regulators - Examples: E300 Ascorbic Acid, E330 Citric Acid, E296 Malic Acid, E334 Tartaric Acid
E400-499: Thickeners, Stabilizers, Emulsifiers - Examples: E407 Carrageenan, E415 Xanthan Gum, E471 Mono- and Diglycerides
E500-599: pH Regulators and Anti-caking Agents - Examples: E500 Sodium Carbonate, E551 Silicon Dioxide
E600-699: Flavor Enhancers - Examples: E621 MSG (Monosodium Glutamate)
E900-999: Miscellaneous - Examples: E950 Acesulfame K, E951 Aspartame""",
                metadata={"category": "e_numbers"}
            ),
            Document(
                page_content="""Common Food Additive Numbers and Their Functions
330: Citric Acid - Acidity regulator, preservative, antioxidant. Generally safe.
296: Malic Acid - Acidity regulator, flavor enhancer. Generally safe.
334: Tartaric Acid - Acidity regulator, antioxidant. Generally safe.
160c: Paprika Extract - Natural color, orange-red pigment. Generally safe.
621: Monosodium Glutamate (MSG) - Flavor enhancer. Some people may be sensitive.
200: Sorbic Acid - Preservative. Generally safe.
211: Sodium Benzoate - Preservative. May cause issues in sensitive individuals.""",
                metadata={"category": "specific_additives"}
            ),
            Document(
                page_content="""Red Dye No. 3 (Erythrosine, E127) - FDA Regulatory Update
FDA Status: The U.S. FDA has initiated proceedings to revoke the authorization for use of Red Dye No. 3 (Erythrosine) in food and ingested drugs.
Chemical Name: Erythrosine, Disodium Salt of 9-(o-carboxyphenyl)-6-hydroxy-2,4,5,7-tetraiodo-3H-xanthen-3-one
E-Number: E127
Color: Cherry-red color
Common Uses: Previously used in candies, baked goods, dairy products, beverages, and various processed foods
Health Concerns: Studies have linked Red Dye No. 3 to thyroid tumors in animal studies. The FDA's action follows mounting scientific evidence and consumer advocacy for safer food coloring alternatives.
International Status: Already banned or restricted in several countries including parts of the EU.
Alternatives: Natural colorings like beetroot extract, paprika extract, annatto, and other approved synthetic dyes.
Source: FDA Consumer Update on food dye authorization revocation""",
                metadata={"category": "food_dyes", "status": "regulatory_action", "source": "FDA"}
            ),
            Document(
                page_content="""Tartrazine (Yellow Dye No. 5, E102) - Comprehensive Chemical Profile
Chemical Name: Tartrazine
IUPAC Name: Trisodium (4E)-5-oxo-1-(4-sulfonatophenyl)-4-[(4-sulfonatophenyl)hydrazono]pyrazole-3-carboxylate
E-Number: E102
FDA Color: FD&C Yellow No. 5
Molecular Formula: C16H9N4Na3O9S2
Molecular Weight: 534.36 g/mol
CAS Number: 1934-21-0
Appearance: Bright lemon-yellow synthetic azo dye
Physical Properties: Water-soluble, heat-stable, light-sensitive
Common Uses: Soft drinks, candies, cereals, snack foods, ice cream, cosmetics, medications, vitamins
Health Concerns: 
- May cause allergic reactions, especially in people with aspirin sensitivity (salicylate intolerance)
- Can trigger asthma attacks and urticaria (hives) in sensitive individuals
- May cause hyperactivity in some children (Southampton Six study results)
- Cross-reactivity with other azo dyes possible
Regulation: Approved for use in US, EU, and many countries but requires mandatory labeling
FDA Requirement: Must be specifically listed as "FD&C Yellow No. 5" or "Tartrazine" on food labels
Cross-Reactivity: People allergic to aspirin, benzoates, or other azo dyes may also react to tartrazine
Alternatives: Natural yellow colorings like turmeric (curcumin), annatto extract, beta-carotene, or riboflavin
Source: PubChem compound database and FDA food additive regulations""",
                metadata={"category": "food_dyes", "allergen_potential": "moderate", "source": "PubChem_FDA"}
            )
        ]
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""],
            length_function=len
        )
        
        split_docs = text_splitter.split_documents(knowledge_base)
        
        # Create embeddings (try new HuggingFace package first, fallback to community)
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("Using HuggingFace embeddings (new package)")
        except ImportError:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                logger.info("Using HuggingFace embeddings (community package)")
            except Exception as e:
                logger.error(f"Failed to create embeddings: {e}")
                raise
        
        # Create vector store
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}
        )
        
        logger.info("RAG system initialized successfully (retriever mode)")
        return True
        
    except Exception as e:
        logger.error(f"RAG system initialization failed: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up Food Safety RAG API...")
    success = await initialize_rag_system()
    if not success:
        logger.error("Failed to initialize RAG system")
    yield
    # Shutdown
    logger.info("Shutting down Food Safety RAG API...")

# Initialize FastAPI app
app = FastAPI(
    title="Food Safety RAG API",
    description="API for analyzing food ingredients using RAG (Retrieval Augmented Generation)",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_and_parse_ingredients(raw_ingredients: List[str]) -> tuple[str, List[str]]:
    """Parse and clean the raw ingredient list"""
    
    # Join all ingredients and clean
    full_text = " ".join(raw_ingredients)
    
    # Extract product name (usually the first meaningful item)
    product_name = raw_ingredients[0] if raw_ingredients else "Unknown Product"
    
    # Clean product name
    product_name = re.sub(r'\d+', '', product_name).strip().strip(',')
    
    # Common ingredient patterns and cleaning
    ingredients = []
    
    # Split by common delimiters and clean
    text_parts = re.split(r'[,\n]', full_text)
    
    current_ingredient = ""
    for part in text_parts:
        part = part.strip()
        if not part or part.isdigit() or len(part) < 2:
            continue
            
        # Skip product codes and meal numbers
        if re.match(r'^(NAMKEEN|Meal|Cereal Products)\s*\d+', part):
            continue
            
        # Handle parenthetical content
        if '(' in part and ')' in part:
            ingredients.append(part)
        elif '(' in part:
            current_ingredient = part
        elif ')' in part and current_ingredient:
            ingredients.append(current_ingredient + " " + part)
            current_ingredient = ""
        elif current_ingredient:
            current_ingredient += " " + part
        else:
            # Clean common patterns
            cleaned = re.sub(r'\s+\d+\s*,?\s*$', '', part)  # Remove trailing numbers
            cleaned = re.sub(r'^[\d\s,]+', '', cleaned)  # Remove leading numbers/spaces
            cleaned = cleaned.strip(',').strip()
            
            if len(cleaned) > 2 and not cleaned.isdigit():
                ingredients.append(cleaned)
    
    # Further cleaning and categorization
    final_ingredients = []
    for ing in ingredients:
        ing = ing.strip()
        if len(ing) > 2:
            # Handle specific patterns
            if "Oil" in ing and "," in ing:
                # Handle oil listings
                oils = [oil.strip() for oil in ing.split(",") if "oil" in oil.lower()]
                if oils:
                    final_ingredients.append(f"Edible Vegetable Oil ({', '.join(oils)})")
                else:
                    final_ingredients.append(ing)
            elif "Acidity Regulators" in ing:
                # Extract E-numbers
                numbers = re.findall(r'\d{3}', ing)
                if numbers:
                    final_ingredients.append(f"Acidity Regulators ({', '.join(numbers)})")
                else:
                    final_ingredients.append(ing)
            elif "Colour" in ing:
                # Handle color additives
                color_match = re.search(r'(\d+[a-z]?)', ing, re.IGNORECASE)
                if color_match:
                    final_ingredients.append(f"Colour ({color_match.group(1)})")
                else:
                    final_ingredients.append(ing)
            else:
                final_ingredients.append(ing)
    
    return product_name, final_ingredients

async def analyze_with_rag(ingredients: List[str]) -> Dict[str, Any]:
    """Analyze ingredients using RAG system"""
    global retriever
    
    if not retriever:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    # Create analysis query
    query = f"Analyze these food ingredients for health concerns, allergens, additives, and NOVA classification: {', '.join(ingredients)}"
    
    try:
        # Get relevant documents (use invoke method if available, fallback to deprecated method)
        try:
            docs = retriever.invoke(query)
        except AttributeError:
            # Fallback to deprecated method
            docs = retriever.get_relevant_documents(query)
        
        # Analyze ingredients
        warnings = []
        nova_group = 1
        additive_count = 0
        
        # Check each ingredient
        for ingredient in ingredients:
            ingredient_lower = ingredient.lower()
            
            # Check for hidden sugars
            hidden_sugars = ['sugar', 'maltodextrin', 'corn syrup', 'fructose', 'glucose', 'sucrose', 'dextrose']
            if any(sugar in ingredient_lower for sugar in hidden_sugars):
                sugar_items = [sugar for sugar in hidden_sugars if sugar in ingredient_lower]
                warnings.append({
                    "type": "hidden_sugar",
                    "message": f"Contains {', '.join(sugar_items)}, which may contribute to hidden sugar content",
                    "item": ingredient
                })
            
            # Check for additives (E-numbers or specific additives)
            if ('colour' in ingredient_lower or 'color' in ingredient_lower or 
                'acidity regulator' in ingredient_lower or 'preservative' in ingredient_lower or
                re.search(r'\d{3}', ingredient)):
                additive_count += 1
                warnings.append({
                    "type": "additive",
                    "message": f"Contains food additive: {ingredient}",
                    "item": ingredient
                })
            
            # Check for allergens
            allergens = ['sesame', 'soy', 'wheat', 'milk', 'egg', 'fish', 'peanut', 'tree nut']
            for allergen in allergens:
                if allergen in ingredient_lower:
                    warnings.append({
                        "type": "allergen",
                        "message": f"Contains potential allergen: {allergen}",
                        "item": ingredient
                    })
        
        # Determine NOVA group
        if additive_count >= 3 or len(ingredients) >= 8:
            nova_group = 4
            nova_description = "Ultra-processed food due to presence of multiple additives and processed ingredients"
        elif additive_count >= 1 or any('oil' in ing.lower() or 'salt' in ing.lower() for ing in ingredients):
            nova_group = 3
            nova_description = "Processed food with added ingredients"
        elif len(ingredients) <= 3:
            nova_group = 1
            nova_description = "Minimally processed food"
        else:
            nova_group = 2
            nova_description = "Processed culinary ingredients"
        
        # Generate health notes
        health_notes = []
        if nova_group == 4:
            health_notes.append("Contains ultra-processed elements; monitor intake due to potential health impacts")
        if any(w['type'] == 'hidden_sugar' for w in warnings):
            health_notes.append("Contains added sugars which may contribute to daily sugar intake")
        if any(w['type'] == 'additive' for w in warnings):
            health_notes.append("Contains food additives - check individual tolerance")
        
        health_notes_str = "; ".join(health_notes) if health_notes else "Generally safe when consumed as part of a balanced diet"
        
        return {
            "warnings": warnings,
            "nova_group": nova_group,
            "nova_description": nova_description,
            "health_notes": health_notes_str
        }
        
    except Exception as e:
        logger.error(f"RAG analysis error: {e}")
        # Fallback analysis
        return {
            "warnings": [{"type": "system", "message": "Limited analysis available", "item": "System"}],
            "nova_group": 3,
            "nova_description": "Analysis unavailable - classified as processed food",
            "health_notes": "Unable to perform full analysis - consume mindfully"
        }

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Food Safety RAG API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    global rag_chain, retriever
    
    return {
        "status": "healthy",
        "rag_system": "active" if retriever else "inactive",
        "endpoints": ["/analyze_ingredients", "/health", "/"]
    }

@app.post("/analyze_ingredients", response_model=IngredientAnalysisResponse)
async def analyze_ingredients(request: IngredientListRequest):
    """
    Analyze food ingredients and return structured safety information
    
    Args:
        request: IngredientListRequest containing list of ingredients
        
    Returns:
        IngredientAnalysisResponse with analysis results
    """
    try:
        if not request.ingredients:
            raise HTTPException(status_code=400, detail="No ingredients provided")
        
        # Parse and clean ingredients
        product_name, clean_ingredients = clean_and_parse_ingredients(request.ingredients)
        
        if not clean_ingredients:
            raise HTTPException(status_code=400, detail="No valid ingredients found")
        
        # Analyze with RAG system
        analysis_result = await analyze_with_rag(clean_ingredients)
        
        # Construct response
        response = IngredientAnalysisResponse(
            product_name=product_name,
            ingredients=clean_ingredients,
            warnings=[Warning(**warning) for warning in analysis_result["warnings"]],
            nova_group=analysis_result["nova_group"],
            nova_description=analysis_result["nova_description"],
            health_notes=analysis_result["health_notes"]
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/analyze_ingredients_simple")
async def analyze_ingredients_simple(request: IngredientListRequest):
    """
    Simplified analysis endpoint that returns raw JSON
    """
    try:
        if not request.ingredients:
            return {"error": "No ingredients provided"}
        
        # Parse and clean ingredients
        product_name, clean_ingredients = clean_and_parse_ingredients(request.ingredients)
        
        if not clean_ingredients:
            return {"error": "No valid ingredients found"}
        
        # Analyze with RAG system
        analysis_result = await analyze_with_rag(clean_ingredients)
        
        return {
            "product_name": product_name,
            "ingredients": clean_ingredients,
            "warnings": analysis_result["warnings"],
            "nova_group": analysis_result["nova_group"],
            "nova_description": analysis_result["nova_description"],
            "health_notes": analysis_result["health_notes"]
        }
        
    except Exception as e:
        logger.error(f"Simple analysis error: {e}")
        return {"error": f"Analysis failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
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
        
        # Load PDF documents from the workspace
        pdf_documents = []
        pdf_files = ["chapter_3.pdf", "appendix_a_b.pdf", "banned_food_additives.pdf"]
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(os.getcwd(), pdf_file)
            if os.path.exists(pdf_path):
                try:
                    logger.info(f"Loading PDF: {pdf_file}")
                    loader = PyPDFLoader(pdf_path)
                    pdf_docs = loader.load()
                    pdf_documents.extend(pdf_docs)
                    logger.info(f"Successfully loaded {len(pdf_docs)} pages from {pdf_file}")
                except Exception as e:
                    logger.warning(f"Failed to load {pdf_file}: {e}")
            else:
                logger.warning(f"PDF file not found: {pdf_path}")
        
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
        
        # Add PDF documents to the knowledge base
        if pdf_documents:
            logger.info(f"Adding {len(pdf_documents)} PDF pages to knowledge base")
            knowledge_base.extend(pdf_documents)
        else:
            logger.warning("No PDF documents were loaded - using built-in knowledge base only")
        
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

def clean_and_parse_ingredients(raw_ingredients: List[str]) -> Tuple[str, List[str]]:
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
    """Analyze ingredients using RAG system with detailed analysis"""
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
        
        # Analyze ingredients with enhanced detail
        warnings = []
        nova_group = 1
        additive_count = 0
        
        # Enhanced ingredient analysis with detailed messaging
        for ingredient in ingredients:
            ingredient_lower = ingredient.lower()
            
            # Enhanced sugar analysis with detailed health implications
            hidden_sugars = {
                'sugar': 'Added refined sugar increases caloric density and can cause rapid blood glucose spikes. Regular consumption is linked to increased risk of obesity, dental caries, and metabolic syndrome.',
                'maltodextrin': 'Maltodextrin is a highly processed starch derivative with a high glycemic index that can cause blood sugar spikes faster than table sugar. It may disrupt gut microbiome balance and contribute to insulin resistance.',
                'corn syrup': 'High fructose corn syrup is associated with increased risk of fatty liver disease, insulin resistance, and weight gain. It bypasses normal satiety signals and may contribute to overeating patterns.',
                'fructose': 'Concentrated fructose can overload liver metabolism, leading to increased fat synthesis and potential development of non-alcoholic fatty liver disease. It does not trigger the same satiety responses as glucose.',
                'glucose': 'While glucose is the body\'s primary energy source, added glucose in processed foods contributes to rapid blood sugar elevation and can lead to energy crashes and increased hunger cycles.',
                'sucrose': 'Table sugar (sucrose) provides empty calories without nutritional value and contributes to dental decay through bacterial fermentation. Regular intake is associated with increased inflammation markers.',
                'dextrose': 'Dextrose is rapidly absorbed glucose that can cause immediate blood sugar spikes, particularly problematic for individuals with diabetes or prediabetes. It offers no nutritional benefits beyond quick energy.'
            }
            
            for sugar_type, description in hidden_sugars.items():
                if sugar_type in ingredient_lower:
                    warnings.append({
                        "type": "hidden_sugar",
                        "message": f"Contains {sugar_type.title()}: {description}",
                        "item": ingredient
                    })
            
            # Enhanced additive analysis with health implications
            if 'colour' in ingredient_lower or 'color' in ingredient_lower:
                additive_count += 1
                color_match = re.search(r'(\d+[a-z]?)', ingredient, re.IGNORECASE)
                if color_match:
                    color_code = color_match.group(1)
                    if color_code.lower() in ['102', 'e102']:
                        warnings.append({
                            "type": "additive",
                            "message": f"Contains Tartrazine (E102/Yellow Dye #5): This synthetic azo dye may trigger hyperactivity in sensitive children and can cause allergic reactions including asthma, skin rashes, and migraines. It requires mandatory labeling due to potential adverse effects and cross-reactivity with aspirin sensitivity.",
                            "item": ingredient
                        })
                    elif color_code.lower() in ['127', 'e127']:
                        warnings.append({
                            "type": "additive",
                            "message": f"Contains Erythrosine (E127/Red Dye #3): This synthetic dye is currently under FDA review for potential ban due to studies linking it to thyroid tumors in animals. It's already restricted in several countries and may cause photosensitivity reactions in some individuals.",
                            "item": ingredient
                        })
                    else:
                        warnings.append({
                            "type": "additive",
                            "message": f"Contains artificial color ({color_code}): Synthetic food dyes have been associated with behavioral changes in children, including increased hyperactivity and attention difficulties. Long-term consumption may contribute to allergic sensitization and potential carcinogenic risks.",
                            "item": ingredient
                        })
                else:
                    warnings.append({
                        "type": "additive",
                        "message": f"Contains artificial coloring agents: These synthetic dyes serve no nutritional purpose and may contribute to hyperactivity disorders in children. Some artificial colors have been linked to allergic reactions and potential long-term health concerns including carcinogenic risk.",
                        "item": ingredient
                    })
            
            if 'acidity regulator' in ingredient_lower or 'acidity regulators' in ingredient_lower:
                additive_count += 1
                numbers = re.findall(r'\b\d{3}\b', ingredient)
                if numbers:
                    regulator_details = []
                    for num in numbers:
                        if num == '330':
                            regulator_details.append("Citric Acid (330) - generally safe but can cause tooth enamel erosion with frequent exposure")
                        elif num == '296':
                            regulator_details.append("Malic Acid (296) - may cause digestive discomfort in sensitive individuals")
                        elif num == '334':
                            regulator_details.append("Tartaric Acid (334) - can cause gastric irritation in large amounts")
                        else:
                            regulator_details.append(f"E{num} - requires further investigation for specific health impacts")
                    
                    warnings.append({
                        "type": "additive",
                        "message": f"Contains multiple acidity regulators: {'; '.join(regulator_details)}. While generally recognized as safe, the combination of multiple acid regulators may contribute to digestive sensitivity and dental enamel weakening over time.",
                        "item": ingredient
                    })
                else:
                    warnings.append({
                        "type": "additive",
                        "message": f"Contains acidity regulators: These chemical compounds alter food pH and may cause digestive discomfort in sensitive individuals. Long-term consumption of multiple acid regulators can contribute to tooth enamel erosion and gastrointestinal irritation.",
                        "item": ingredient
                    })
            
            if 'preservative' in ingredient_lower:
                additive_count += 1
                warnings.append({
                    "type": "additive",
                    "message": f"Contains chemical preservatives: These synthetic compounds extend shelf life but may disrupt gut microbiome balance and contribute to allergic sensitization. Some preservatives have been linked to behavioral changes in children and potential carcinogenic effects with long-term exposure.",
                    "item": ingredient
                })
            
            # Check for other additive numbers
            if re.search(r'\b\d{3}\b', ingredient) and 'acidity regulator' not in ingredient_lower and 'colour' not in ingredient_lower:
                additive_count += 1
                warnings.append({
                    "type": "additive",
                    "message": f"Contains numbered food additive: This processed chemical compound serves a technological function but provides no nutritional value. Regular consumption of multiple additives may contribute to cumulative toxic load and potential health impacts including allergic reactions.",
                    "item": ingredient
                })
            
            # Enhanced allergen analysis
            allergen_details = {
                'sesame': 'Sesame is a potent allergen that can cause severe anaphylactic reactions even in trace amounts. Cross-contamination is common in food processing facilities, making it particularly dangerous for sensitive individuals.',
                'soy': 'Soy contains natural estrogen-like compounds (phytoestrogens) that may interfere with hormone balance, particularly concerning for children and pregnant women. It\'s also a common allergen that can cause digestive issues and skin reactions.',
                'wheat': 'Wheat contains gluten proteins that can trigger celiac disease and non-celiac gluten sensitivity, causing intestinal damage, nutrient malabsorption, and systemic inflammation. Modern wheat varieties may be more inflammatory than ancient grains.',
                'milk': 'Dairy products contain lactose and casein proteins that many adults cannot properly digest, leading to digestive distress, inflammation, and potential hormonal disruption from growth hormones used in conventional dairy farming.',
                'egg': 'Eggs are among the most common childhood allergens and can cause severe reactions including anaphylaxis. Factory-farmed eggs may contain antibiotic residues and higher levels of inflammatory omega-6 fatty acids.',
                'fish': 'Fish can contain heavy metals like mercury and persistent organic pollutants that accumulate in body tissues. Fish allergies can be severe and life-threatening, with reactions sometimes triggered by cooking vapors.',
                'peanut': 'Peanut allergies are among the most severe and can cause fatal anaphylactic reactions from trace exposure. Peanuts are also prone to aflatoxin contamination, a potent carcinogenic mold toxin.',
                'tree nut': 'Tree nut allergies often persist into adulthood and can cause severe systemic reactions. Cross-contamination between different nuts is common in processing facilities, increasing exposure risk for sensitive individuals.'
            }
            
            for allergen, details in allergen_details.items():
                if allergen in ingredient_lower:
                    warnings.append({
                        "type": "allergen",
                        "message": f"Contains {allergen.title()} Allergen: {details}",
                        "item": ingredient
                    })
            
            # Enhanced oil analysis
            if 'oil' in ingredient_lower and 'vegetable' in ingredient_lower:
                warnings.append({
                    "type": "processing_concern",
                    "message": f"Contains processed vegetable oils: These highly refined oils are often extracted using chemical solvents and high heat, creating inflammatory trans fats and oxidized compounds. They typically have imbalanced omega-6 to omega-3 ratios that promote systemic inflammation when consumed regularly.",
                    "item": ingredient
                })
        
        # Determine NOVA group with enhanced descriptions
        if additive_count >= 3 or len(ingredients) >= 8:
            nova_group = 4
            nova_description = "Ultra-processed food containing multiple industrial additives and processed ingredients. These foods are associated with increased risk of obesity, cardiovascular disease, type 2 diabetes, and certain cancers due to their high caloric density, inflammatory ingredients, and lack of protective nutrients."
        elif additive_count >= 1 or any('oil' in ing.lower() or 'salt' in ing.lower() for ing in ingredients):
            nova_group = 3
            nova_description = "Processed food with added industrial ingredients including oils, preservatives, or flavor enhancers. While not as harmful as ultra-processed foods, regular consumption may contribute to nutrient displacement and increased intake of inflammatory compounds."
        elif len(ingredients) <= 3:
            nova_group = 1
            nova_description = "Minimally processed whole food with simple preparation methods. These foods retain their natural nutrient profile and beneficial compounds while being safe and convenient to consume."
        else:
            nova_group = 2
            nova_description = "Processed culinary ingredients derived from whole foods through traditional methods. These should be used in moderation to enhance the flavor and preparation of minimally processed foods."
        
        # Generate comprehensive health notes
        health_notes = []
        if nova_group == 4:
            health_notes.append("This ultra-processed product contains multiple synthetic additives that may contribute to chronic inflammation, metabolic dysfunction, and increased disease risk")
            health_notes.append("Consider limiting consumption and choosing whole food alternatives to reduce exposure to potentially harmful processing chemicals")
        if any(w['type'] == 'hidden_sugar' for w in warnings):
            health_notes.append("Multiple added sugars significantly increase caloric density and may contribute to blood sugar dysregulation, dental problems, and metabolic syndrome")
        if any(w['type'] == 'additive' for w in warnings):
            health_notes.append("Chemical additives provide no nutritional value and may cause allergic reactions, behavioral changes, or long-term health concerns with regular consumption")
        if any(w['type'] == 'allergen' for w in warnings):
            health_notes.append("Contains major allergens that can trigger severe reactions - exercise extreme caution if you have known sensitivities or allergies")
        
        if not health_notes:
            health_notes.append("This product appears to have minimal concerning ingredients based on current analysis")
            health_notes.append("However, consider overall dietary pattern and consume as part of a balanced, whole-foods-based diet")
        
        health_notes_str = ". ".join(health_notes)
        
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
            "warnings": [
                {
                    "type": "system", 
                    "message": "Limited analysis available due to system constraints. This product may contain ingredients that require further investigation for potential health impacts. Consider consulting ingredient databases or nutritional professionals for comprehensive analysis.",
                    "item": "System Analysis"
                }
            ],
            "nova_group": 3,
            "nova_description": "Analysis unavailable - classified as processed food as precautionary measure. Many processed foods contain additives and refined ingredients that may have health implications with regular consumption.",
            "health_notes": "Unable to perform comprehensive analysis due to technical limitations. Exercise caution with highly processed foods and prioritize whole, minimally processed alternatives when possible for optimal health outcomes."
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
        "endpoints": ["/analyze_ingredients", "/health", "/documents", "/"]
    }

@app.get("/documents")
async def get_loaded_documents():
    """Get information about loaded documents in the knowledge base"""
    global retriever
    
    if not retriever:
        return {"error": "RAG system not initialized"}
    
    try:
        # Try to get some sample documents to show what's loaded
        sample_query = "food additives"
        docs = retriever.invoke(sample_query)
        
        document_info = {
            "total_documents_in_retriever": len(docs) if docs else 0,
            "sample_sources": []
        }
        
        # Get unique sources from sample documents
        sources = set()
        for doc in docs[:5]:  # Just check first 5 docs
            if hasattr(doc, 'metadata') and doc.metadata:
                if 'source' in doc.metadata:
                    sources.add(doc.metadata['source'])
                elif 'category' in doc.metadata:
                    sources.add(f"Built-in: {doc.metadata['category']}")
        
        document_info["sample_sources"] = list(sources)
        return document_info
        
    except Exception as e:
        return {"error": f"Could not retrieve document info: {e}"}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

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

class IngredientFlag(BaseModel):
    severity: str  # "low", "medium", "high", "critical"
    category: str  # "allergen", "additive", "sugar", "preservative", "processing_concern", etc.
    description: str

class IndividualIngredientAnalysis(BaseModel):
    name: str
    classification: str  # "whole_food", "processed_ingredient", "additive", "sugar", "oil", etc.
    safety_level: str  # "safe", "caution", "concern", "avoid"
    detailed_description: str
    health_impact: str
    flags: List[IngredientFlag]

class IngredientAnalysisResponse(BaseModel):
    product_name: str
    ingredient_analyses: List[IndividualIngredientAnalysis]
    nova_group: int
    nova_description: str
    overall_health_assessment: str
    recommendations: List[str]

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

def analyze_individual_ingredient(ingredient: str) -> IndividualIngredientAnalysis:
    """Analyze a single ingredient in detail"""
    ingredient_lower = ingredient.lower()
    flags = []
    
    # Comprehensive ingredient knowledge base
    ingredient_database = {
        # Sugars and sweeteners
        'sugar': {
            'classification': 'refined_sugar',
            'safety_level': 'caution',
            'description': 'Refined white sugar (sucrose) extracted from sugar cane or sugar beets through industrial processing involving chemical purification.',
            'health_impact': 'Provides empty calories with no nutritional value. Rapidly absorbed causing blood glucose spikes followed by crashes. Regular consumption linked to obesity, diabetes, dental caries, and increased inflammation markers.',
            'flags': [
                {'severity': 'medium', 'category': 'sugar', 'description': 'High glycemic index causes rapid blood sugar spikes'},
                {'severity': 'medium', 'category': 'processing', 'description': 'Highly refined product with all nutrients removed'},
                {'severity': 'low', 'category': 'dental_health', 'description': 'Feeds harmful bacteria causing tooth decay'}
            ]
        },
        'maltodextrin': {
            'classification': 'processed_starch',
            'safety_level': 'concern',
            'description': 'Highly processed polysaccharide derived from corn, potato, rice, or wheat starch through enzymatic or acid hydrolysis.',
            'health_impact': 'Has a higher glycemic index than table sugar, causing more rapid blood glucose elevation. May disrupt gut microbiome balance and contribute to insulin resistance.',
            'flags': [
                {'severity': 'high', 'category': 'sugar', 'description': 'Glycemic index higher than table sugar (105-136)'},
                {'severity': 'medium', 'category': 'gut_health', 'description': 'May negatively impact beneficial gut bacteria'},
                {'severity': 'medium', 'category': 'processing', 'description': 'Heavily processed using enzymes or acids'}
            ]
        },
        # Oils and fats
        'vegetable oil': {
            'classification': 'processed_oil',
            'safety_level': 'caution',
            'description': 'Generic term for oils extracted from various plants including soybean, corn, canola, or sunflower through chemical or mechanical processes.',
            'health_impact': 'Often highly refined using chemical solvents and high heat, creating inflammatory compounds. Typically high in omega-6 fatty acids promoting inflammation.',
            'flags': [
                {'severity': 'medium', 'category': 'processing', 'description': 'May be extracted using chemical solvents like hexane'},
                {'severity': 'medium', 'category': 'inflammation', 'description': 'High omega-6 content can promote inflammatory pathways'},
                {'severity': 'low', 'category': 'oxidation', 'description': 'Prone to oxidation creating harmful free radicals'}
            ]
        },
        # Preservatives
        'sodium benzoate': {
            'classification': 'chemical_preservative',
            'safety_level': 'caution',
            'description': 'Synthetic preservative (E211) commonly used to prevent bacterial and fungal growth in acidic foods and beverages.',
            'health_impact': 'Generally recognized as safe but may cause allergic reactions in sensitive individuals. Can form benzene (a carcinogen) when combined with vitamin C under certain conditions.',
            'flags': [
                {'severity': 'medium', 'category': 'additive', 'description': 'Synthetic chemical preservative with no nutritional value'},
                {'severity': 'low', 'category': 'allergen', 'description': 'May trigger allergic reactions in sensitive individuals'},
                {'severity': 'medium', 'category': 'chemical_reaction', 'description': 'Can form carcinogenic benzene with vitamin C'}
            ]
        },
        # Natural whole foods
        'tomatoes': {
            'classification': 'whole_food',
            'safety_level': 'safe',
            'description': 'Fresh tomatoes are nutrient-dense whole foods rich in lycopene, vitamin C, folate, and potassium.',
            'health_impact': 'Excellent source of antioxidants, particularly lycopene which supports cardiovascular health and may reduce cancer risk. Low in calories and high in water content.',
            'flags': []
        },
        'wheat flour': {
            'classification': 'processed_grain',
            'safety_level': 'caution',
            'description': 'Flour milled from wheat grains, may be enriched with synthetic vitamins and minerals to replace nutrients lost during processing.',
            'health_impact': 'Contains gluten which can trigger autoimmune reactions in susceptible individuals. Refined flour has a high glycemic index and reduced nutritional value compared to whole grain.',
            'flags': [
                {'severity': 'high', 'category': 'allergen', 'description': 'Contains gluten - major allergen causing celiac disease and sensitivity'},
                {'severity': 'medium', 'category': 'processing', 'description': 'Refined product with reduced fiber and nutrients'},
                {'severity': 'low', 'category': 'glycemic', 'description': 'High glycemic index contributes to blood sugar spikes'}
            ]
        }
    }
    
    # Direct database lookup
    for key, data in ingredient_database.items():
        if key in ingredient_lower:
            return IndividualIngredientAnalysis(
                name=ingredient,
                classification=data['classification'],
                safety_level=data['safety_level'],
                detailed_description=data['description'],
                health_impact=data['health_impact'],
                flags=[IngredientFlag(**flag) for flag in data['flags']]
            )
    
    # Pattern-based analysis for ingredients not in database
    classification = "unknown"
    safety_level = "safe"
    description = f"{ingredient} - ingredient analysis in progress."
    health_impact = "Health impact assessment pending detailed analysis."
    flags = []
    
    # Sugar pattern analysis
    sugar_patterns = ['sugar', 'syrup', 'dextrose', 'fructose', 'glucose', 'sucrose', 'maltose', 'lactose']
    if any(pattern in ingredient_lower for pattern in sugar_patterns):
        classification = "sugar_sweetener"
        safety_level = "caution"
        description = f"{ingredient} is a sugar-based sweetener that provides rapid energy through simple carbohydrates."
        health_impact = "Contributes to daily sugar intake. May cause blood glucose elevation and provide empty calories without essential nutrients."
        flags.append(IngredientFlag(severity="medium", category="sugar", description="Added sugar contributes to glycemic load"))
    
    # Color additive analysis
    if 'colour' in ingredient_lower or 'color' in ingredient_lower:
        classification = "artificial_additive"
        safety_level = "concern"
        
        # Extract color codes
        color_match = re.search(r'(\d+[a-z]?)', ingredient, re.IGNORECASE)
        if color_match:
            color_code = color_match.group(1)
            if color_code.lower() in ['102', 'e102']:
                description = f"Tartrazine (E102/Yellow #5) - synthetic azo dye derived from petroleum compounds."
                health_impact = "May trigger hyperactivity in children, allergic reactions including asthma and migraines. Requires mandatory labeling due to adverse effects."
                flags.extend([
                    IngredientFlag(severity="high", category="behavioral", description="Linked to hyperactivity in sensitive children"),
                    IngredientFlag(severity="medium", category="allergen", description="Can cause allergic reactions and asthma"),
                    IngredientFlag(severity="medium", category="additive", description="Synthetic petroleum-derived dye")
                ])
            elif color_code.lower() in ['127', 'e127']:
                description = f"Erythrosine (E127/Red #3) - synthetic xanthene dye under regulatory scrutiny."
                health_impact = "Currently under FDA review for potential ban due to animal studies linking it to thyroid tumors. May cause photosensitivity reactions."
                flags.extend([
                    IngredientFlag(severity="critical", category="regulatory", description="Under FDA review for potential ban"),
                    IngredientFlag(severity="high", category="carcinogenic", description="Linked to thyroid tumors in animal studies"),
                    IngredientFlag(severity="medium", category="additive", description="Synthetic chemical dye")
                ])
            else:
                description = f"Artificial color ({color_code}) - synthetic dye used for food coloring."
                health_impact = "Synthetic food dyes may contribute to behavioral changes and allergic sensitization. No nutritional value."
                flags.extend([
                    IngredientFlag(severity="medium", category="behavioral", description="May affect behavior in sensitive individuals"),
                    IngredientFlag(severity="medium", category="additive", description="Synthetic chemical with no nutritional value")
                ])
        else:
            description = f"{ingredient} - artificial coloring agent of unspecified type."
            health_impact = "Artificial colors serve only aesthetic purposes and may cause adverse reactions in sensitive individuals."
            flags.append(IngredientFlag(severity="medium", category="additive", description="Artificial coloring with no nutritional benefit"))
    
    # Acidity regulator analysis
    if 'acidity regulator' in ingredient_lower:
        classification = "acidity_regulator"
        safety_level = "caution"
        numbers = re.findall(r'\b\d{3}\b', ingredient)
        
        if numbers:
            description = f"Acidity regulator containing E-numbers: {', '.join(numbers)}. These are synthetic compounds used to control food pH."
            health_impact = "Generally recognized as safe but may cause digestive sensitivity. Combination of multiple acid regulators can contribute to dental enamel erosion."
            for num in numbers:
                if num == '330':
                    flags.append(IngredientFlag(severity="low", category="dental", description="Citric acid can erode tooth enamel"))
                elif num == '296':
                    flags.append(IngredientFlag(severity="low", category="digestive", description="Malic acid may cause digestive discomfort"))
                elif num == '334':
                    flags.append(IngredientFlag(severity="low", category="digestive", description="Tartaric acid can cause gastric irritation"))
                else:
                    flags.append(IngredientFlag(severity="medium", category="additive", description=f"E{num} requires further safety evaluation"))
        else:
            description = f"{ingredient} - chemical compound used to regulate food acidity and pH levels."
            health_impact = "Used to control food pH but may cause digestive sensitivity in some individuals."
            flags.append(IngredientFlag(severity="low", category="digestive", description="May cause digestive sensitivity"))
    
    # Oil analysis
    if 'oil' in ingredient_lower:
        classification = "processed_oil"
        safety_level = "caution"
        if 'vegetable' in ingredient_lower or 'edible' in ingredient_lower:
            description = f"{ingredient} - processed oil extracted from plant sources through mechanical or chemical methods."
            health_impact = "Highly refined oils may contain inflammatory compounds and have imbalanced fatty acid profiles promoting systemic inflammation."
            flags.extend([
                IngredientFlag(severity="medium", category="processing", description="May involve chemical extraction methods"),
                IngredientFlag(severity="medium", category="inflammation", description="High omega-6 content can promote inflammation")
            ])
    
    # Allergen analysis
    allergens = {
        'wheat': {'severity': 'high', 'description': 'Contains gluten - major allergen'},
        'milk': {'severity': 'high', 'description': 'Dairy allergen - contains lactose and casein'},
        'egg': {'severity': 'high', 'description': 'Common allergen that can cause anaphylaxis'},
        'soy': {'severity': 'medium', 'description': 'Contains phytoestrogens and allergenic proteins'},
        'sesame': {'severity': 'critical', 'description': 'Potent allergen causing severe reactions'},
        'peanut': {'severity': 'critical', 'description': 'Life-threatening allergen with cross-contamination risks'},
        'fish': {'severity': 'high', 'description': 'May contain heavy metals and cause severe allergic reactions'},
        'tree nut': {'severity': 'high', 'description': 'Persistent allergen with cross-contamination concerns'}
    }
    
    for allergen, info in allergens.items():
        if allergen in ingredient_lower:
            flags.append(IngredientFlag(
                severity=info['severity'], 
                category="allergen", 
                description=info['description']
            ))
            if safety_level == "safe":
                safety_level = "caution" if info['severity'] in ['medium'] else "concern"
    
    # Preservative analysis
    preservatives = ['preservative', 'sodium benzoate', 'potassium sorbate', 'calcium propionate']
    if any(pres in ingredient_lower for pres in preservatives):
        classification = "chemical_preservative"
        safety_level = "caution"
        description = f"{ingredient} - synthetic preservative used to extend shelf life by inhibiting microbial growth."
        health_impact = "May disrupt gut microbiome balance and cause allergic reactions in sensitive individuals."
        flags.append(IngredientFlag(severity="medium", category="preservative", description="Synthetic preservative may affect gut health"))
    
    # Additive number detection
    if re.search(r'\b[Ee]?\d{3}\b', ingredient):
        if classification == "unknown":
            classification = "food_additive"
            safety_level = "caution"
        description += f" Contains numbered food additive indicating industrial processing."
        flags.append(IngredientFlag(severity="medium", category="additive", description="Numbered food additive with technological function"))
    
    # Default safe classification for unmatched whole foods
    if classification == "unknown" and not flags:
        whole_food_indicators = ['tomato', 'onion', 'garlic', 'salt', 'pepper', 'herbs', 'spice']
        if any(indicator in ingredient_lower for indicator in whole_food_indicators):
            classification = "whole_food"
            safety_level = "safe"
            description = f"{ingredient} appears to be a whole food ingredient with minimal processing."
            health_impact = "Likely provides nutritional value with minimal processing concerns."
    
    return IndividualIngredientAnalysis(
        name=ingredient,
        classification=classification,
        safety_level=safety_level,
        detailed_description=description,
        health_impact=health_impact,
        flags=flags
    )

async def analyze_with_rag(ingredients: List[str]) -> Dict[str, Any]:
    """Analyze ingredients using RAG system with individual ingredient analysis"""
    global retriever
    
    if not retriever:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    # Analyze each ingredient individually
    ingredient_analyses = []
    additive_count = 0
    high_concern_count = 0
    allergen_count = 0
    
    for ingredient in ingredients:
        analysis = analyze_individual_ingredient(ingredient)
        ingredient_analyses.append(analysis)
        
        # Count different types for NOVA classification
        if analysis.classification in ['food_additive', 'chemical_preservative', 'artificial_additive']:
            additive_count += 1
        
        if analysis.safety_level in ['concern', 'avoid']:
            high_concern_count += 1
            
        if any(flag.category == 'allergen' for flag in analysis.flags):
            allergen_count += 1
    
    # Determine NOVA group
    if additive_count >= 3 or len(ingredients) >= 8:
        nova_group = 4
        nova_description = "Ultra-processed food containing multiple industrial additives and processed ingredients. Associated with increased risk of obesity, cardiovascular disease, type 2 diabetes, and certain cancers."
    elif additive_count >= 1 or any(analysis.classification in ['processed_oil', 'refined_sugar'] for analysis in ingredient_analyses):
        nova_group = 3
        nova_description = "Processed food with added industrial ingredients. While not as harmful as ultra-processed foods, regular consumption may contribute to nutrient displacement and inflammatory compound intake."
    elif len(ingredients) <= 3 and all(analysis.classification == 'whole_food' for analysis in ingredient_analyses):
        nova_group = 1
        nova_description = "Minimally processed whole food retaining natural nutrient profile and beneficial compounds while being safe and convenient to consume."
    else:
        nova_group = 2
        nova_description = "Processed culinary ingredients derived from whole foods. Should be used in moderation to enhance flavor and preparation of minimally processed foods."
    
    # Generate overall health assessment
    if high_concern_count >= 3:
        health_assessment = "High concern - multiple ingredients with significant health risks identified. Consider avoiding or limiting consumption."
    elif high_concern_count >= 1 or allergen_count >= 2:
        health_assessment = "Moderate concern - some ingredients may pose health risks. Exercise caution, especially if you have sensitivities."
    elif additive_count >= 3:
        health_assessment = "Mild concern - contains multiple additives but no severe health risks identified. Consume in moderation as part of balanced diet."
    else:
        health_assessment = "Low concern - most ingredients appear safe with minimal processing. Consider overall dietary pattern and balance."
    
    # Generate personalized recommendations
    recommendations = []
    
    if allergen_count > 0:
        recommendations.append("Check all ingredients carefully if you have known food allergies or sensitivities")
    
    if additive_count >= 3:
        recommendations.append("Consider choosing products with fewer artificial additives and preservatives")
    
    if any(analysis.classification == 'refined_sugar' for analysis in ingredient_analyses):
        recommendations.append("Monitor total daily sugar intake from all sources to maintain healthy blood glucose levels")
    
    if nova_group >= 3:
        recommendations.append("Balance processed foods with whole, minimally processed alternatives in your diet")
    
    if high_concern_count > 0:
        recommendations.append("Consult with healthcare professionals if you have specific health conditions or dietary restrictions")
    
    if not recommendations:
        recommendations.append("This product appears relatively safe - maintain overall dietary variety and moderation")
        recommendations.append("Focus on whole foods and minimally processed options when possible")
    
    return {
        "ingredient_analyses": ingredient_analyses,
        "nova_group": nova_group,
        "nova_description": nova_description,
        "overall_health_assessment": health_assessment,
        "recommendations": recommendations
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
    Analyze food ingredients individually and return detailed safety information
    
    Args:
        request: IngredientListRequest containing list of ingredients
        
    Returns:
        IngredientAnalysisResponse with individual ingredient analyses
    """
    try:
        if not request.ingredients:
            raise HTTPException(status_code=400, detail="No ingredients provided")
        
        # Parse and clean ingredients
        product_name, clean_ingredients = clean_and_parse_ingredients(request.ingredients)
        
        if not clean_ingredients:
            raise HTTPException(status_code=400, detail="No valid ingredients found")
        
        # Analyze with enhanced individual analysis
        analysis_result = await analyze_with_rag(clean_ingredients)
        
        # Construct response with individual ingredient analyses
        response = IngredientAnalysisResponse(
            product_name=product_name,
            ingredient_analyses=analysis_result["ingredient_analyses"],
            nova_group=analysis_result["nova_group"],
            nova_description=analysis_result["nova_description"],
            overall_health_assessment=analysis_result["overall_health_assessment"],
            recommendations=analysis_result["recommendations"]
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        
        # Enhanced fallback response with individual analysis
        fallback_analyses = []
        for ingredient in request.ingredients:
            fallback_analyses.append(IndividualIngredientAnalysis(
                name=ingredient,
                classification="unknown",
                safety_level="caution",
                detailed_description=f"{ingredient} - comprehensive analysis unavailable due to system limitations.",
                health_impact="Unable to assess health impact. Consider consulting ingredient databases or nutrition professionals.",
                flags=[IngredientFlag(
                    severity="medium",
                    category="system",
                    description="Analysis limited due to technical constraints"
                )]
            ))
        
        return IngredientAnalysisResponse(
            product_name="Product Analysis",
            ingredient_analyses=fallback_analyses,
            nova_group=3,
            nova_description="Classification unavailable - using precautionary processed food designation.",
            overall_health_assessment="Unable to provide comprehensive assessment due to system limitations.",
            recommendations=["Consult nutrition professionals for detailed ingredient analysis", "Exercise caution with heavily processed foods"]
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

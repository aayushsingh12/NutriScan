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
        pdf_files = ["chapter_3.pdf", "appendix_a_b.pdf", "banned_food_additives.pdf", "openknowledge.pdf"]
        
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
            search_kwargs={
                "k": 12,  # Increased from 8 to get more context with new PDF
                "score_threshold": 0.5  # Only return chunks with similarity > 0.5
            }
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
    """Parse and clean the raw OCR ingredient list"""
    
    # Join all ingredients with spaces and normalize
    full_text = " ".join(raw_ingredients)
    full_text = re.sub(r'\s+', ' ', full_text).strip()
    
    # Extract product name
    product_name = "NAMKEEN" if "NAMKEEN" in full_text else "Snack Product"
    
    # Remove obvious OCR artifacts and standalone numbers
    full_text = re.sub(r'\b\d+\s*,?\s*$', '', full_text)  # Remove trailing standalone numbers
    full_text = re.sub(r'^\d+\s+', '', full_text)  # Remove leading standalone numbers
    full_text = re.sub(r'\s+\d+\s+', ' ', full_text)  # Remove middle standalone numbers
    
    # Remove product codes like "NAMKEEN 151", "Meal 44", etc.
    full_text = re.sub(r'\b(NAMKEEN|Meal|Cereal Products)\s+\d+\s*,?\s*', '', full_text)
    
    # Clean up the text
    full_text = re.sub(r'\s*,\s*,\s*', ', ', full_text)  # Fix double commas
    full_text = re.sub(r'\s+', ' ', full_text)  # Normalize spaces
    full_text = full_text.strip(' ,')  # Remove leading/trailing commas and spaces
    
    # Split by commas but handle complex cases
    ingredients = []
    parts = [part.strip() for part in full_text.split(',')]
    
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        
        # Skip empty parts or pure numbers
        if not part or re.match(r'^\d+$', part) or len(part) < 2:
            i += 1
            continue
        
        # Handle multi-word ingredients that got split
        if part.endswith('Oil') and i + 1 < len(parts) and not parts[i + 1].strip():
            # Handle cases like "Sesame" "Oil ," - combine them
            combined = part
            j = i + 1
            while j < len(parts) and (not parts[j].strip() or parts[j].strip() == ''):
                j += 1
            if j < len(parts) and 'oil' in parts[j].lower():
                combined = part + " " + parts[j].strip()
                i = j
            ingredients.append(combined)
        elif 'Edible Vegetable Oil' in part:
            # Handle complex oil descriptions
            oil_desc = part
            # Look for continuation in next parts
            j = i + 1
            while j < len(parts) and j < i + 3:  # Look ahead max 3 parts
                next_part = parts[j].strip()
                if next_part and ('oil' in next_part.lower() or 'Oil' in next_part):
                    oil_desc += ", " + next_part
                    i = j
                    break
                j += 1
            ingredients.append(oil_desc)
        elif 'Flavour' in part and 'Natural' in part:
            # Handle flavor descriptions that might be split
            flavor_desc = part
            j = i + 1
            while j < len(parts) and j < i + 3:
                next_part = parts[j].strip()
                if next_part and ('Nature' in next_part or 'Identical' in next_part or 'Flavouring' in next_part or 'Substances' in next_part):
                    flavor_desc += ", " + next_part
                    i = j
                else:
                    break
                j += 1
            ingredients.append(flavor_desc)
        elif 'Acidity Regulators' in part:
            # Handle acidity regulators with numbers
            acidity_desc = part
            numbers = re.findall(r'\b\d{3}\b', acidity_desc)
            if numbers:
                ingredients.append(f"Acidity Regulators ({', '.join(numbers)})")
            else:
                ingredients.append(acidity_desc)
        elif 'Colour' in part:
            # Handle color additives
            color_match = re.search(r'(\d+[a-z]?)', part, re.IGNORECASE)
            if color_match:
                ingredients.append(f"Colour ({color_match.group(1)})")
            else:
                ingredients.append(part)
        else:
            # Clean and add regular ingredients
            cleaned = re.sub(r'\s+\d+\s*$', '', part)  # Remove trailing numbers
            cleaned = re.sub(r'^\d+\s+', '', cleaned)  # Remove leading numbers
            cleaned = cleaned.strip()
            
            if len(cleaned) > 2 and not re.match(r'^\d+$', cleaned):
                ingredients.append(cleaned)
        
        i += 1
    
    # Final cleanup and deduplication
    final_ingredients = []
    seen = set()
    
    for ing in ingredients:
        ing = ing.strip(' ,')
        
        # Skip if empty or too short
        if len(ing) < 3:
            continue
            
        # Skip if already seen (case insensitive)
        if ing.lower() in seen:
            continue
            
        seen.add(ing.lower())
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

async def analyze_single_ingredient_with_rag(ingredient: str) -> IndividualIngredientAnalysis:
    """Analyze a single ingredient using RAG system"""
    global retriever
    
    if not retriever:
        # Fallback to pattern analysis if RAG not available
        return analyze_individual_ingredient(ingredient)
    
    try:
        # Query RAG system specifically for this ingredient
        rag_query = f"Analyze the food ingredient '{ingredient}'. Provide safety information, health impacts, classification, and any flags or concerns."
        
        relevant_docs = retriever.invoke(rag_query)
        
        # Extract relevant information from RAG results
        rag_context = "\n".join([doc.page_content for doc in relevant_docs[:5]])
        
        # Use RAG context to enhance the analysis
        enhanced_analysis = enhance_analysis_with_rag_context(ingredient, rag_context)
        
        return enhanced_analysis
        
    except Exception as e:
        logger.warning(f"RAG analysis failed for '{ingredient}': {e}")
        # Fallback to pattern analysis
        return analyze_individual_ingredient(ingredient)

def enhance_analysis_with_rag_context(ingredient: str, rag_context: str) -> IndividualIngredientAnalysis:
    """Enhance ingredient analysis using RAG context"""
    
    # Start with pattern-based analysis
    base_analysis = analyze_individual_ingredient(ingredient)
    
    # Enhance with RAG context
    ingredient_lower = ingredient.lower()
    rag_lower = rag_context.lower()
    
    # Look for specific mentions of this ingredient in RAG context
    if ingredient_lower in rag_lower:
        # Extract relevant information from RAG
        if 'allergen' in rag_lower and ingredient_lower in rag_lower:
            base_analysis.flags.append(IngredientFlag(
                severity="high",
                category="allergen", 
                description="Identified as potential allergen in food safety documents"
            ))
        
        if 'banned' in rag_lower and ingredient_lower in rag_lower:
            base_analysis.safety_level = "avoid"
            base_analysis.flags.append(IngredientFlag(
                severity="critical",
                category="regulatory",
                description="May be subject to regulatory restrictions"
            ))
        
        if 'e-number' in rag_lower or 'e number' in rag_lower:
            base_analysis.classification = "food_additive"
            base_analysis.detailed_description += " This ingredient has an E-number designation indicating it's a regulated food additive."
    
    return base_analysis

def generate_overall_assessment(analyses: List[IndividualIngredientAnalysis]) -> Dict[str, Any]:
    """Generate overall assessment from individual ingredient analyses"""
    
    additive_count = sum(1 for a in analyses if 'additive' in a.classification.lower())
    concern_count = sum(1 for a in analyses if a.safety_level in ['concern', 'avoid'])
    allergen_count = sum(1 for a in analyses if any(f.category == 'allergen' for f in a.flags))
    
    # NOVA classification
    if additive_count >= 3 or len(analyses) >= 8:
        nova_group = 4
        nova_desc = "Ultra-processed food with multiple industrial additives"
    elif additive_count >= 1:
        nova_group = 3
        nova_desc = "Processed food with some industrial ingredients"
    else:
        nova_group = 2
        nova_desc = "Minimally processed ingredients"
    
    # Health assessment
    if concern_count >= 2:
        health_assessment = "High concern - multiple problematic ingredients identified"
    elif concern_count >= 1 or allergen_count >= 2:
        health_assessment = "Moderate concern - some ingredients may pose risks"
    else:
        health_assessment = "Low to moderate concern - most ingredients appear acceptable"
    
    # Recommendations
    recommendations = []
    if allergen_count > 0:
        recommendations.append("Check for allergen sensitivities")
    if additive_count >= 3:
        recommendations.append("Consider products with fewer additives")
    if not recommendations:
        recommendations.append("Product appears relatively safe for general consumption")
    
    return {
        "nova_group": nova_group,
        "nova_description": nova_desc,
        "health_assessment": health_assessment,
        "recommendations": recommendations
    }

@app.post("/analyze_ingredients", response_model=IngredientAnalysisResponse)
async def analyze_ingredients(request: IngredientListRequest):
    """
    Analyze food ingredients individually and return detailed safety information
    """
    try:
        if not request.ingredients:
            raise HTTPException(status_code=400, detail="No ingredients provided")
        
        # NO COMPLEX PARSING - just clean each ingredient individually
        clean_ingredients = []
        for ingredient in request.ingredients:
            cleaned = ingredient.strip()
            if len(cleaned) > 1:  # Keep anything longer than 1 character
                clean_ingredients.append(cleaned)
        
        if not clean_ingredients:
            raise HTTPException(status_code=400, detail="No valid ingredients found")
        
        # Analyze each ingredient completely separately
        ingredient_analyses = []
        
        logger.info(f"Analyzing {len(clean_ingredients)} ingredients individually:")
        for i, ingredient in enumerate(clean_ingredients):
            logger.info(f"  {i+1}. Analyzing: '{ingredient}'")
            
            # Each ingredient gets its own individual analysis
            analysis = await analyze_single_ingredient_with_rag(ingredient)
            ingredient_analyses.append(analysis)
        
        # Generate overall assessment from individual analyses
        overall_result = generate_overall_assessment(ingredient_analyses)
        
        return IngredientAnalysisResponse(
            product_name="Food Product",
            ingredient_analyses=ingredient_analyses,
            nova_group=overall_result["nova_group"],
            nova_description=overall_result["nova_description"], 
            overall_health_assessment=overall_result["health_assessment"],
            recommendations=overall_result["recommendations"]
        )
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Terminal-Based Food Safety RAG System
# Modern RAG System for Food Safety Analysis with Terminal Interface

import subprocess
import time
import requests
import os
import psutil
import json
from pathlib import Path
import sys
import random

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q",
            "langchain", "langchain_community", "faiss-cpu",
            "sentence-transformers", "pypdf", "requests", "psutil"
        ], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def kill_existing_ollama():
    """Kill any existing Ollama processes"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] and 'ollama' in proc.info['name'].lower():
                print(f"🔄 Killing existing Ollama process: {proc.info['pid']}")
                proc.terminate()
                proc.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            pass

def install_ollama():
    """Install Ollama with proper error handling"""
    try:
        print("🔄 Installing Ollama...")
        result = subprocess.run(
            ['curl', '-fsSL', 'https://ollama.com/install.sh'],
            capture_output=True, text=True, check=True, timeout=60
        )
        subprocess.run(['bash', '-c', result.stdout], check=True, timeout=120)
        print("✅ Ollama installed successfully")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"❌ Failed to install Ollama: {e}")
        return False

def wait_for_ollama_server(max_wait=60):
    """Wait for Ollama server to become available"""
    print("⏳ Waiting for Ollama server to start...")
    for i in range(max_wait):
        try:
            response = requests.get('http://127.0.0.1:11434/api/version', timeout=2)
            if response.status_code == 200:
                print(f"✅ Ollama server ready after {i+1} seconds")
                return True
        except requests.RequestException:
            if i % 10 == 0 and i > 0:
                print(f"   Still waiting... ({i}s)")
            time.sleep(1)

    print("❌ Ollama server failed to start within timeout")
    return False

def start_ollama_server():
    """Start Ollama server in background"""
    try:
        kill_existing_ollama()
        print("🚀 Starting Ollama server...")

        process = subprocess.Popen(
            ['ollama', 'serve'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if wait_for_ollama_server():
            return process
        else:
            process.terminate()
            return None

    except Exception as e:
        print(f"❌ Error starting Ollama server: {e}")
        return None

def pull_model(model_name="llama3.2"):
    """Pull Ollama model with retry logic"""
    print(f"📥 Pulling {model_name} model...")
    max_retries = 3

    for attempt in range(max_retries):
        try:
            print(f"   Attempt {attempt + 1}/{max_retries}")
            result = subprocess.run(
                ['ollama', 'pull', model_name],
                capture_output=True, text=True, timeout=600
            )
            if result.returncode == 0:
                print(f"✅ Model {model_name} pulled successfully")
                return True
            else:
                print(f"❌ Attempt {attempt + 1} failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"⏱️ Attempt {attempt + 1} timed out")
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} error: {e}")

        if attempt < max_retries - 1:
            print("   Retrying in 10 seconds...")
            time.sleep(10)

    print(f"❌ Failed to pull {model_name} after {max_retries} attempts")
    return False

def download_with_retry(url, path, description, max_retries=3):
    """Download file with retry logic"""
    print(f"📥 Downloading {description}...")

    for attempt in range(max_retries):
        try:
            print(f"   Attempt {attempt + 1}/{max_retries}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()

            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            if Path(path).exists() and Path(path).stat().st_size > 0:
                print(f"✅ {description} downloaded successfully ({Path(path).stat().st_size} bytes)")
                return True
            else:
                print(f"❌ {description} file appears empty")

        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("   Retrying in 5 seconds...")
                time.sleep(5)

    print(f"❌ Failed to download {description}")
    return False

def show_progress_bar(current, total, width=50):
    """Display a progress bar"""
    progress = int(width * current / total)
    bar = '█' * progress + '░' * (width - progress)
    percentage = int(100 * current / total)
    print(f'\r[{bar}] {percentage}%', end='', flush=True)
    if current == total:
        print()

def initialize_system():
    """Initialize the entire RAG system"""
    print("="*60)
    print("🥗 NUTRISCAN - FOOD SAFETY RAG SYSTEM INITIALIZATION")
    print("="*60)

    total_steps = 8
    current_step = 0

    try:
        # Step 1: Install dependencies
        current_step += 1
        print(f"\nStep {current_step}/{total_steps}: Installing dependencies")
        show_progress_bar(current_step, total_steps)
        if not install_dependencies():
            return False, None, None

        # Import libraries after installation
        print("📚 Importing libraries...")
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain_community.llms import Ollama
        from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.chains import create_retrieval_chain
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.documents import Document

        # Step 2: Setup Ollama
        current_step += 1
        print(f"\nStep {current_step}/{total_steps}: Setting up Ollama")
        show_progress_bar(current_step, total_steps)

        OLLAMA_MODEL = "llama3"
        if install_ollama():
            server_process = start_ollama_server()
            if server_process:
                model_pulled = False
                for model in ["llama3.2", "llama3.1", "llama3"]:
                    if pull_model(model):
                        OLLAMA_MODEL = model
                        model_pulled = True
                        break

                if not model_pulled:
                    print("⚠️ No model pulled, using default")
            else:
                print("⚠️ Server failed to start, continuing...")
        else:
            print("⚠️ Installation failed, continuing...")

        # Step 3: Download data sources
        current_step += 1
        print(f"\nStep {current_step}/{total_steps}: Downloading food safety databases")
        show_progress_bar(current_step, total_steps)

        downloads = [
            ("https://fssai.gov.in/upload/uploadfiles/files/appendix_a_and_b_revised(30-12-2011).pdf",
             "appendix_a_b.pdf", "FSSAI Appendix A and B"),
            ("https://www.fssai.gov.in/upload/uploadfiles/files/18_%20Chapter%203%20(Substances%20added%20to%20food)_compressed.pdf",
             "chapter_3.pdf", "FSSAI Chapter 3"),
            ("https://static.openfoodfacts.org/data/taxonomies/additives.json",
             "additives.json", "Open Food Facts Additives")
        ]

        downloaded_files = {}
        for url, path, desc in downloads:
            downloaded_files[path] = download_with_retry(url, path, desc)

        # Step 4: Process documents
        current_step += 1
        print(f"\nStep {current_step}/{total_steps}: Processing documents")
        show_progress_bar(current_step, total_steps)

        documents = []

        # Load PDFs
        pdf_loaders = [
            ("appendix_a_b.pdf", "FSSAI Appendix A and B"),
            ("chapter_3.pdf", "FSSAI Chapter 3")
        ]

        for pdf_file, description in pdf_loaders:
            if downloaded_files.get(pdf_file, False):
                try:
                    loader = PyPDFLoader(pdf_file)
                    docs = loader.load()
                    if docs:
                        documents.extend(docs)
                        print(f"✅ Loaded {len(docs)} pages from {description}")
                    else:
                        print(f"⚠️ No content extracted from {description}")
                except Exception as e:
                    print(f"❌ Failed to load {description}: {e}")

        # Load JSON additives data
        if downloaded_files.get("additives.json", False):
            try:
                with open("additives.json", 'r', encoding='utf-8') as f:
                    additives_data = json.load(f)

                additive_count = 0
                for key, value in additives_data.items():
                    if isinstance(value, dict) and len(key) > 0:
                        def safe_extract(data, path):
                            try:
                                if isinstance(data, dict):
                                    if 'en' in data:
                                        return str(data['en'])
                                    elif len(data) > 0:
                                        return str(list(data.values())[0])
                                return str(data) if data else 'N/A'
                            except:
                                return 'N/A'

                        name = safe_extract(value.get('name', {}), 'name')
                        vegan = safe_extract(value.get('vegan', {}), 'vegan')
                        vegetarian = safe_extract(value.get('vegetarian', {}), 'vegetarian')
                        description = safe_extract(value.get('description', {}), 'description')

                        content = f"""Additive Code: {key}
Name: {name}
Vegan Status: {vegan}
Vegetarian Status: {vegetarian}
Description: {description}"""

                        documents.append(Document(
                            page_content=content,
                            metadata={"source": "additives.json", "id": key, "category": "additive"}
                        ))
                        additive_count += 1

                print(f"✅ Processed {additive_count} food additives")

            except Exception as e:
                print(f"❌ Failed to process additives: {e}")

        # Add comprehensive knowledge base
        knowledge_base = [
            Document(
                page_content="""NOVA Group 1: Unprocessed or Minimally Processed Foods
Definition: Natural foods altered only by removal of inedible parts, drying, crushing, grinding, fractioning, filtering, roasting, boiling, non-alcoholic fermentation, pasteurization, chilling, freezing, placing in containers and vacuum-packaging.
Examples: Fresh, dried, ground, chilled, frozen, pasteurized fruits and vegetables; grains like brown rice, corn kernels, wheat berries; legumes like beans, lentils, chickpeas; nuts and seeds; meat, poultry, fish and seafood; eggs; milk.
Health Impact: Generally healthiest option, rich in nutrients, fiber, and beneficial compounds.""",
                metadata={"category": "nova_classification", "group": "1"}
            ),
            Document(
                page_content="""NOVA Group 4: Ultra-Processed Foods
Definition: Industrial formulations made entirely or mostly from substances extracted from foods (oils, fats, sugar, starch, and proteins), derived from food constituents (hydrogenated fats and modified starch), or synthesized in laboratories from food substrates or other organic sources (flavor enhancers, colors, and emulsifiers).
Examples: Carbonated soft drinks, sweet or savory packaged snacks, ice-cream, chocolate, candies, mass-produced packaged breads and buns, margarines, breakfast cereals, cereal and energy bars, instant soups, many ready-to-heat products.
Health Concerns: Linked to obesity, type 2 diabetes, cardiovascular disease, and some cancers. High in calories, sugar, unhealthy fats, and sodium while being low in protein, fiber, and micronutrients.""",
                metadata={"category": "nova_classification", "group": "4"}
            ),
            Document(
                page_content="""Hidden Sugars and Alternative Names: Complete List
Common hidden sugars include: agave nectar, apple juice concentrate, barley malt, brown rice syrup, cane juice, cane sugar, coconut sugar, corn syrup, corn syrup solids, date sugar, dextran, dextrose, evaporated cane juice, fructose, fruit juice concentrate, glucose, glucose syrup, golden syrup, high fructose corn syrup (HFCS), honey, invert sugar, lactose, malt syrup, maltodextrin, maltose, maple syrup, molasses, raw sugar, rice syrup, sucrose, treacle, turbinado sugar.
Health Impact: All contribute to daily sugar intake. Fruit juice concentrates are particularly concerning as they concentrate natural sugars while removing beneficial fiber.""",
                metadata={"category": "hidden_sugars"}
            ),
            Document(
                page_content="""Major Food Allergens (EU Top 14 List)
1. Cereals containing gluten: wheat, rye, barley, oats, spelt, kamut, triticale
2. Crustaceans: prawns, crabs, lobster, crayfish, shrimp
3. Eggs: all forms including powdered, liquid
4. Fish: all fish species and fish-derived products
5. Peanuts: groundnuts and all peanut products
6. Soybeans: soy, soya, and all soy derivatives
7. Milk: all dairy products, lactose, casein, whey
8. Tree nuts: almonds, hazelnuts, walnuts, cashews, pecans, Brazil nuts, pistachios, macadamia nuts, Queensland nuts
9. Celery: including celeriac and celery seed
10. Mustard: seeds, leaves, and mustard preparations
11. Sesame seeds: tahini, sesame oil, halva
12. Sulphur dioxide and sulphites: when >10mg/kg or 10mg/L
13. Lupin: legume used in some flours and pastries
14. Molluscs: mussels, oysters, snails, squid, octopus""",
                metadata={"category": "allergens"}
            ),
            Document(
                page_content="""Traffic Light Nutrition Labeling System
GREEN (Low): Fat ≤3g per 100g, Saturated Fat ≤1.5g per 100g, Sugar ≤5g per 100g, Salt ≤0.3g per 100g
AMBER (Medium): Fat 3.1-17.5g per 100g, Saturated Fat 1.6-5g per 100g, Sugar 5.1-22.5g per 100g, Salt 0.31-1.5g per 100g
RED (High): Fat >17.5g per 100g, Saturated Fat >5g per 100g, Sugar >22.5g per 100g, Salt >1.5g per 100g
Recommendation: Choose more green, fewer amber, avoid red when possible.""",
                metadata={"category": "nutritional_guidelines"}
            ),
            Document(
                page_content="""E-Number Classification System
E100-199: Colors (natural and artificial dyes)
E200-299: Preservatives (prevent spoilage and extend shelf life)
E300-399: Antioxidants and acidity regulators (prevent rancidity and control pH)
E400-499: Thickeners, stabilizers, and emulsifiers (improve texture)
E500-599: pH regulators and anti-caking agents (control acidity and prevent clumping)
E600-699: Flavor enhancers (improve taste)
E700-799: Antibiotics (rarely used in food)
E900-999: Miscellaneous (glazing agents, gases, sweeteners)
E1000+: Additional chemicals as needed""",
                metadata={"category": "e_numbers"}
            )
        ]

        documents.extend(knowledge_base)
        print(f"✅ Added comprehensive knowledge base ({len(knowledge_base)} documents)")
        print(f"📊 Total documents loaded: {len(documents)}")

        # Step 5: Create embeddings
        current_step += 1
        print(f"\nStep {current_step}/{total_steps}: Creating embeddings")
        show_progress_bar(current_step, total_steps)

        embedding_created = False
        embeddings = None

        # Try Ollama embeddings first
        try:
            embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
            test_embedding = embeddings.embed_query("test")
            if test_embedding:
                print(f"✅ Using Ollama embeddings with {OLLAMA_MODEL}")
                embedding_created = True
        except Exception as e:
            print(f"⚠️ Ollama embeddings failed: {e}")

        # Fallback to HuggingFace
        if not embedding_created:
            try:
                print("🔄 Falling back to HuggingFace embeddings...")
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                test_embedding = embeddings.embed_query("test")
                if test_embedding:
                    print("✅ Using HuggingFace embeddings (all-MiniLM-L6-v2)")
                    embedding_created = True
            except Exception as e:
                print(f"❌ HuggingFace embeddings failed: {e}")

        if not embedding_created:
            print("❌ Could not create embeddings. Exiting.")
            return False, None, None

        # Step 6: Create vector store
        current_step += 1
        print(f"\nStep {current_step}/{total_steps}: Building vector database")
        show_progress_bar(current_step, total_steps)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""],
            length_function=len
        )

        split_docs = text_splitter.split_documents(documents)
        print(f"📝 Split documents into {len(split_docs)} chunks")

        try:
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            print("✅ FAISS vector store created successfully")
        except Exception as e:
            print(f"❌ Failed to create vector store: {e}")
            return False, None, None

        # Step 7: Setup LLM
        current_step += 1
        print(f"\nStep {current_step}/{total_steps}: Initializing language model")
        show_progress_bar(current_step, total_steps)

        llm_connected = False
        try:
            llm = Ollama(model=OLLAMA_MODEL, temperature=0.1)
            test_response = llm.invoke("Test")
            if test_response:
                print(f"✅ Connected to Ollama LLM ({OLLAMA_MODEL})")
                llm_connected = True
        except Exception as e:
            print(f"⚠️ Ollama LLM connection failed: {e}")
            print("⚠️ Will use fallback mode (context-based responses only)")

        # Step 8: Create RAG chain
        current_step += 1
        print(f"\nStep {current_step}/{total_steps}: Finalizing RAG system")
        show_progress_bar(current_step, total_steps)

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}
        )

        prompt_template = """You are an expert food safety and nutrition analyst. Use ONLY the provided context to answer questions about food ingredients, additives, allergens, nutrition, and safety.

Instructions:
- Answer based strictly on the provided context
- If information is not available in my knowledge base, clearly state "This information is not available in my knowledge base"
- For health concerns, mention specific risks and recommendations
- For allergens, include severity levels and cross-contamination warnings
- For additives, explain their function and any safety concerns
- Use specific examples when available
- Be concise but comprehensive

Context Information:
{context}

User Question: {input}

Expert Analysis:"""

        prompt = ChatPromptTemplate.from_template(prompt_template)

        if llm_connected:
            document_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, document_chain)
            print("✅ RAG chain created successfully")
        else:
            rag_chain = None
            print("⚠️ Limited functionality - vector search only")

        return True, rag_chain, retriever

    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False, None, None

def analyze_food_ingredient(question, rag_chain, retriever):
    """Analyze food ingredients using the RAG system"""
    try:
        if rag_chain:
            result = rag_chain.invoke({"input": question})
            return result['answer'], result.get('context', [])
        elif retriever:
            docs = retriever.get_relevant_documents(question)
            if docs:
                return f"Based on available information: {docs[0].page_content[:500]}...", docs
            else:
                return "No relevant information found in the database.", []
        else:
            return "System not properly initialized.", []
    except Exception as e:
        return f"Error processing question: {e}", []

def display_help():
    """Display help information"""
    print("\n" + "="*60)
    print("🥗 NUTRISCAN - HELP MENU")
    print("="*60)
    print("Available commands:")
    print("  help     - Show this help menu")
    print("  examples - Show example questions")
    print("  topics   - Show popular topics")
    print("  clear    - Clear screen")
    print("  status   - Show system status")
    print("  quit     - Exit the system")
    print()
    print("Simply type your food safety question and press Enter!")
    print("="*60)

def show_examples():
    """Show example questions"""
    examples = [
        "What is E250 and is it safe?",
        "Explain NOVA food classifications",
        "Is peanut a major allergen?",
        "What are hidden sugars?",
        "How do traffic light nutrition labels work?",
        "What are the health risks of ultra-processed foods?",
        "Is monosodium glutamate (MSG) safe to consume?",
        "Which food additives should people with asthma avoid?",
        "What's the difference between natural and artificial flavors?",
        "What does 'E numbers' mean on food labels?"
    ]

    print("\n🔍 EXAMPLE QUESTIONS:")
    print("-" * 40)
    for i, example in enumerate(examples, 1):
        print(f"{i:2d}. {example}")
    print()

def show_topics():
    """Show popular topics"""
    topics = {
        "🍯 Hidden Sugars": "Learn about hidden sugars and where they're found",
        "🥜 Food Allergens": "Major food allergens and cross-contamination risks",
        "🧪 E-Numbers": "Understanding E-numbers on food labels",
        "🍟 Ultra-Processed": "What are ultra-processed foods and health impacts",
        "🚦 Traffic Light": "Traffic light nutrition labeling system",
        "🌱 Vegan/Vegetarian": "Food additives and dietary restrictions",
        "⚠️ Safety Concerns": "Food additives with potential health risks",
        "🏷️ NOVA Classification": "NOVA food processing categories"
    }

    print("\n📚 POPULAR TOPICS:")
    print("-" * 40)
    for topic, description in topics.items():
        print(f"{topic}: {description}")
    print()

def show_status(rag_chain, retriever):
    """Show system status"""
    print("\n📊 SYSTEM STATUS:")
    print("-" * 30)
    if rag_chain:
        print("✅ AI System: Online (Full functionality)")
    elif retriever:
        print("⚠️ AI System: Limited (Search only)")
    else:
        print("❌ AI System: Offline")

    print("📋 Available Knowledge:")
    print("   - FSSAI food safety regulations")
    print("   - Food additive database")
    print("   - Allergen information")
    print("   - NOVA food classifications")
    print("   - Nutritional guidelines")
    print("   - E-number explanations")
    print()

def show_colab_instructions():
    """Show Colab-specific setup instructions"""
    print("=" * 70)
    print("🔧 GOOGLE COLAB SETUP INSTRUCTIONS")
    print("=" * 70)
    print("Running in Google Colab detected!")
    print()
    print("📋 To use this system in Colab:")
    print("1. The dependencies are already installed via the !pip command at the top")
    print("2. Run this entire cell to initialize the system")
    print("3. The terminal interface will start automatically")
    print("4. If initialization fails, restart runtime and try again")
    print()
    print("⚠️  Note: Colab sessions are temporary. You'll need to re-run")
    print("    the initialization if your session disconnects.")
    print("=" * 70)
    print()

def check_colab_environment():
    """Check if the code is running in a Google Colab environment."""
    try:
        from google.colab import userdata
        return True
    except ImportError:
        return False

def main():
    """Main function - optimized for both Colab and local environments"""
    # Check environment and show appropriate instructions
    is_colab = check_colab_environment()

    if is_colab:
        show_colab_instructions()

    print("🥗 NutriScan - Food Safety Analysis System")
    if is_colab:
        print("🔧 Running in Google Colab environment")

    print("Starting initialization...")

    success, rag_chain, retriever = initialize_system()

    if not success:
        print("\n❌ System initialization failed.")
        if is_colab:
            print("🔧 In Colab: Try restarting runtime and running again")
            print("   Runtime → Restart runtime, then re-run this cell")
        else:
            print("🔧 Please check your internet connection and try again.")
        return

    print("\n" + "="*60)
    print("✅ NUTRISCAN IS READY!")
    if is_colab:
        print("🔧 Running in Google Colab - terminal interface active")
    print("="*60)
    print("🤖 Ask me about food ingredients, additives, allergens, and nutrition!")
    print("📝 Type 'help' for commands, 'examples' for sample questions")
    print("🚪 Type 'quit' or 'exit' to stop")
    print("="*60)

    question_count = 0

    while True:
        try:
            # Get user input
            user_input = input("\n🔍 Your Question > ").strip()

            # Handle empty input
            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using NutriScan! Stay healthy!")
                if is_colab:
                    print("🔧 To restart the system, re-run this cell")
                break
            elif user_input.lower() in ['help', 'h']:
                display_help()
                continue
            elif user_input.lower() == 'examples':
                show_examples()
                continue
            elif user_input.lower() == 'topics':
                show_topics()
                continue
            elif user_input.lower() == 'clear':
                if is_colab:
                    # In Colab, we can't actually clear but can add some space
                    print("\n" * 50)
                    print("🧹 Screen cleared (in Colab)")
                else:
                    os.system('clear' if os.name == 'posix' else 'cls')
                continue
            elif user_input.lower() == 'status':
                show_status(rag_chain, retriever)
                continue

            # Process the question
            question_count += 1
            print(f"\n🔬 Analyzing question #{question_count}...")
            print("⏳ Please wait...")

            start_time = time.time()
            answer, context = analyze_food_ingredient(user_input, rag_chain, retriever)
            end_time = time.time()

            # Display results
            print("\n" + "="*60)
            print("💡 ANALYSIS RESULTS")
            print("="*60)
            print(f"❓ Question: {user_input}")
            print(f"\n🤖 Analysis:\n{answer}")

            # Show sources if available
            if context and len(context) > 0:
                print(f"\n📚 Sources consulted:")
                for i, doc in enumerate(context[:3], 1):
                    source = doc.metadata.get('source', 'Knowledge Base')
                    category = doc.metadata.get('category', 'General')
                    print(f"   {i}. {category} ({source})")

            print(f"\n⏱️ Response time: {end_time - start_time:.2f} seconds")
            print("="*60)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using NutriScan!")
            if is_colab:
                print("🔧 To restart the system, re-run this cell")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again or type 'help' for assistance.")

if __name__ == "__main__":
    main()

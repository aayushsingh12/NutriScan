import cv2
import json
import re
import sys
import google.generativeai as genai
import os
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API with your API key
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise KeyError("GOOGLE_API_KEY not found in .env file")
    genai.configure(api_key=api_key)
except KeyError as e:
    print(f"Error: {e}")
    print("Please ensure you have a .env file with GOOGLE_API_KEY=your_api_key_here")
    sys.exit(1)

# ...rest of your existing code...
# You can choose a suitable Gemini model
MODEL_NAME = "gemini-1.5-flash"

# ...existing code...

def extract_ingredients_gemini(image_path, skip_words=None):
    if skip_words is None:
        skip_words = ["ingredients", "contains", "proprietary food", "as flavouring agent"]

    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Modified prompt to request cleaner formatting
    prompt = (
        "Extract the list of ingredients from this food label image. "
        "Return the list as a clean, comma-separated string. "
        "Ensure ingredients are complete words, not fragments. "
        "Remove words like 'ingredients', 'contains', 'proprietary food', or 'as flavouring agent'. "
        "Fix any OCR errors and combine split words."
    )

    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content([prompt, img])
    
    # Simplified text processing
    raw_text = response.text.strip()
    
    # Split only by commas and handle 'and' separately
    ingredients = [
        item.strip() 
        for item in raw_text.replace(" and ", ", ").split(",")
        if item.strip() and not any(skip.lower() in item.lower() for skip in skip_words)
    ]

    return {"ingredients": ingredients}

# ...existing code...
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ocr_label_gemini.py <image_path>")
        sys.exit(1)

    # The image_path is now correctly taken from the command-line argument
    image_path = sys.argv[1]
    
    try:
        result = extract_ingredients_gemini(image_path)

        # Print JSON output
        json_output = json.dumps(result, indent=4)
        print(json_output)

        # Optional: save to file
        with open("ingredients.json", "w") as f:
            json.dump(result, f, indent=4)
            
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

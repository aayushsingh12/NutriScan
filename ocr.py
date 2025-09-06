import cv2
import easyocr
import json
import re
import sys

# Initialize EasyOCR reader once (GPU if available)
reader = easyocr.Reader(['en'], gpu=True)

def extract_ingredients(image_path, skip_words=None):
    if skip_words is None:
        skip_words = ["ingredients", "contains", "proprietary food", "as flavouring agent"]

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 35, 11
    )

    # Morphological opening to clean small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # OCR directly on numpy array (no disk write)
    texts = reader.readtext(clean, detail=0)

    # Filter OCR results
    ingredients = []
    for text in texts:
        if not any(skip_word.lower() in text.lower() for skip_word in skip_words):
            # Remove unwanted characters and trim
            text = re.sub(r'[^A-Za-z0-9, ]+', '', text).strip()
            if text:
                ingredients.append(text)

    return {"ingredients": ingredients}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ocr_label.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    result = extract_ingredients(image_path)

    # Print JSON output
    json_output = json.dumps(result, indent=4)
    print(json_output)

    # Optional: save to file for RAG
    with open("ingredients.json", "w") as f:
        json.dump(result, f, indent=4)

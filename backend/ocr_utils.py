import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
vision_model = genai.GenerativeModel("models/gemini-1.5-flash")

def gemini_ocr_text_sync(image_bytes):
    try:
        image_part = {"mime_type": "image/jpeg", "data": image_bytes}
        response = vision_model.generate_content([image_part, "Extract all text."])
        print("Gemini OCR Response:", response.text)
        return response.text
    except Exception as e:
        raise RuntimeError(f"Gemini OCR failed: {str(e)}")


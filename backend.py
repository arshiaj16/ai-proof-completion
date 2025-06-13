import os
import json
import re
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Gemini imports
import google.generativeai as genai

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
vision_model = genai.GenerativeModel("models/gemini-1.5-flash")

# Initialize Groq LLM
llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    groq_api_key=GROQ_API_KEY
)
output_parser = StrOutputParser()

# Prompt template for Aadhaar
aadhaar_prompt = PromptTemplate.from_template("""
You are an intelligent document parser.

You will be given raw OCR text from an Aadhaar card.

The text may contain the following fields:
- Full Name
- Date of Birth
- Aadhaar Number (12-digit)
- Gender

Here is the OCR extracted text:

{document_text}

Extract the following fields:
- Full Name
- Date of Birth (YYYY-MM-DD)
- Aadhaar Number (no spaces)
- Gender ("Male", "Female", "Transgender")

Respond ONLY with valid JSON:
{{
  "name": "...",
  "dob": "...",
  "gender": "...",
  "aadhar": "..."
}}
""")

# Prompt template for PAN
pan_prompt = PromptTemplate.from_template("""
You are an intelligent document parser.

Here is the OCR extracted text from an Indian PAN card:

{document_text}

Extract the following fields:
- Name
- Father's Name
- PAN Number

Respond ONLY with valid JSON:
{{
  "name": "...",
  "father_name": "...",
  "pan": "..."
}}
""")

# Clean Aadhaar number by removing spaces
def clean_aadhar_number(aadhar: str) -> str:
    return aadhar.replace(" ", "").strip()

# OCR using Gemini
def gemini_ocr_text(image_bytes):
    image_part = {"mime_type": "image/jpeg", "data": image_bytes}
    response = vision_model.generate_content([image_part, "Extract all text."])
    print("Gemini OCR Response:", response.text)
    return response.text

# Endpoint to parse Aadhaar & PAN and return extracted data
@app.post("/upload-documents/")
async def upload_documents(aadhaar_file: UploadFile = File(...), pan_file: UploadFile = File(...)):
    try:
        # Aadhaar OCR
        aadhaar_bytes = await aadhaar_file.read()
        aadhaar_text = gemini_ocr_text(aadhaar_bytes)

        if not aadhaar_text or len(aadhaar_text) < 20:
            return JSONResponse(content={
                "error": "OCR failed or returned insufficient Aadhaar text.",
                "ocr_output": aadhaar_text
            }, status_code=400)

        # Extract Aadhaar number via regex
        aadhaar_number_match = re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', aadhaar_text)
        aadhaar_number_extracted = aadhaar_number_match.group(0).replace(" ", "") if aadhaar_number_match else ""

        # Aadhaar Parsing via LLM
        aadhaar_chain = aadhaar_prompt | llm | output_parser
        aadhaar_result_raw = aadhaar_chain.invoke({"document_text": aadhaar_text})
        # aadhaar_result_cleaned = aadhaar_result_raw.strip().strip("").strip()

        # Clean LLM output and extract JSON block
        def extract_json_block(text):
            match = re.search(r"{.*}", text, re.DOTALL)
            if match:
                return match.group(0)
            return None

        aadhaar_result_cleaned = extract_json_block(aadhaar_result_raw)

        if not aadhaar_result_cleaned:
            return JSONResponse(
                content={"error": "Failed to find JSON in Aadhaar output", "raw": aadhaar_result_raw},
                status_code=400
            )
        
        try:
            aadhaar_data = json.loads(aadhaar_result_cleaned)
            if not aadhaar_data.get("aadhar") or not re.fullmatch(r"\d{12}", aadhaar_data["aadhar"]):
                if re.fullmatch(r"\d{12}", aadhaar_number_extracted):
                    aadhaar_data["aadhar"] = aadhaar_number_extracted
            aadhaar_data["aadhar"] = clean_aadhar_number(aadhaar_data["aadhar"])
        except Exception:
            return JSONResponse(content={"error": "Failed to parse Aadhaar output", "raw": aadhaar_result_raw}, status_code=400)

        # PAN OCR
        pan_bytes = await pan_file.read()
        pan_text = gemini_ocr_text(pan_bytes)

        pan_chain = pan_prompt | llm | output_parser
        pan_result_raw = pan_chain.invoke({"document_text": pan_text})
        pan_result_cleaned = pan_result_raw.strip().strip("").strip()
        pan_data = json.loads(pan_result_cleaned)

        # Compare Aadhaar and PAN name
        aadhaar_name = aadhaar_data["name"].strip().lower()
        pan_name = pan_data["name"].strip().lower()

        if aadhaar_name != pan_name:
            return JSONResponse(content={
                "error": "Aadhaar and PAN appear to belong to different individuals.",
                "aadhaar_name": aadhaar_data["name"],
                "pan_name": pan_data["name"]
            }, status_code=400)

        # Final return (dictionary instead of DB)
        return JSONResponse(content={
            "message": "Documents parsed successfully.",
            "data": {
                "name": aadhaar_data["name"],
                "dob": aadhaar_data["dob"],
                "gender": aadhaar_data["gender"],
                "father_name": pan_data["father_name"],
                "aadhar": aadhaar_data["aadhar"],
                "pan": pan_data["pan"]
            }
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500) clean data make separarte py for everything without changing the code

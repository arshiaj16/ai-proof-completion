from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
import json
import re

from ocr_utils import gemini_ocr_text_sync
from parsers import llm, output_parser
from prompts import aadhaar_prompt, pan_prompt
from utils import clean_aadhar_number, extract_json_block

router = APIRouter()

@router.post("/upload-documents/")
async def upload_documents(aadhaar_file: UploadFile = File(...), pan_file: UploadFile = File(...)):
    try:
        # Read files
        aadhaar_bytes = await aadhaar_file.read()
        pan_bytes = await pan_file.read()

        # Aadhaar OCR
        aadhaar_text = await run_in_threadpool(gemini_ocr_text_sync, aadhaar_bytes)
        if not aadhaar_text or len(aadhaar_text) < 20:
            return JSONResponse(content={"error": "OCR failed or returned insufficient Aadhaar text."}, status_code=400)

        aadhaar_number_extracted = ""
        match = re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', aadhaar_text)
        if match:
            aadhaar_number_extracted = match.group(0).replace(" ", "")

        # Aadhaar LLM
        aadhaar_chain = aadhaar_prompt | llm | output_parser
        aadhaar_result_raw = await run_in_threadpool(aadhaar_chain.invoke, {"document_text": aadhaar_text})
        aadhaar_json_text = extract_json_block(aadhaar_result_raw)
        if not aadhaar_json_text:
            return JSONResponse(content={"error": "Aadhaar output does not contain valid JSON."}, status_code=400)
        
        aadhaar_data = json.loads(aadhaar_json_text)
        if not aadhaar_data.get("aadhar") or not re.fullmatch(r"\d{12}", aadhaar_data["aadhar"]):
            if re.fullmatch(r"\d{12}", aadhaar_number_extracted):
                aadhaar_data["aadhar"] = aadhaar_number_extracted
        aadhaar_data["aadhar"] = clean_aadhar_number(aadhaar_data["aadhar"])

        # PAN OCR
        pan_text = await run_in_threadpool(gemini_ocr_text_sync, pan_bytes)
        pan_chain = pan_prompt | llm | output_parser
        pan_result_raw = await run_in_threadpool(pan_chain.invoke, {"document_text": pan_text})
        pan_result_cleaned = pan_result_raw.strip("`").strip()
        pan_data = json.loads(pan_result_cleaned)

        # Name matching
        if aadhaar_data["name"].strip().lower() != pan_data["name"].strip().lower():
            return JSONResponse(content={
                "error": "Aadhaar and PAN appear to belong to different individuals.",
                "aadhaar_name": aadhaar_data["name"],
                "pan_name": pan_data["name"]
            }, status_code=400)

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
        return JSONResponse(content={"error": str(e)}, status_code=500)


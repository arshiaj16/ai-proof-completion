#aadhar
#import os
#from fastapi import FastAPI, File, UploadFile
#from fastapi.responses import JSONResponse
#from dotenv import load_dotenv
#from PIL import Image
#import pytesseract
#import io
#from sqlalchemy import create_engine, Column, Integer, String, Date, MetaData, Table
#from sqlalchemy.orm import sessionmaker
#import json
#
## LangChain imports
#from langchain_groq import ChatGroq
#from langchain_core.prompts import PromptTemplate
#from langchain_core.output_parsers import StrOutputParser
#
## Load .env
#load_dotenv()
#GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#DATABASE_URL = os.getenv("DATABASE_URL")
#
## Initialize FastAPI app
#app = FastAPI()
#
## Initialize ChatGroq LLM
#llm = ChatGroq(
#    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
#    groq_api_key=GROQ_API_KEY
#)
#
## Define Prompt Template
#prompt_template = PromptTemplate.from_template("""
#You are an intelligent document parser.
#
#Here is the OCR extracted text of an official identity document (like Aadhaar or PAN card):
#
#{document_text}
#
#Your task is to extract the following fields:
#- Full Name
#- Date of Birth (format: YYYY-MM-DD)
#- Aadhar Number (if present)
#- PAN Card (if present)
#
#Respond with ONLY the pure JSON object below — no explanation, no backticks.
#
#Format:
#{{
#    "name": "...",
#    "dob": "...",
#    "aadhar": "..."
#}}
#""")
#
## Output parser
#output_parser = StrOutputParser()
#
## Set up database
#engine = create_engine(DATABASE_URL)
#metadata = MetaData()
#
#documents_table = Table(
#    "documents",
#    metadata,
#    Column("id", Integer, primary_key=True),
#    Column("name", String),
#    Column("dob", Date),
#    Column("aadhar", String)
#)
#
#metadata.create_all(engine)
#
#SessionLocal = sessionmaker(bind=engine)
#
## FastAPI endpoint
#@app.post("/upload-image")
#async def upload_image(file: UploadFile = File(...)):
#    try:
#        # Read image
#        image_bytes = await file.read()
#        image = Image.open(io.BytesIO(image_bytes))
#
#        # OCR
#        extracted_text = pytesseract.image_to_string(image)
#
#        # Prompt formatting
#        formatted_prompt = prompt_template.format(document_text=extracted_text)
#        chain = prompt_template | llm | output_parser
#        model_reply = chain.invoke({"document_text": extracted_text})
#
#        # Parse JSON
#        try:
#            parsed_fields = json.loads(model_reply)
#        except Exception as e:
#            return JSONResponse(content={
#                "error": "Failed to parse model output",
#                "raw_output": model_reply
#            })
#
#        # Insert into DB
#        session = SessionLocal()
#        insert_stmt = documents_table.insert().values(
#            name=parsed_fields.get("name"),
#            dob=parsed_fields.get("dob"),
#            aadhar=parsed_fields.get("aadhar")
#        )
#        session.execute(insert_stmt)
#        session.commit()
#
#        # Return success
#        return JSONResponse(content={"message": "Data inserted successfully", "data": parsed_fields})
#
#    except Exception as e:
#        return JSONResponse(content={"error": str(e)}, status_code=500)
#


#pan
#import os
#from fastapi import FastAPI, File, UploadFile
#from fastapi.responses import JSONResponse
#from dotenv import load_dotenv
#from PIL import Image
#import pytesseract
#import io
#from sqlalchemy import create_engine, Column, Integer, String, Date, MetaData, Table
#from sqlalchemy.orm import sessionmaker
#import json
#
## LangChain imports
#from langchain_groq import ChatGroq
#from langchain_core.prompts import PromptTemplate
#from langchain_core.output_parsers import StrOutputParser
#
## Load .env
#load_dotenv()
#GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#DATABASE_URL = os.getenv("DATABASE_URL")
#
## Initialize FastAPI app
#app = FastAPI()
#
## Initialize ChatGroq LLM
#llm = ChatGroq(
#    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
#    groq_api_key=GROQ_API_KEY
#)
#
## Define Prompt Template (common for both Aadhaar and PAN)
#prompt_template = PromptTemplate.from_template("""
#You are an intelligent document parser.
#
#Here is the OCR extracted text of an official identity document (like Aadhaar or PAN card):
#
#{document_text}
#
#Your task is to extract the following fields:
#- Full Name
#- Date of Birth (format: YYYY-MM-DD)
#- Aadhar Number (if present)
#- PAN Card (if present)
#
#Respond with ONLY the pure JSON object below — no explanation, no backticks.
#
#Format:
#{{
#    "name": "...",
#    "dob": "...",
#    "aadhar": "...",
#    "pan": "..."
#}}
#""")
#
## Output parser
#output_parser = StrOutputParser()
#
## Set up database
#engine = create_engine(DATABASE_URL)
#metadata = MetaData()
#
#documents_table = Table(
#    "documents",
#    metadata,
#    Column("id", Integer, primary_key=True),
#    Column("name", String),
#    Column("dob", Date),
#    Column("aadhar", String),
#    Column("pan", String)
#)
#
#metadata.create_all(engine)
#
#SessionLocal = sessionmaker(bind=engine)
#
## Common function to process image and insert into DB
#def process_and_insert(image_bytes):
#    image = Image.open(io.BytesIO(image_bytes))
#    extracted_text = pytesseract.image_to_string(image)
#
#    formatted_prompt = prompt_template.format(document_text=extracted_text)
#    chain = prompt_template | llm | output_parser
#    model_reply = chain.invoke({"document_text": extracted_text})
#
#    try:
#        parsed_fields = json.loads(model_reply)
#    except Exception as e:
#        return {"error": "Failed to parse model output", "raw_output": model_reply}
#
#    # Clean fields → convert "" to None
#    name = parsed_fields.get("name") or None
#    dob = parsed_fields.get("dob") or None
#    aadhar = parsed_fields.get("aadhar") or None
#    pan = parsed_fields.get("pan") or None
#
#    # Insert into DB
#    with SessionLocal() as session:
#        insert_stmt = documents_table.insert().values(
#            name=name,
#            dob=dob,
#            aadhar=aadhar,
#            pan=pan
#        )
#        session.execute(insert_stmt)
#        session.commit()
#
#    return {"message": "Data inserted successfully", "data": parsed_fields}
#
## Aadhaar upload endpoint
#@app.post("/upload-aadhar")
#async def upload_aadhar(file: UploadFile = File(...)):
#    try:
#        image_bytes = await file.read()
#        result = process_and_insert(image_bytes)
#        return JSONResponse(content=result)
#    except Exception as e:
#        return JSONResponse(content={"error": str(e)}, status_code=500)
#
## PAN card upload endpoint
#@app.post("/upload-pan")
#async def upload_pan(file: UploadFile = File(...)):
#    try:
#        image_bytes = await file.read()
#        result = process_and_insert(image_bytes)
#        return JSONResponse(content=result)
#    except Exception as e:
#        return JSONResponse(content={"error": str(e)}, status_code=500)

#
#import os
#from fastapi import FastAPI, File, UploadFile, Body
#from fastapi.responses import JSONResponse
#from dotenv import load_dotenv
#from PIL import Image
#import pytesseract
#import io
#from sqlalchemy import create_engine, Column, Integer, String, Date, MetaData, Table
#from sqlalchemy.orm import sessionmaker
#import json
#
## LangChain imports
#from langchain_groq import ChatGroq
#from langchain_core.prompts import PromptTemplate
#from langchain_core.output_parsers import StrOutputParser
#
## Load .env
#load_dotenv()
#GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#DATABASE_URL = os.getenv("DATABASE_URL")
#
## Initialize FastAPI app
#app = FastAPI()
#
## Initialize ChatGroq LLM
#llm = ChatGroq(
#    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
#    groq_api_key=GROQ_API_KEY
#)
#
## Define Prompt Template
#prompt_template = PromptTemplate.from_template("""
#You are an intelligent document parser.
#
#Here is the OCR extracted text of an official identity document (like Aadhaar or PAN card):
#
#{document_text}
#
#Your task is to extract the following fields:
#- Full Name
#- Date of Birth (format: YYYY-MM-DD)
#- Aadhar Number (if present)
#- PAN Card (if present)
#
#Respond with ONLY the pure JSON object below — no explanation, no backticks.
#
#Format:
#{{
#    "name": "...",
#    "dob": "...",
#    "aadhar": "...",
#    "pan": "..."
#}}
#""")
#
## Output parser
#output_parser = StrOutputParser()
#
## Set up database
#engine = create_engine(DATABASE_URL)
#metadata = MetaData()
#
#documents_table = Table(
#    "documents",
#    metadata,
#    Column("id", Integer, primary_key=True),
#    Column("name", String),
#    Column("dob", Date),
#    Column("aadhar", String),
#    Column("pan", String)
#)
#
#metadata.create_all(engine)
#
#SessionLocal = sessionmaker(bind=engine)
#
## Common function to process image and insert into DB
#def process_and_insert(image_bytes):
#    image = Image.open(io.BytesIO(image_bytes))
#    extracted_text = pytesseract.image_to_string(image)
#
#    formatted_prompt = prompt_template.format(document_text=extracted_text)
#    chain = prompt_template | llm | output_parser
#    model_reply = chain.invoke({"document_text": extracted_text})
#
#    try:
#        parsed_fields = json.loads(model_reply)
#    except Exception as e:
#        return {"error": "Failed to parse model output", "raw_output": model_reply}
#
#    # Clean fields → convert "" to None
#    name = parsed_fields.get("name") or None
#    dob = parsed_fields.get("dob") or None
#    aadhar = parsed_fields.get("aadhar") or None
#    pan = parsed_fields.get("pan") or None
#
#    # Insert into DB
#    with SessionLocal() as session:
#        insert_stmt = documents_table.insert().values(
#            name=name,
#            dob=dob,
#            aadhar=aadhar,
#            pan=pan
#        )
#        session.execute(insert_stmt)
#        session.commit()
#
#    return {"message": "Data inserted successfully", "data": parsed_fields}
#
## Aadhaar upload endpoint
#@app.post("/upload-aadhar")
#async def upload_aadhar(file: UploadFile = File(...)):
#    try:
#        image_bytes = await file.read()
#        result = process_and_insert(image_bytes)
#        return JSONResponse(content=result)
#    except Exception as e:
#        return JSONResponse(content={"error": str(e)}, status_code=500)
#
## PAN card upload endpoint
#@app.post("/upload-pan")
#async def upload_pan(file: UploadFile = File(...)):
#    try:
#        image_bytes = await file.read()
#        result = process_and_insert(image_bytes)
#        return JSONResponse(content=result)
#    except Exception as e:
#        return JSONResponse(content={"error": str(e)}, status_code=500)
#
## Manual insert endpoint
#@app.post("/create-manual")
#async def create_manual(
#    name: str = Body(...),
#    dob: str = Body(...),
#    aadhar: str = Body(...),
#    pan: str = Body(...)
#):
#    try:
#        # Clean fields → convert "" to None
#        name = name or None
#        dob = dob or None
#        aadhar = aadhar or None
#        pan = pan or None
#
#        # Insert into DB
#        with SessionLocal() as session:
#            insert_stmt = documents_table.insert().values(
#                name=name,
#                dob=dob,
#                aadhar=aadhar,
#                pan=pan
#            )
#            session.execute(insert_stmt)
#            session.commit()
#
#        return JSONResponse(content={"message": "Data inserted successfully", "data": {
#            "name": name,
#            "dob": dob,
#            "aadhar": aadhar,
#            "pan": pan
#        }})
#
#    except Exception as e:
#        return JSONResponse(content={"error": str(e)}, status_code=500)


#import os
#from fastapi import FastAPI, File, UploadFile, Body
#from fastapi.responses import JSONResponse
#from dotenv import load_dotenv
#from PIL import Image
#import pytesseract
#import io
#from sqlalchemy import create_engine, Column, Integer, String, Date, MetaData, Table
#from sqlalchemy.orm import sessionmaker
#import json
#
## LangChain imports
#from langchain_groq import ChatGroq
#from langchain_core.prompts import PromptTemplate
#from langchain_core.output_parsers import StrOutputParser
#
## Load .env
#load_dotenv()
#GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#DATABASE_URL = os.getenv("DATABASE_URL")
#
## Initialize FastAPI app
#app = FastAPI()
#
## Initialize ChatGroq LLM
#llm = ChatGroq(
#    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
#    groq_api_key=GROQ_API_KEY
#)
#
## Define Prompt Template
#prompt_template = PromptTemplate.from_template("""
#You are an intelligent document parser.
#
#Here is the OCR extracted text of an official identity document (like Aadhaar or PAN card):
#
#{document_text}
#
#Your task is to extract the following fields:
#- Full Name
#- Date of Birth (format: YYYY-MM-DD)
#- Aadhar Number (if present)
#- PAN Card (if present)
#
#Respond with ONLY the pure JSON object below — no explanation, no backticks.
#
#Format:
#{{
#    "name": "...",
#    "dob": "...",
#    "aadhar": "...",
#    "pan": "..."
#}}
#""")
#
## Output parser
#output_parser = StrOutputParser()
#
## Set up database
#engine = create_engine(DATABASE_URL)
#metadata = MetaData()
#
#documents_table = Table(
#    "documents",
#    metadata,
#    Column("id", Integer, primary_key=True),
#    Column("name", String),
#    Column("dob", Date),
#    Column("aadhar", String),
#    Column("pan", String)
#)
#
#metadata.create_all(engine)
#
#SessionLocal = sessionmaker(bind=engine)
#
## Common function to process image and insert into DB
#def process_and_insert(image_bytes):
#    image = Image.open(io.BytesIO(image_bytes))
#    extracted_text = pytesseract.image_to_string(image)
#
#    formatted_prompt = prompt_template.format(document_text=extracted_text)
#    chain = prompt_template | llm | output_parser
#    model_reply = chain.invoke({"document_text": extracted_text})
#
#    try:
#        parsed_fields = json.loads(model_reply)
#    except Exception as e:
#        return {"error": "Failed to parse model output", "raw_output": model_reply}
#
#    # Clean fields → convert "" to None
#    name = parsed_fields.get("name") or None
#    dob = parsed_fields.get("dob") or None
#    aadhar = parsed_fields.get("aadhar") or None
#    pan = parsed_fields.get("pan") or None
#
#    # Insert into DB
#    with SessionLocal() as session:
#        insert_stmt = documents_table.insert().values(
#            name=name,
#            dob=dob,
#            aadhar=aadhar,
#            pan=pan
#        )
#        session.execute(insert_stmt)
#        session.commit()
#
#    return {"message": "Data inserted successfully", "data": parsed_fields}
#
## Aadhaar upload endpoint
#@app.post("/upload-aadhar")
#async def upload_aadhar(file: UploadFile = File(...)):
#    try:
#        image_bytes = await file.read()
#        result = process_and_insert(image_bytes)
#        return JSONResponse(content=result)
#    except Exception as e:
#        return JSONResponse(content={"error": str(e)}, status_code=500)
#
## PAN card upload endpoint
#@app.post("/upload-pan")
#async def upload_pan(file: UploadFile = File(...)):
#    try:
#        image_bytes = await file.read()
#        result = process_and_insert(image_bytes)
#        return JSONResponse(content=result)
#    except Exception as e:
#        return JSONResponse(content={"error": str(e)}, status_code=500)
#
## Manual insert endpoint with consistency check
#@app.post("/create-manual")
#async def create_manual(
#    name: str = Body(...),
#    dob: str = Body(...),
#    aadhar: str = Body(...),
#    pan: str = Body(...)
#):
#    try:
#        # Clean fields → convert "" to None
#        name = name or None
#        dob = dob or None
#        aadhar = aadhar or None
#        pan = pan or None
#
#        with SessionLocal() as session:
#            # Check if Aadhaar already exists
#            result = session.execute(
#                documents_table.select().where(documents_table.c.aadhar == aadhar)
#            ).fetchone()
#
#            if result:
#                # Aadhaar exists → check if name and dob match
#                existing_name = result[documents_table.c.name]
#                existing_dob = result[documents_table.c.dob]
#
#                if existing_name != name or str(existing_dob) != dob:
#                    # Mismatch → reject insert
#                    return JSONResponse(
#                        content={
#                            "error": "Name or DOB mismatch with existing Aadhaar entry. Please check your details."
#                        },
#                        status_code=400
#                    )
#
#                # Name & DOB match → allow updating PAN if needed
#                update_stmt = (
#                    documents_table.update()
#                    .where(documents_table.c.aadhar == aadhar)
#                    .values(pan=pan)
#                )
#                session.execute(update_stmt)
#                session.commit()
#
#                return JSONResponse(
#                    content={
#                        "message": "PAN updated successfully for existing Aadhaar entry.",
#                        "data": {
#                            "name": name,
#                            "dob": dob,
#                            "aadhar": aadhar,
#                            "pan": pan
#                        }
#                    }
#                )
#            else:
#                # No existing Aadhaar → allow new insert
#                insert_stmt = documents_table.insert().values(
#                    name=name,
#                    dob=dob,
#                    aadhar=aadhar,
#                    pan=pan
#                )
#                session.execute(insert_stmt)
#                session.commit()
#
#                return JSONResponse(
#                    content={
#                        "message": "New record inserted successfully.",
#                        "data": {
#                            "name": name,
#                            "dob": dob,
#                            "aadhar": aadhar,
#                            "pan": pan
#                        }
#                    }
#                )
#
#    except Exception as e:
#        return JSONResponse(content={"error": str(e)}, status_code=500)



#import os
#import io
#import json
#import re
#from fastapi import FastAPI, File, UploadFile
#from fastapi.responses import JSONResponse
#from dotenv import load_dotenv
#from PIL import Image
#import pytesseract
#from sqlalchemy import create_engine, Column, Integer, String, Date, MetaData, Table
#from sqlalchemy.orm import sessionmaker
# 
## LangChain imports
#from langchain_groq import ChatGroq
#from langchain_core.prompts import PromptTemplate
#from langchain_core.output_parsers import StrOutputParser
# 
## Load environment variables
#load_dotenv()
#GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#DATABASE_URL = os.getenv("DATABASE_URL")
# 
## Initialize FastAPI app
#app = FastAPI()
# 
## Initialize Groq LLM
#llm = ChatGroq(
#    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
#    groq_api_key=GROQ_API_KEY
#)
#output_parser = StrOutputParser()
# 
## Prompt template for Aadhaar
#aadhaar_prompt = PromptTemplate.from_template("""
#You are an intelligent document parser.
# 
#You will be given raw OCR text from an Aadhaar card.
# 
#The text may contain the following fields:
#- Full Name (look for patterns like: "Name: John Doe", or standalone names at the top)
#- Date of Birth (formats: YYYY-MM-DD, YYYY/MM/DD, or DD-MM-YYYY)
#- Aadhaar Number — a **12-digit number**, sometimes written with spaces (e.g., 1234 5678 9012) and **does not have any label**. It usually appears at the bottom or near the bottom of the text.
# 
#Here is the OCR extracted text:
# 
#{document_text}
# 
#Extract the following fields:
#- Full Name
#- Date of Birth (convert to format YYYY-MM-DD)
#- Aadhaar Number (strip any spaces; must be exactly 12 digits)
# 
#Respond ONLY with valid JSON and nothing else.
#Do not add explanations, markdown, or comments.
# 
#Return in the following strict format:
#{{
#  "name": "...",
#  "dob": "...",
#  "aadhar": "..."
#}}
#""")
# 
# 
## Prompt template for PAN (PAN number only)
#pan_prompt = PromptTemplate.from_template("""
#You are an intelligent document parser.
# 
#Here is the OCR extracted text from an Indian PAN card:
# 
#{document_text}
# 
#Extract only the PAN Number (ignore everything else). The PAN number is a 10-character alphanumeric code (5 uppercase letters, 4 digits, 1 uppercase letter).
# 
#Respond only with JSON:
#{{
#    "pan": "..."
#}}
#""")
# 
## Database setup
#engine = create_engine(DATABASE_URL)
#metadata = MetaData()
#documents_table = Table(
#    "documents",
#    metadata,
#    Column("id", Integer, primary_key=True),
#    Column("name", String),
#    Column("dob", String),
#    Column("aadhar", String),
#    Column("pan", String),
#)
#metadata.create_all(engine)
#SessionLocal = sessionmaker(bind=engine)
# 
## Clean Aadhaar spaces
#def clean_aadhar_number(aadhar: str) -> str:
#    return aadhar.replace(" ", "").strip()
# 
#@app.post("/upload-documents/")
#async def upload_documents(aadhaar_file: UploadFile = File(...), pan_file: UploadFile = File(...)):
#    try:
#        # === Aadhaar OCR and LLM Parsing ===
#        aadhaar_bytes = await aadhaar_file.read()
#        aadhaar_image = Image.open(io.BytesIO(aadhaar_bytes))
#        aadhaar_text = pytesseract.image_to_string(aadhaar_image)
# 
#       
# 
#        # Use regex to find a valid 12-digit Aadhaar number (with or without spaces)
#        aadhaar_number_match = re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', aadhaar_text)
#        aadhaar_number_extracted = aadhaar_number_match.group(0).replace(" ", "") if aadhaar_number_match else ""
# 
#        aadhaar_chain = aadhaar_prompt | llm | output_parser
#        aadhaar_result_raw = aadhaar_chain.invoke({"document_text": aadhaar_text})
#        aadhaar_result_cleaned = aadhaar_result_raw.strip().strip("`").strip()
#        try:
#            aadhaar_data = json.loads(aadhaar_result_cleaned)
#            # Assign Aadhaar number: use LLM if present, else fallback to OCR+regex
#            if not aadhaar_data.get("aadhar") or not re.fullmatch(r"\d{12}", aadhaar_data["aadhar"]):
#                if re.fullmatch(r"\d{12}", aadhaar_number_extracted):
#                    aadhaar_data["aadhar"] = aadhaar_number_extracted
#            aadhaar_data["aadhar"] = clean_aadhar_number(aadhaar_data["aadhar"])
#        except Exception:
#            return JSONResponse(content={
#                "error": "Failed to parse Aadhaar output",
#                "raw": aadhaar_result_raw
#            }, status_code=400)
# 
#        # === PAN OCR and LLM Parsing (PAN number only) ===
#        pan_bytes = await pan_file.read()
#        pan_image = Image.open(io.BytesIO(pan_bytes))
#        pan_text = pytesseract.image_to_string(pan_image)
# 
#        pan_chain = pan_prompt | llm | output_parser
#        pan_result_raw = pan_chain.invoke({"document_text": pan_text})
#        pan_result_cleaned = pan_result_raw.strip().strip("`").strip()
#        try:
#            pan_data = json.loads(pan_result_cleaned)
#        except Exception:
#            session = SessionLocal()
#            try:
#                insert_stmt = documents_table.insert().values(
#                    name=aadhaar_data["name"].strip(),
#                    dob=aadhaar_data["dob"],
#                    aadhar=aadhaar_data["aadhar"],
#                    pan=pan_data["pan"].strip()
#                )
#                session.execute(insert_stmt)
#                session.commit()
#            finally:
#                session.close()
# 
#        return JSONResponse(content={
#            "message": "Document data inserted successfully",
#            "data": {
#                "name": aadhaar_data["name"],
#                "dob": aadhaar_data["dob"],
#                "aadhar": aadhaar_data["aadhar"],
#                "pan": pan_data["pan"]
#            }
#        })
# 
#    except Exception as e:
#        return JSONResponse(content={"error": str(e)}, status_code=500)
 
 

import os
import json
import re
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from PIL import Image
from sqlalchemy import create_engine, Column, Integer, String, MetaData, Table
from sqlalchemy.orm import sessionmaker
 
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
DATABASE_URL = os.getenv("DATABASE_URL")
 
 
# Initialize FastAPI app
app = FastAPI()
 
# Enable CORS
from fastapi.middleware.cors import CORSMiddleware
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific URLs in production
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
- Full Name (look for patterns like: "Name: John Doe", or standalone names at the top)
- Date of Birth (formats: YYYY-MM-DD, YYYY/MM/DD, or DD-MM-YYYY)
- Aadhaar Number — a **12-digit number**, sometimes written with spaces (e.g., 1234 5678 9012) and **does not have any label**. It usually appears at the bottom or near the bottom of the text.
- Gender (Male, Female, Transgender; may appear as M, F, T)
                                             
Here is the OCR extracted text:
 
{document_text}
 
Extract the following fields:
- Full Name
- Date of Birth (convert to format YYYY-MM-DD)
- Aadhaar Number (strip any spaces; must be exactly 12 digits)
- Gender (Normalize as "Male", "Female", or "Transgender")
                                             
Respond ONLY with valid JSON and nothing else.
Do not add explanations, markdown, or comments.
 
Return in the following strict format:
{{
  "name": "...",
  "dob": "...",
  "gender": "...",
  "aadhar": "..."
}}
""")
 
 
# Prompt template for PAN (PAN number only)
pan_prompt = PromptTemplate.from_template("""
You are an intelligent document parser.
 
Here is the OCR extracted text from an Indian PAN card:
 
{document_text}
 
Extract the following fields:
- Name: The full name of the PAN holder (usually after the PAN number)
- Father's Name: Usually listed below the name.
- PAN Number: A 10-character alphanumeric code (5 uppercase letters, 4 digits, 1 uppercase letter)
 
 
Respond only with valid JSON:
{{
  "name": "...",
  "father_name": "...",
  "pan": "..."
}}
""")
 
 
# Database setup
engine = create_engine(DATABASE_URL)
metadata = MetaData()
documents_table = Table(
    "documents",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String),
    Column("dob", String),
    Column("gender", String),
    Column("father_name", String),
    Column("aadhar", String),
    Column("pan", String),
)
metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)
 
 
# Helper
def clean_aadhar_number(aadhar: str) -> str:
    return aadhar.replace(" ", "").strip()
 
 
# Perform OCR via Gemini
def gemini_ocr_text(image_bytes):
    image_part = {"mime_type": "image/jpeg", "data": image_bytes}
    response = vision_model.generate_content([image_part, "Extract all text."])
    print("Gemini OCR Response:", response.text)
    return response.text
 
 
#Endpoint of the FastAPI app to handle document uploads and storing them in the database
@app.post("/upload-documents/")
async def upload_documents(aadhaar_file: UploadFile = File(...), pan_file: UploadFile = File(...)):
    try:
        # Aadhaar OCR using Gemini
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
 
        # LLM Parsing
        aadhaar_chain = aadhaar_prompt | llm | output_parser
        aadhaar_result_raw = aadhaar_chain.invoke({"document_text": aadhaar_text})
        aadhaar_result_cleaned = aadhaar_result_raw.strip().strip("`").strip()
       
        try:
            aadhaar_data = json.loads(aadhaar_result_cleaned)
           
            # Fallback if LLM missed Aadhaar number
            if not aadhaar_data.get("aadhar") or not re.fullmatch(r"\d{12}", aadhaar_data["aadhar"]):
                if re.fullmatch(r"\d{12}", aadhaar_number_extracted):
                    aadhaar_data["aadhar"] = aadhaar_number_extracted
           
            aadhaar_data["aadhar"] = clean_aadhar_number(aadhaar_data["aadhar"])
       
        except Exception:
            return JSONResponse(content={"error": "Failed to parse Aadhaar output", "raw": aadhaar_result_raw}, status_code=400)
 
        # PAN OCR using Gemini
        pan_bytes = await pan_file.read()
        pan_text = gemini_ocr_text(pan_bytes)
 
        pan_chain = pan_prompt | llm | output_parser
        pan_result_raw = pan_chain.invoke({"document_text": pan_text})
        pan_result_cleaned = pan_result_raw.strip().strip("`").strip()
        pan_data = json.loads(pan_result_cleaned)
 
        # Normalize and compare names
        aadhaar_name = aadhaar_data["name"].strip().lower()
        pan_name = pan_data["name"].strip().lower()  # Ensure 'name' is extracted in pan_data via prompt
 
        if aadhaar_name != pan_name:
            return JSONResponse(content={
                "error": "Aadhaar and PAN appear to belong to different individuals.",
                "aadhaar_name": aadhaar_data["name"],
                "pan_name": pan_data["name"]
            }, status_code=400)
 
        # Save to DB
        session = SessionLocal()
        try:
            insert_stmt = documents_table.insert().values(
                name=aadhaar_data["name"].strip(),
                dob=aadhaar_data["dob"],
                gender=aadhaar_data["gender"],
                father_name=pan_data["father_name"].strip(),
                aadhar=aadhaar_data["aadhar"],
                pan=pan_data["pan"].strip()
            )
            session.execute(insert_stmt)
            session.commit()
        finally:
            session.close()
 
        return JSONResponse(content={
            "message": "Document data inserted successfully",
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
 
 
# Endpoint to validate user entered data against the database
from fastapi import Body
 
@app.post("/validate-documents/")
async def validate_documents(data: dict = Body(...)):
    # Extract fields
    user_name = data.get("name", "").strip()
    dob = data.get("dob", "").strip()
    gender = data.get("gender", "").strip()
    father_name = data.get("father_name", "").strip()
    aadhar = data.get("aadhar", "").strip()
    pan = data.get("pan", "").strip()
 
    if not aadhar and not pan:
        return JSONResponse(content={"error": "Aadhaar or PAN number must be provided."}, status_code=400)
 
    session = SessionLocal()
    try:
        # Search by Aadhaar or PAN
        query = documents_table.select().where(
            (documents_table.c.aadhar == aadhar) | (documents_table.c.pan == pan)
        )
        result = session.execute(query).fetchone()
 
        if not result:
            return JSONResponse(content={"error": "No matching record found in the database."}, status_code=404)
 
        # Convert result row to dict
        record = dict(result._mapping)
 
        # Compare fields
        mismatches = {}
        if user_name.strip().lower() != record["name"].strip().lower():
            mismatches["name"] = {"entered": user_name, "expected": record["name"]}
        if dob != record["dob"]:
            mismatches["dob"] = {"entered": dob, "expected": record["dob"]}
        if gender.strip().lower() != record["gender"].strip().lower():
            mismatches["gender"] = {"entered": gender, "expected": record["gender"]}
        if father_name.strip().lower() != record["father_name"].strip().lower():
            mismatches["father_name"] = {"entered": father_name, "expected": record["father_name"]}
        if aadhar and aadhar != record["aadhar"]:
            mismatches["aadhar"] = {"entered": aadhar, "expected": record["aadhar"]}
        if pan and pan != record["pan"]:
            mismatches["pan"] = {"entered": pan, "expected": record["pan"]}
 
        if mismatches:
            return JSONResponse(content={
                "status": "validation_failed",
                "mismatched_fields": mismatches
            }, status_code=400)
 
        return JSONResponse(content={
            "status": "success",
            "message": "All fields validated successfully."
        })
 
    finally:
        session.close()
 

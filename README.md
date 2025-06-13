# AI proof completion 
This project is a FastAPI-based backend application that extracts structured information from uploaded Aadhaar and PAN card images using:
1.Google Gemini Vision API for OCR (image → text)
2.Groq-hosted LLaMA-4 LLM via LangChain for intelligent parsing of noisy OCR text
Features:
1.Upload Aadhaar & PAN card images via /upload-documents/ endpoint
2.Extract:
Name
Date of Birth (DOB)
Aadhaar number
Gender
Father's name
PAN number
3.Intelligent post-processing with LLM to clean and validate extracted data
4.Log structured data in /log-event/ endpoint
5.Clean and modular FastAPI architecture with reusable components

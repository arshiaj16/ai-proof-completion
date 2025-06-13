from langchain_core.prompts import PromptTemplate

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


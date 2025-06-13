import re

def clean_aadhar_number(aadhar: str) -> str:
    return aadhar.replace(" ", "").strip()

def extract_json_block(text):
    try:
        match = re.search(r"{.*}", text, re.DOTALL)
        return match.group(0) if match else None
    except Exception as e:
        raise ValueError("Failed to extract JSON block from text")


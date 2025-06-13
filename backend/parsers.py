from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from config import GROQ_API_KEY

llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    groq_api_key=GROQ_API_KEY
)

output_parser = StrOutputParser()


from dotenv import load_dotenv
import os

load_dotenv("C:/Users/eliza/CLINICO-Medical-Chatbot/.env", override=True)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

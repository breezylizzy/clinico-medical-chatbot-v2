from dotenv import load_dotenv
import os

load_dotenv("C:/Users/eliza/mediskin-chatbot/.env", override=True)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

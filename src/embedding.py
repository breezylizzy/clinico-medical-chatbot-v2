from langchain_huggingface import HuggingFaceEmbeddings 
import torch

def get_embeddings():
    model_name = "BAAI/bge-small-en-v1.5"   
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    return embeddings
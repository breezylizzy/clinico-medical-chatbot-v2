from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.config import PINECONE_API_KEY
from src.loader import load_pdf_files, filter_to_minimal_docs, text_split
from src.embedding import get_embeddings

import time

index_name = "clinico"

def vectorstore():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if not pc.has_index(index_name):
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        
        print("Waiting for index to be ready...")
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print("Index is ready!")
        time.sleep(5)
        
        print("Loading PDF files...")
        extracted_data = load_pdf_files("C:/Users/eliza/CLINICO-Medical-Chatbot/data")
        print(f"Loaded {len(extracted_data)} documents")
        
        print("Filtering documents...")
        filter_data = filter_to_minimal_docs(extracted_data)
        
        print("Splitting text into chunks...")
        text_chunks = text_split(filter_data)
        print(f"Created {len(text_chunks)} text chunks")
        
        print("Getting embeddings...")
        embedding = get_embeddings()

        print("Uploading to Pinecone...")
        PineconeVectorStore.from_documents(
            documents=text_chunks,
            index_name=index_name,
            embedding=embedding
        )
        print("Upload complete!")

    embedding = get_embeddings()

    return PineconeVectorStore(
        index_name=index_name,
        embedding=embedding
    )
from sentence_transformers import CrossEncoder
from langchain_core.runnables import RunnableLambda
from langchain.schema import Document
from typing import List
import torch

from src.retriever import get_base_retriever

reranker = CrossEncoder(
    "BAAI/bge-reranker-base",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

def rerank_docs(query: str, docs: List[Document], top_k: int = 5):
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for score, doc in ranked[:top_k]]

def get_rerank_retriever(top_k_retriever=20, top_k_rerank=5):
    base_retriever = get_base_retriever(k=top_k_retriever)

    def rerank_retriever_fn(input_data):
        query = input_data["input"] if isinstance(input_data, dict) else str(input_data)

        docs = base_retriever.invoke(query)
        return rerank_docs(query, docs, top_k=top_k_rerank)

    return RunnableLambda(rerank_retriever_fn)

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

def _history_to_text(history, max_msgs=6) -> str:
    """history biasanya list of messages. Ambil beberapa terakhir biar query tidak kepanjangan."""
    if not history:
        return ""
    tail = history[-max_msgs:]
    lines = []
    for m in tail:
        role = getattr(m, "type", None) or m.__class__.__name__
        content = getattr(m, "content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

def get_rerank_retriever(top_k_retriever=20, top_k_rerank=5):
    base_retriever = get_base_retriever(k=top_k_retriever)

    def rerank_retriever_fn(input_data):
        if isinstance(input_data, dict):
            user_q = input_data.get("input", "")
            hist = input_data.get("history")  # <-- ini yang tadinya diabaikan
        else:
            user_q = str(input_data)
            hist = None

        hist_text = _history_to_text(hist, max_msgs=6)
        if hist_text:
            query = (
                "Use the conversation context to interpret the user's follow-up question.\n"
                f"CONTEXT:\n{hist_text}\n\n"
                f"QUESTION:\n{user_q}"
            )
        else:
            query = user_q

        docs = base_retriever.invoke(query)
        return rerank_docs(query, docs, top_k=top_k_rerank)

    return RunnableLambda(rerank_retriever_fn)
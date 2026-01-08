from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.prompt import prompt
from src.memory import get_session_history
from src.reranker import get_rerank_retriever


def create_rag_chain(chat_model):
    qa_chain = create_stuff_documents_chain(
        llm=chat_model,
        prompt=prompt
    )
    
    retriever = get_rerank_retriever()

    rag_chain = create_retrieval_chain(
        retriever,
        qa_chain
    )

    rag_with_memory = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer",  
    )
    return rag_with_memory

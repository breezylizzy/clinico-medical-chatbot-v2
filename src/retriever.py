from src.vectorstore import vectorstore

def get_base_retriever(k: int = 20):
    docsearch = vectorstore()

    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    return retriever

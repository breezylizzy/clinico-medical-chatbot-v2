from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document

def load_pdf_files(data):
    loader = DirectoryLoader(
        data, 
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []   
    for doc in docs:
        src = doc.metadata.get("source", "")
        filename = src.split("/")[-1] if src else ""
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={'source': filename}
            )
        )
    return minimal_docs

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

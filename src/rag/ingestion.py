# importing necessary libraries
import os
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import DirectoryLoader
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
#from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers import EnsembleRetriever
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import CrossEncoder
import logging
from docling.document_converter import DocumentConverter
from pathlib import Path
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
'''from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredImageLoader,
)'''

# loading environment variables
load_dotenv()
data_path=os.getenv("data_path")
vector_db=os.getenv("vector_db")
'''
loader_map={
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".docx": Docx2txtLoader,
    ".html": UnstructuredHTMLLoader,
    ".md": UnstructuredMarkdownLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".jpg": UnstructuredImageLoader,
    ".png": UnstructuredImageLoader,
    ".jpeg": UnstructuredImageLoader,
}'''
def load_documents(folder_path):
    documents=[]
    converter=DocumentConverter()
    for file_path in Path(folder_path).iterdir():
        if file_path.is_file():
            result=converter.convert(str(file_path))
            result=result.document.export_to_markdown()
            
            documents.append(
                Document(
                    page_content=result,
                    metadata={
                        "source":str(file_path),
                        "file_type": file_path.suffix,
                    }
                )            
                             )
    return documents
def semantic_chunking(docs):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    splitter = SemanticChunker(embeddings)

    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    chunks = splitter.create_documents(
        texts=texts,
        metadatas=metadatas
    )

    return chunks
def recursive_chunking(docs):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks=text_splitter.split_documents(docs)
    return chunks
def storing_db(chunks):
    embeddings=OllamaEmbeddings(model="nomic-embed-text")
    vectordb=Chroma.from_documents(embedding=embeddings,persist_directory=vector_db,documents=chunks)
    return vectordb 


# load and store db 
if __name__=="__main__":
    docs=load_documents(data_path)
    chunks=recursive_chunking(docs) 
    vectordb=storing_db(chunks)
    print("Database created successfully")
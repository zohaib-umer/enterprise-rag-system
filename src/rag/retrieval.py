import os
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, MultiQueryRetriever as LangChainMultiQueryRetriever
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from sentence_transformers import CrossEncoder

load_dotenv()
data_path = os.getenv("data_path", "./data")
vector_db = os.getenv("vector_db", "./db")

def load_vectordb():
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma(persist_directory=vector_db, embedding_function=embedding)
    return vectordb

def vector_retrieval(docs, query):
    vectordb = load_vectordb()
    vector_retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    return vector_retriever.invoke(query)

def Bm25_retrievel(docs, query):
    bm25 = BM25Retriever.from_documents(docs)
    return bm25.invoke(query)

def hybrid_retrievel(docs, query):
    vectordb = load_vectordb()
    vector_retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    bm25_retriever = BM25Retriever.from_documents(docs)
    
    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble.invoke(query)

def Cross_encoder(query, docs):
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[query, doc.page_content] for doc in docs]
    scores = cross_encoder.predict(pairs, batch_size=8, show_progress_bar=False)
    ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:5]
    return ranked_docs

def get_response(query, docs):
    llm = ChatOllama(model="llama3.2")
    prompt = ChatPromptTemplate.from_template(
        """
        You are an Enterprise AI Assistant.
        Your job is to answer questions ONLY using the provided context.
        
        Rules:
        1. Use ONLY the information provided in the context.
        2. Do NOT make up facts or use outside knowledge.
        3. If the answer is not present in the context, respond:
           "I couldn't find this information."
        4. Keep the answer clear, professional, and concise.
        
        ------------------------
        Context:
        {context}
        ------------------------
        Question:
        {question}
        Answer:
        """
    )
    context_str = "\n\n".join([doc.page_content for doc in docs])
    formatted_prompt = prompt.format(context=context_str, question=query)
    response = llm.invoke(formatted_prompt)
    return response.content
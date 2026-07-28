import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from src.api.schemas import (
    QueryRequest, 
    QueryResponse, 
    IngestRequest, 
    IngestResponse, 
    SourceDocument
)
from src.rag.ingestion import load_documents, recursive_chunking, semantic_chunking, storing_db
from src.rag.retrieval import (
    load_vectordb,
    vector_retrieval,
    Bm25_retrievel,
    hybrid_retrievel,
    Cross_encoder,
    get_response
)

load_dotenv()
data_path = os.getenv("data_path", "./data")

app = FastAPI(
    title="Enterprise RAG System API",
    description="Configurable RAG pipeline with dynamic chunking and retrieval controls.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def run_ingestion(params: IngestRequest):
    try:
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail=f"Data directory '{data_path}' not found.")
        
        raw_docs = load_documents(data_path)
        if not raw_docs:
            return IngestResponse(
                status="warning",
                message="No valid documents found in data directory.",
                total_documents=0,
                chunks_created=0
            )

        # Dynamic Chunking Selection
        if params.chunking_strategy == "semantic":
            chunks = semantic_chunking(raw_docs)
        else:
            chunks = recursive_chunking(raw_docs)
            
        storing_db(chunks)
        
        return IngestResponse(
            status="success",
            message=f"Successfully indexed using [{params.chunking_strategy.upper()}] chunking.",
            total_documents=len(raw_docs),
            chunks_created=len(chunks)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/query", response_model=QueryResponse, tags=["Retrieval"])
async def answer_question(request: QueryRequest):
    try:
        query_text = request.query.strip()
        if not query_text:
            raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
        vectordb = load_vectordb()
        stored_data = vectordb.get()
        
        from langchain_core.documents import Document
        all_docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(stored_data["documents"], stored_data["metadatas"])
        ]
        
        if not all_docs:
            raise HTTPException(status_code=400, detail="Database is empty. Please run ingestion first.")

        # Dynamic Retrieval Selection
        strategy = request.retrieval_strategy.lower()
        
        if strategy == "vector":
            retrieved_docs = vector_retrieval(all_docs, query_text)
        elif strategy == "bm25":
            retrieved_docs = Bm25_retrievel(all_docs, query_text)
        elif strategy == "cross_encoder":
            initial_docs = hybrid_retrievel(all_docs, query_text)
            ranked = Cross_encoder(query_text, initial_docs)
            retrieved_docs = [doc for doc, score in ranked]
        else: # Default: Hybrid Search
            retrieved_docs = hybrid_retrievel(all_docs, query_text)
        
        # Limit to requested top_k
        top_docs = retrieved_docs[:request.top_k]
        
        # Generate Answer
        llm_answer = get_response(query_text, top_docs)
        
        sources_list = [
            SourceDocument(
                source=os.path.basename(doc.metadata.get("source", "Unknown")),
                content_snippet=doc.page_content[:200] + "..."
            )
            for doc in top_docs
        ]
        
        return QueryResponse(
            question=query_text,
            answer=llm_answer,
            strategy_used=strategy.upper(),
            sources=sources_list
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/health", tags=["System"])
async def health():
    return {"status": "healthy"}
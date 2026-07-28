from pydantic import BaseModel, Field
from typing import List, Optional

# Request parameters for Ingestion
class IngestRequest(BaseModel):
    chunking_strategy: str = Field("recursive", description="Option: 'recursive' or 'semantic'")
    chunk_size: int = Field(1000, example=1000)
    chunk_overlap: int = Field(200, example=200)

# Request parameters for Querying
class QueryRequest(BaseModel):
    query: str = Field(..., example="What are the Q3 financial results?")
    retrieval_strategy: str = Field("hybrid", description="Option: 'vector', 'bm25', 'hybrid', 'cross_encoder'")
    top_k: int = Field(5, example=5)

# Source document metadata schema
class SourceDocument(BaseModel):
    source: str
    content_snippet: str
    relevance_score: Optional[float] = None

# Query response schema
class QueryResponse(BaseModel):
    question: str
    answer: str
    strategy_used: str
    sources: Optional[List[SourceDocument]] = []

# Ingest response schema
class IngestResponse(BaseModel):
    status: str
    message: str
    total_documents: int
    chunks_created: int
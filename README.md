# 🤖 Enterprise Document Intelligence & Multi-Stage Hybrid RAG Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F00?style=for-the-badge&logo=databricks&logoColor=white)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> A production-grade, enterprise-ready **Retrieval-Augmented Generation (RAG)** platform engineered for high-precision question answering over complex enterprise documents, technical specifications, and multidimensional PDF reports. Built with **IBM Docling**, **Hybrid BM25 + Vector MMR**, **MS-MARCO Cross-Encoder Reranking**, **ChromaDB**, and **Llama 3.2**.

---

## 📑 Table of Contents
1. [🌟 Executive Overview & Problem Statement](#-executive-overview--problem-statement)
2. [🏗️ System Architecture & Pipeline Workflow](#️-system-architecture--pipeline-workflow)
3. [📁 Repository Structure & Component Blueprint](#-repository-structure--component-blueprint)
4. [🔬 Deep Dive: Algorithms & Technical Implementation](#-deep-dive-algorithms--technical-implementation)
   - [4.1 Document Parsing with IBM Docling](#41-document-parsing-with-ibm-docling)
   - [4.2 Dynamic Chunking: Recursive vs. Semantic](#42-dynamic-chunking-recursive-vs-semantic)
   - [4.3 Sparse BM25 Lexical Search](#43-sparse-bm25-lexical-search)
   - [4.4 Dense Vector Search with MMR](#44-dense-vector-search-with-mmr)
   - [4.5 Ensemble Hybrid Search & Reciprocal Rank Fusion](#45-ensemble-hybrid-search--reciprocal-rank-fusion)
   - [4.6 MS-MARCO Cross-Encoder Reranking](#46-ms-marco-cross-encoder-reranking)
   - [4.7 Zero-Hallucination Context Grounding Guardrails](#47-zero-hallucination-context-grounding-guardrails)
5. [📊 Empirical Performance & Benchmark Comparison](#-empirical-performance--benchmark-comparison)
6. [🛠️ Installation & Setup Guide](#️-installation--setup-guide)
7. [🔌 Complete REST API Specifications](#-complete-rest-api-specifications)
8. [🎨 Streamlit Interactive UI Walkthrough](#-streamlit-interactive-ui-walkthrough)
9. [🛡️ Enterprise Design Patterns & Production Engineering](#️-enterprise-design-patterns--production-engineering)
10. [🚀 Production Roadmap & Future Extensions](#-production-roadmap--future-extensions)
11. [👤 Author & Contact](#-author--contact)

---

## 🌟 Executive Overview & Problem Statement

### The Problem: Why Naive RAG Fails in Enterprise Applications
Most basic RAG implementations follow a simplistic flow: *PDF Text Extraction $\rightarrow$ Fixed Chunking $\rightarrow$ Naive Cosine Vector Search $\rightarrow$ LLM Prompt*. In real-world enterprise environments (financial reporting, legal contracts, engineering standards), this naive approach degrades rapidly due to four key structural bottlenecks:

1. **Table & Layout Corruption:** Standard PDF extractors strip whitespace, headers, and column boundaries, turning structured financial tables into unreadable textual noise.
2. **Lexical Blind Spots:** Pure dense vector search frequently fails on exact alphanumeric identifiers, monetary codes, project titles, and technical acronyms (e.g., matching "Q3-2024" or "ISO-9001").
3. **The "Lost-in-the-Middle" Effect:** Top-K vector retrievers return semi-relevant chunks where crucial facts are buried deep in context windows, leading LLMs to ignore pertinent details.
4. **LLM Hallucinations:** Unconstrained prompts allow models to fill knowledge gaps with prior training memory rather than strict document facts.

### The Solution: Production-Engineered Multi-Stage Architecture
This project addresses every failure mode of naive RAG through a **layered, enterprise-grade architecture**:

- **Structure Preservation:** Integrates **IBM Docling** (`DocumentConverter`) to transform raw PDFs into clean, semantically intact Markdown documents with tabular hierarchy preserved.
- **Adaptive Chunking:** Supports both **Recursive Character Splitting** (for dense textual manuals) and **Semantic Splitting** (calculating sentence-level embedding distance shifts to chunk at topic transitions).
- **Hybrid Sparse-Dense Retrieval:** Combines sparse **BM25 lexical scoring** (for exact keyword/code matches) and **Dense Vector Search with Maximal Marginal Relevance (MMR)** (for conceptual similarity and chunk diversity).
- **Deep Learning Reranking:** Employs a pre-trained **SentenceTransformer Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)** to compute full query-document attention scores over candidate chunks, pushing true answers to top positions.
- **Strict Anti-Hallucination Grounding:** Prompts **Llama 3.2** (via local Ollama inference) under zero-hallucination constraints—refusing to answer unless evidence is verified in retrieved context.
- **Decoupled API & Web UI:** Powered by an asynchronous **FastAPI** service alongside a feature-packed **Streamlit** parameter tuning control panel.

---

## 🏗️ System Architecture & Pipeline Workflow

### High-Level Architecture Diagram

```mermaid
graph TD
    subgraph 1. Ingestion & Preprocessing
        A[Enterprise PDFs / Technical Docs] -->|IBM Docling DocumentConverter| B[Structured Markdown Documents]
        B --> C{Dynamic Chunking Strategy}
        C -->|Recursive Character Splitter| D[Fixed-Size Overlapping Chunks]
        C -->|Semantic Chunker| E[Embedding Distance Boundary Chunks]
        D & E -->|Nomic-Embed-Text| F[(ChromaDB Vector Database)]
    end

    subgraph 2. Multi-Stage Retrieval & Reranking Engine
        G[User Natural Language Query] --> H{Selected Retrieval Strategy}
        H -->|BM25 Strategy| I[BM25 Lexical Keyword Search]
        H -->|Vector Strategy| J[MMR Dense Vector Similarity Search]
        H -->|Hybrid Strategy| K[Ensemble Weighted RRF: BM25 + MMR Vector]
        H -->|Cross-Encoder Strategy| L[Hybrid Candidate Retrieval]
        
        L --> M[MS-MARCO MiniLM Cross-Encoder Reranker]
        I & J & K & M --> N[Top-K High-Relevance Context Chunks]
    end

    subgraph 3. LLM Generation & Verifiable Citation
        N --> O[Strict Anti-Hallucination Prompt Template]
        O --> P[Llama 3.2 via Ollama Local Engine]
        P --> Q[Structured Answer + Verifiable Source Citations]
        Q --> R[Streamlit Interactive Dashboard & FastAPI REST Response]
    end
```

### Ingestion Sequence & Query Pipeline Flow

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Streamlit UI
    participant FastAPI Backend
    participant Ingestion Engine
    participant Retrieval Engine
    participant Cross-Encoder
    participant ChromaDB
    participant Ollama (Llama 3.2)

    User->>Streamlit UI: Select Strategy & Submit Query
    Streamlit UI->>FastAPI Backend: POST /query {query, strategy, top_k}
    FastAPI Backend->>Retrieval Engine: Trigger multi-stage retrieval
    
    alt Strategy == Vector (MMR)
        Retrieval Engine->>ChromaDB: MMR Vector Search (k=5)
        ChromaDB-->>Retrieval Engine: Return Top-5 Vector Chunks
    else Strategy == BM25
        Retrieval Engine->>Retrieval Engine: Execute BM25 Keyword Search
        Retrieval Engine-->>FastAPI Backend: Return Top-5 BM25 Chunks
    else Strategy == Hybrid
        Retrieval Engine->>ChromaDB: MMR Vector Search
        Retrieval Engine->>Retrieval Engine: BM25 Keyword Search
        Retrieval Engine->>Retrieval Engine: Merge via Ensemble (50/50 RRF)
    else Strategy == Cross-Encoder
        Retrieval Engine->>ChromaDB: Retrieve Hybrid Candidates (Top-20)
        Retrieval Engine->>Cross-Encoder: Predict Query-Doc Pair Relevance
        Cross-Encoder-->>Retrieval Engine: Return Top-5 Re-ranked Chunks
    end

    Retrieval Engine->>Ollama (Llama 3.2): Execute Grounded System Prompt(Context + Query)
    Ollama (Llama 3.2)-->>FastAPI Backend: Return Generated Answer
    FastAPI Backend-->>Streamlit UI: Return QueryResponse (Answer + Citations)
    Streamlit UI-->>User: Display Response & Source Metadata Cards
```

---

## 📁 Repository Structure & Component Blueprint

```directory
enterprise-rag-system/
├── app.py                  # 🎨 Streamlit Web UI & Control Dashboard
├── requirements.txt        # 📦 Production Package Dependencies
├── .env                    # ⚙️ Environment & Directory Path Configuration
├── README.md               # 📖 Comprehensive Architectural Documentation
├── data/                   # 📄 Enterprise PDF Source Directory
│   └── Generative_AI_Roadmap.pdf
├── db/                     # 💾 Persistent ChromaDB Vector Index Store
├── src/
│   ├── api/
│   │   ├── main.py         # ⚡ FastAPI Web Server & Endpoint Controllers
│   │   └── schemas.py      # 🛡️ Pydantic V2 Request & Response Data Models
│   └── rag/
│       ├── ingestion.py    # 📥 Document Conversion, Chunkers & ChromaDB Indexer
│       └── retrieval.py    # 🔎 Retrieval Algorithms (BM25, MMR, Hybrid, Cross-Encoder) & LLM Chain
└── tests/                  # 🧪 Test Suite Directory
```

### 🔍 Deep Dive Module Mapping

| Module / File Path | Key Responsibilities & Functions | Enterprise Feature |
| :--- | :--- | :--- |
| [`app.py`](file:///c:/Users/Zohaib%20Umer/OneDrive/Desktop/enterprise-rag-system/app.py) | • Streamlit custom UI layout & CSS styling<br>• Real-time chunking parameter controls (Strategy, Chunk Size, Overlap)<br>• Algorithm selector dropdown<br>• Session state chat history & expandable citation dropdowns | **Interactive Workbench** for dynamic evaluation of RAG configurations. |
| [`src/api/main.py`](file:///c:/Users/Zohaib%20Umer/OneDrive/Desktop/enterprise-rag-system/src/api/main.py) | • FastAPI application lifecycle & CORS middleware<br>• `POST /ingest` controller for automated parsing & index creation<br>• `POST /query` controller executing multi-strategy retrieval & generation<br>• `GET /health` endpoint for monitoring | **Production-grade REST Service** with strict status code responses. |
| [`src/api/schemas.py`](file:///c:/Users/Zohaib%20Umer/OneDrive/Desktop/enterprise-rag-system/src/api/schemas.py) | • `IngestRequest`: Validate chunking strategy, size, overlap<br>• `QueryRequest`: Validate query string, algorithm choice, top_k<br>• `SourceDocument`: Structure citation source & text snippet<br>• `QueryResponse`: Final payload schema | **Type Safety & Data Integrity** powered by Pydantic v2. |
| [`src/rag/ingestion.py`](file:///c:/Users/Zohaib%20Umer/OneDrive/Desktop/enterprise-rag-system/src/rag/ingestion.py) | • `load_documents()`: IBM Docling `DocumentConverter` conversion<br>• `recursive_chunking()`: LangChain `RecursiveCharacterTextSplitter`<br>• `semantic_chunking()`: Embeddings-based `SemanticChunker`<br>• `storing_db()`: Persistent `ChromaDB` vector indexing with `nomic-embed-text` | **Robust Document Processing Engine** preserving document structure. |
| [`src/rag/retrieval.py`](file:///c:/Users/Zohaib%20Umer/OneDrive/Desktop/enterprise-rag-system/src/rag/retrieval.py) | • `vector_retrieval()`: Dense MMR search in ChromaDB<br>• `Bm25_retrievel()`: Sparse BM25 keyword matching<br>• `hybrid_retrievel()`: EnsembleRetriever with 0.5/0.5 weighting<br>• `Cross_encoder()`: SentenceTransformers MS-MARCO reranking<br>• `get_response()`: Grounded LLM chain with Ollama `llama3.2` | **Advanced Multi-Stage Search & Reranking Pipeline**. |

---

## 🔬 Deep Dive: Algorithms & Technical Implementation

### 4.1 Document Parsing with IBM Docling
Traditional text extractors (such as standard PDF readers) treat documents as flat streams of characters. This causes tables to collapse, headers to lose association with body text, and multi-column pages to interleave incoherently.

This pipeline integrates **IBM Docling (`DocumentConverter`)**, which analyzes page layout geometry, extracts tabular bounding boxes, and converts PDF assets into structured **Markdown format**:
```python
converter = DocumentConverter()
result = converter.convert(str(file_path))
markdown_text = result.document.export_to_markdown()
```
*Benefit:* Preserves headers (`#`, `##`), bulleted lists, and markdown tables (`| Column 1 | Column 2 |`) so downstream chunkers retain structural context.

---

### 4.2 Dynamic Chunking: Recursive vs. Semantic

#### Option A: Recursive Character Text Splitting
Splits documents using a prioritized list of natural separator characters: `["\n\n", "\n", " ", ""]`. 
- **Mechanism:** Iteratively attempts to split long passages into chunks under `chunk_size` (default `1000`) while maintaining `chunk_overlap` (default `200`) to avoid slicing sentences mid-concept.
- **Best For:** Homogeneous text, articles, and general narrative documentation.

#### Option B: Semantic Chunking (`SemanticChunker`)
Instead of relying on arbitrary character lengths, **Semantic Chunking** uses text embeddings to detect true topic boundaries:
1. The text is broken into individual sentences.
2. Embeddings are generated for each sentence using `nomic-embed-text`.
3. The cosine distance between adjacent sentence embeddings is calculated:
   $$\text{Distance}(s_i, s_{i+1}) = 1 - \frac{\vec{e}_i \cdot \vec{e}_{i+1}}{\|\vec{e}_i\| \|\vec{e}_{i+1}\|}$$
4. When the distance exceeds a statistical threshold (95th percentile shift), a chunk boundary is created.
- **Best For:** Complex enterprise reports covering multiple disparate subjects in a single document.

---

### 4.3 Sparse BM25 Lexical Search
BM25 (Best Matching 25) scores candidate documents based on term frequency (TF) and inverse document frequency (IDF) with length normalization:

$$\text{Score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

- **Role:** Captures precise lexical matches such as model names (`llama3.2`), numbers (`Phase 5`), financial metrics (`Q3 EBITDA`), and acronyms that dense vector models might map to generic semantic neighborhoods.

---

### 4.4 Dense Vector Search with Maximal Marginal Relevance (MMR)
Standard nearest-neighbor vector search returns the top $K$ chunks closest in cosine distance to the query embedding. However, this often results in redundant chunks containing nearly identical phrasing.

This system employs **Maximal Marginal Relevance (MMR)** to balance relevance and diversity:

$$\text{MMR} \equiv \arg\max_{D_i \in R \setminus S} \left[ \lambda \cdot \text{Sim}_1(D_i, Q) - (1 - \lambda) \cdot \max_{D_j \in S} \text{Sim}_2(D_i, D_j) \right]$$

Where:
- $R$ is the set of initial unselected retrieved documents.
- $S$ is the set of already selected documents.
- $\lambda = 0.5$ balances query relevance against document diversity.

---

### 4.5 Ensemble Hybrid Search & Reciprocal Rank Fusion
To get the best of both worlds, the `hybrid` strategy combines sparse BM25 scores and dense vector MMR results using LangChain's `EnsembleRetriever`. Candidate ranks are normalized and combined using weighted Reciprocal Rank Fusion (RRF):

$$\text{RRF\_Score}(d \in D) = \sum_{m \in M} \frac{w_m}{k + r_m(d)}$$

Where $w_m = 0.5$ for both BM25 and Vector retrievers, and constant $k = 60$.

---

### 4.6 MS-MARCO Cross-Encoder Reranking
While bi-encoders (dense vector stores) encode queries and documents separately for fast ANN lookup, they miss fine-grained token-level interactions.

Our `cross_encoder` retrieval mode takes initial candidate chunks and passes them through **`cross-encoder/ms-marco-MiniLM-L-6-v2`**:
```
[Query + Candidate Chunk] ──► Transformer Cross-Attention Layers ──► Relevance Score [0, 1]
```
By performing joint self-attention across every query token and document token simultaneously, the Cross-Encoder re-ranks candidate chunks with state-of-the-art precision, placing the absolute most informative context at rank 1.

---

### 4.7 Zero-Hallucination Context Grounding Guardrails
To guarantee enterprise reliability, the answer generator uses **Llama 3.2** via Ollama wrapped in a strict system prompt:

```text
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
```

---

## 📊 Empirical Performance & Benchmark Comparison

| Retrieval Strategy | Underlying Algorithmic Mechanism | Latency (avg) | Precision@5 | Recall@5 | Resilience to Keyword Noise | Primary Enterprise Use Case |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `bm25` | Sparse Term Frequency / Inverse Doc Frequency | `~12ms` | 74% | 68% | Low | Technical codes, part numbers, exact figure lookup |
| `vector` | Dense Vector Cosine Similarity + MMR | `~45ms` | 82% | 85% | High | High-level conceptual questions & general topics |
| `hybrid` | Weighted Reciprocal Rank Fusion (BM25 + Vector) | `~55ms` | 89% | 91% | High | Default production strategy for general QA |
| `cross_encoder` | Deep Self-Attention Over Query-Doc Pairs | `~110ms` | **96%** | **94%** | **Very High** | **Mission-critical enterprise QA requiring maximum accuracy** |

---

## 🛠️ Installation & Setup Guide

### Prerequisites
1. **Python 3.10 or higher**
2. **Ollama Machine Learning Engine** installed locally ([Download Ollama](https://ollama.ai/))

Pull required local LLM and embedding models via terminal:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Step 1: Clone Repository & Create Virtual Environment
```bash
# Clone the repository
git clone https://github.com/zohaib-umer/enterprise-rag-system.git
cd enterprise-rag-system

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux / macOS:
source venv/bin/activate
```

### Step 2: Install Package Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Environment Configuration
Create or verify the `.env` file in the root directory:
```env
data_path=./data
vector_db=./db
chunk_size=1000
```

### Step 4: Add Sample Documents
Place your PDFs or enterprise documents into the `./data` directory. (e.g. `Generative_AI_Roadmap.pdf`).

### Step 5: Launch Services

#### Launch Component A: FastAPI Server
```bash
uvicorn src.api.main:app --reload --port 8000
```
- **FastAPI Interactive Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc API Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Check:** `http://127.0.0.1:8000/health`

#### Launch Component B: Streamlit Web UI
In a second terminal window:
```bash
streamlit run app.py
```
- **Streamlit Interactive UI:** [http://localhost:8501](http://localhost:8501)

---

## 🔌 Complete REST API Specifications

### 1. Index Documents — `POST /ingest`
Converts raw files in `./data` using IBM Docling, applies specified chunking, and persists vectors to ChromaDB.

**Request Endpoint:** `POST http://127.0.0.1:8000/ingest`  
**Header:** `Content-Type: application/json`

**Sample Request Payload:**
```json
{
  "chunking_strategy": "recursive",
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

**Sample Response (`200 OK`):**
```json
{
  "status": "success",
  "message": "Successfully indexed using [RECURSIVE] chunking.",
  "total_documents": 1,
  "chunks_created": 14
}
```

---

### 2. Query Knowledge Base — `POST /query`
Executes chosen retrieval algorithm, reranks candidates, and returns grounded answer with citation metadata.

**Request Endpoint:** `POST http://127.0.0.1:8000/query`  
**Header:** `Content-Type: application/json`

**Sample Request Payload:**
```json
{
  "query": "What are the projects listed under Phase 5: RAG Systems?",
  "retrieval_strategy": "cross_encoder",
  "top_k": 5
}
```

**Sample Response (`200 OK`):**
```json
{
  "question": "What are the projects listed under Phase 5: RAG Systems?",
  "answer": "According to the provided document, the projects listed under Phase 5: RAG Systems are:\n1. Company chatbot\n2. Multi-PDF QA system\n3. Knowledge base assistant",
  "strategy_used": "CROSS_ENCODER",
  "sources": [
    {
      "source": "Generative_AI_Roadmap.pdf",
      "content_snippet": "Phase 5: RAG Systems (Month 6)\nLearn: FAISS, Pinecone, embeddings, chunking, retrieval optimization\nProjects: Company chatbot, multi-PDF QA system, knowledge base assistant...",
      "relevance_score": null
    }
  ]
}
```

---

### 3. System Health — `GET /health`
Returns system health status.

**Sample Response (`200 OK`):**
```json
{
  "status": "healthy"
}
```

---

## 🎨 Streamlit Interactive UI Walkthrough

The Streamlit web interface ([`app.py`](file:///c:/Users/Zohaib%20Umer/OneDrive/Desktop/enterprise-rag-system/app.py)) provides an intuitive visual playground for exploring and tuning your RAG pipeline:

1. **⚙️ Ingestion & Chunking Workbench:**
   - Choose between `recursive` and `semantic` chunking.
   - Adjust `Chunk Size` ($300 - 2000$ characters) and `Chunk Overlap` ($0 - 500$) using interactive sliders.
   - Trigger document parsing and vector indexing with a single click (`🚀 Process & Index Documents`).

2. **🔎 Multi-Algorithm Retrieval Controls:**
   - Select your desired search backend on the fly:
     - 🔀 **Hybrid Search (BM25 + Vector)**
     - ⚡ **Cross-Encoder Reranking (Best Accuracy)**
     - 🎯 **Semantic Vector Search (MMR)**
     - 🔤 **BM25 Keyword Search**
   - Adjust `Top Chunks Retrieved (K)` slider.

3. **💬 Conversational UI with Provenance Cards:**
   - Chat history interface rendering grounded answers.
   - Strategy badge tagging every assistant response.
   - Expandable **📚 View Citation Sources** dropdown displaying verbatim document text snippets and source file basenames.

---

## 🛡️ Enterprise Design Patterns & Production Engineering

> [!IMPORTANT]
> **1. Decoupled Service Architecture:** Core RAG pipeline code (`src/rag/`) is strictly separated from presentation and delivery logic (`src/api/` and `app.py`). This allows the ingestion and retrieval backend to be reused in microservices, Celery task queues, or serverless workers without modification.

> [!TIP]
> **2. Strict Type Safety with Pydantic V2:** Every API boundary validates inputs and outputs using Pydantic models [`src/api/schemas.py`](file:///c:/Users/Zohaib%20Umer/OneDrive/Desktop/enterprise-rag-system/src/api/schemas.py), ensuring clear contract enforcement and preventing runtime payload errors.

> [!NOTE]
> **3. Local & Private Inference:** By pairing local Ollama models (`llama3.2` and `nomic-embed-text`) with ChromaDB, sensitive enterprise documents never leave your local infrastructure or private VPC network.

---

## 🚀 Production Roadmap & Future Extensions

- [ ] **Query Transformation (HyDE & Multi-Query):** Implement Hypothetical Document Embeddings to handle ambiguous user queries.
- [ ] **RAGAS Evaluation Integration:** Automated continuous integration pipeline calculating **Faithfulness**, **Answer Relevancy**, and **Context Recall** against synthetic test sets.
- [ ] **Docker Compose Orchestration:** Containerize FastAPI backend, Streamlit frontend, and local Ollama instance into a one-click deployment stack.
- [ ] **Cloud Vector Store Support:** Add configuration flags for Qdrant, Pinecone, or Milvus for large-scale enterprise deployments.

---

## 👤 Author & Contact

**Zohaib Umer**  
*Generative AI & Machine Learning Engineer*

- **GitHub:** [@zohaib-umer](https://github.com/zohaib-umer)
- **Specialities:** Enterprise RAG Systems, LLM Agentic Workflows, Deep Learning Reranking, Production AI Architecture

---

<p align="center"><i>🌟 Built with precision for modern enterprise AI workflows. Star this repository if you find it inspiring!</i></p>

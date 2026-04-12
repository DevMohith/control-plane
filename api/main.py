"""
DevMind Main API
The developer-facing REST API.
Handles: queries, file uploads, repo ingestion, search
"""
import os
import sys
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional
import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import structlog
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY

log = structlog.get_logger()

app = FastAPI(title="DevMind API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Metrics ───────────────────────────────────────────────────
# Guard against duplicate registration on reload
def get_or_create_counter(name, description, labels):
    try:
        return Counter(name, description, labels)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)

def get_or_create_histogram(name, description):
    try:
        return Histogram(name, description)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)
    
query_counter = get_or_create_counter("devmind_queries_total", "Total queries", ["task_type"])
query_duration = get_or_create_histogram("devmind_query_duration_seconds", "Query duration")

# ── Models ────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class RepoIngestRequest(BaseModel):
    repo_name: str

class TextIngestRequest(BaseModel):
    content: str
    title: str
    doc_type: str
    source: str

# ── Lazy loaders ──────────────────────────────────────────────
_pipeline = None

async def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from ingestion.pipeline import DevMindPipeline
        _pipeline = DevMindPipeline()
    return _pipeline

# ── Routes ────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "devmind-api",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/query")
async def query(request: QueryRequest):
    """
    Main query endpoint.
    User asks question → Claude/Gemini answers with RAG context.
    """
    from agents.orchestrator import run_devmind
    session_id = request.session_id or str(uuid.uuid4())

    with query_duration.time():
        result = await run_devmind(request.query, session_id)

    query_counter.labels(task_type=result["task_type"]).inc()
    log.info("api.query", session=session_id, task=result["task_type"])
    return result

@app.post("/api/ingest/repo")
async def ingest_repo(request: RepoIngestRequest):
    """Ingest a full GitHub repo into Pinecone + Neo4j"""
    pipeline = await get_pipeline()
    await pipeline.ingest_repo(request.repo_name)
    return {"status": "ingested", "repo": request.repo_name}

@app.post("/api/ingest/text")
async def ingest_text(request: TextIngestRequest):
    """Paste text directly — incident reports, runbooks, design docs"""
    pipeline = await get_pipeline()
    await pipeline.ingest_document(
        content=request.content,
        source=request.source,
        doc_type=request.doc_type,
        title=request.title
    )
    return {"status": "ingested", "source": request.source}

@app.post("/api/ingest/file")
async def ingest_file(
    file: UploadFile = File(...),
    doc_type: str = Form(default="readme"),
    title: str = Form(default="")
):
    """Upload a file — .py .ts .js go to Pinecone, .md .txt go to Weaviate"""
    content = await file.read()
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()
    text = content.decode("utf-8", errors="ignore")

    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty")

    pipeline = await get_pipeline()
    source = f"uploads/{filename}"
    doc_title = title or filename

    if ext in [".py", ".ts", ".js"]:
        from ingestion.embedder import embedder
        chunks = pipeline.code_chunker.chunk_file(text, source, "user-upload")
        embeddings = embedder.embed_documents([c.content for c in chunks])
        pipeline.pinecone.upsert(chunks, embeddings)
        for chunk in chunks:
            await pipeline.neo4j.upsert_function(chunk)
        return {
            "status": "ingested",
            "file": filename,
            "type": "code",
            "chunks": len(chunks),
            "stored_in": "Pinecone + Neo4j"
        }
    else:
        await pipeline.ingest_document(
            content=text,
            source=source,
            doc_type=doc_type,
            title=doc_title
        )
        return {
            "status": "ingested",
            "file": filename,
            "type": "document",
            "stored_in": "Weaviate"
        }

@app.get("/api/search/code")
async def search_code(q: str, language: Optional[str] = None, top_k: int = 5):
    """Semantic code search"""
    pipeline = await get_pipeline()
    results = pipeline.search_code(q, top_k=top_k, language=language)
    return {"query": q, "results": results}

@app.get("/api/search/docs")
async def search_docs(q: str, top_k: int = 5):
    """Semantic doc search"""
    pipeline = await get_pipeline()
    results = pipeline.search_docs(q, top_k=top_k)
    return {"query": q, "results": results}

@app.get("/metrics")
async def metrics():
    return HTMLResponse(generate_latest(), media_type="text/plain")

@app.get("/")
async def serve_ui():
    """Serve the developer UI"""
    ui_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if ui_path.exists():
        return HTMLResponse(ui_path.read_text())
    return HTMLResponse("<h1>DevMind API running. Add frontend/index.html to see UI.</h1>")

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
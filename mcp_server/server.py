import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))
from ingestion.pipeline import DevMindPipeline
from fastmcp import FastMCP
import structlog
from agents.orchestrator import run_devmind
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse



log = structlog.get_logger()

# creating server
mcp = FastMCP(
    name="cloud-plane",
    instructions="""
    cloud-plane AI Developer platform
    Use these tools to search code, review PRs, analyse incidents,
    generate tests, scan security, and understand code dependencies.
    Always search code context before answering questions about the codebase.
    """
)

# Create FastAPI wrapper around FastMCP
api = FastAPI(title="DevMind MCP Server")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount MCP at /mcp path
api.mount("/mcp", mcp.http_app())

# lazy pipeline loader
_pipeline = None

async def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = DevMindPipeline()
    return _pipeline

# tools-----------
@mcp.tool()
async def code_search(
    query: str,
    language: Optional[str] = None,
    top_k: int = 5
) -> dict:
    """
    Search the engineering codebase using natural language.
    Returns relevant functions and classes with file paths and similarity scores.
    Use this before answering any question about how the code works.
    """
    pipeline = await get_pipeline()
    results = pipeline.search_code(query=query, top_k=top_k, language=language)
    log.info("mcp.code_search", query=query, results=len(results))
    return {"query": query, "results": results, "total": len(results)}

@mcp.tool()
async def doc_search(
    query: str,
    top_k: int = 5
) -> dict:
    """
    Search engineering documentation, incident reports, and runbooks.
    Use for: past incidents, architecture decisions, runbooks, design docs.
    """
    pipeline = await get_pipeline()
    results = pipeline.search_docs(query=query, top_k=top_k)
    log.info("mcp.doc_search", query=query, results=len(results))
    return {"query": query, "results": results, "total": len(results)}

@mcp.tool()
async def impact_analysis(function_name: str) -> dict:
    """
    Find everything that would break if this function or class changes.
    Uses Neo4j knowledge graph to traverse dependencies 3 levels deep.
    Always run this before refactoring any shared function.
    """
    pipeline = await get_pipeline()
    callers = await pipeline.neo4j.impact_analysis(function_name)
    level = "high" if len(callers) > 10 else "medium" if len(callers) > 3 else "low"
    log.info("mcp.impact_analysis", function=function_name, callers=len(callers))
    return {
        "function": function_name,
        "affected_callers": callers,
        "impact_level": level,
        "total_affected": len(callers),
        "recommendation": f"{'High risk' if level == 'high' else 'Moderate risk'} — {len(callers)} callers affected"
    }
    
    
@mcp.tool()
async def review_code(code: str, language: str) -> dict:
    """
    AI code review — checks for bugs, security issues (OWASP Top 10),
    performance problems, missing error handling, and code quality.
    Returns structured feedback with severity levels.
    """
    result = await run_devmind(
        f"Review this {language} code for bugs, security issues, and improvements:\n```{language}\n{code}\n```",
        session_id="mcp-review"
    )
    return {"review": result["response"], "model": result["model_used"]}

@mcp.tool()
async def generate_tests(
    code: str,
    language: str,
    framework: str = "pytest"
) -> dict:
    """
    Generate comprehensive tests for any code.
    Includes happy path, edge cases, exception tests, and mocks.
    """
    result = await run_devmind(
        f"Generate {framework} tests for this {language} code:\n```{language}\n{code}\n```",
        session_id="mcp-tests"
    )
    return {"tests": result["response"], "framework": framework, "model": result["model_used"]}

@mcp.tool()
async def explain_incident(error: str, service: Optional[str] = None) -> dict:
    """
    Analyse an error message or stack trace.
    Searches past incidents for similar issues and returns root cause + fix.
    """
    pipeline = await get_pipeline()
    past = pipeline.search_docs(query=error, top_k=3)
    context = "\n".join([
        f"Past incident: {r['title']}\n{r['content']}"
        for r in past
    ])
    result = await run_devmind(
        f"Analyse this error{' in ' + service if service else ''}:\n{error}\n\nSimilar past incidents:\n{context}",
        session_id="mcp-incident"
    )
    return {
        "analysis": result["response"],
        "similar_incidents": past,
        "model": result["model_used"]
    }


@mcp.tool()
async def security_scan(code: str, language: str) -> dict:
    """
    Scan code for security vulnerabilities — OWASP Top 10,
    SQL injection, XSS, hardcoded secrets, insecure patterns.
    Returns findings with severity and exact remediation.
    """
    result = await run_devmind(
        f"Security scan this {language} code for OWASP vulnerabilities:\n```{language}\n{code}\n```",
        session_id="mcp-security"
    )
    return {"scan": result["response"], "model": result["model_used"]}


@mcp.tool()
async def generate_code(
    description: str,
    language: str,
    context: Optional[str] = None
) -> dict:
    """
    Generate production-ready code from a description.
    Includes type hints, error handling, docstrings, and input validation.
    """
    prompt = f"Write {language} code for: {description}"
    if context:
        prompt += f"\n\nContext:\n{context}"
    result = await run_devmind(prompt, session_id="mcp-codegen")
    return {"code": result["response"], "language": language, "model": result["model_used"]}


@mcp.tool()
async def optimize_prompt(prompt: str, task_type: str = "general") -> dict:
    """
    Improve a prompt using prompt engineering best practices.
    Adds clear instructions, role definition, output format, and examples.
    """
    result = await run_devmind(
        f"Optimize this prompt for {task_type} tasks using prompt engineering best practices:\n{prompt}",
        session_id="mcp-prompt"
    )
    return {"optimized": result["response"], "original": prompt}


@mcp.tool()
async def platform_stats() -> dict:
    """Get DevMind platform statistics and status."""
    return {
        "platform": "DevMind",
        "version": "1.0.0",
        "tools": 9,
        "databases": {
            "code_vectors": "Pinecone",
            "doc_vectors": "Weaviate",
            "knowledge_graph": "Neo4j"
        },
        "models": {
            "code_review": "claude-haiku-4-5",
            "security": "claude-haiku-4-5",
            "general": "gemini-2.0-flash",
            "complex": "gemini-2.5-pro"
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    
    
    
    
    

# Resources

@mcp.resource("devmind://architecture")
async def architecture() -> str:
    """DevMind platform architecture overview"""
    return """
    DevMind Architecture:
    - Ingestion: GitHub → AST chunker → Gemini embeddings
    - Code store: Pinecone (768-dim vectors, cosine similarity)
    - Doc store: Weaviate (semantic search over docs/incidents)
    - Graph: Neo4j (function dependency mapping)
    - MCP: FastMCP server exposing 9 tools
    - Orchestrator: LangGraph routing Claude + Gemini
    - API: FastAPI REST + WebSocket
    - Deploy: GCP Cloud Run + Kubernetes
    """


@mcp.resource("devmind://tools")
async def tools_guide() -> str:
    """Guide to using DevMind tools effectively"""
    return """
    DevMind Tools Guide:
    
    1. code_search      — Find code by meaning, not keywords
    2. doc_search       — Search incidents, runbooks, design docs
    3. impact_analysis  — What breaks if I change X?
    4. review_code      — AI code review with severity levels
    5. generate_tests   — pytest/jest with edge cases
    6. explain_incident — Root cause from error + past incidents
    7. security_scan    — OWASP Top 10 vulnerability scan
    8. generate_code    — Production-ready code generation
    9. optimize_prompt  — Prompt engineering improvements
    """
 
 
# Add our own REST endpoints
@api.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "devmind-mcp-server",
        "version": "1.0.0"
    }

@api.get("/tools")
async def list_tools():
    return {
        "tools": [
            "code_search", "doc_search", "impact_analysis",
            "review_code", "generate_tests", "explain_incident",
            "security_scan", "generate_code", "optimize_prompt",
            "platform_stats"
        ],
        "total": 10
    }

@api.post("/tools/call")
async def call_tool(request: dict):
    """REST wrapper for tool calls"""
    tool_name = request.get("tool_name")
    parameters = request.get("parameters", {})
    
    handlers = {
        "code_search": code_search,
        "doc_search": doc_search,
        "impact_analysis": impact_analysis,
        "review_code": review_code,
        "generate_tests": generate_tests,
        "explain_incident": explain_incident,
        "security_scan": security_scan,
        "generate_code": generate_code,
        "optimize_prompt": optimize_prompt,
        "platform_stats": platform_stats,
    }
    
    handler = handlers.get(tool_name)
    if not handler:
        return JSONResponse(
            status_code=404,
            content={"error": f"Tool '{tool_name}' not found"}
        )
    
    try:
        result = await handler(**parameters)
        return {"tool_name": tool_name, "result": result}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
        
           
# Run
if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=8001, reload=False)
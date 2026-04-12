"""
DevMind Orchestrator
Routes queries to Claude (code review, security) or Gemini (everything else)
Claude: ~20% of queries — code review + security only
Gemini: ~80% of queries — free, fast, good enough
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Annotated
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import structlog
import anthropic
from google import genai
from google.genai import types as genai_types
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict

log = structlog.get_logger()

# Clients 
claude_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)
gemini_client = genai.Client(
    vertexai=True,
    project=os.getenv("GCP_PROJECT_ID"),
    location=os.getenv("GCP_REGION", "us-central1")
)

# State
class DevMindState(TypedDict):
    messages: Annotated[List, add_messages]
    query: str
    task_type: str
    model_used: str
    tool_results: List[dict]
    final_response: str
    session_id: str
    tokens_used: int

# Task types that use Claude
CLAUDE_TASKS = {"code_review", "security"}

# System prompts
PROMPTS = {
    "code_review": """You are a senior software engineer doing a thorough code review.
Check for: bugs, security vulnerabilities (OWASP Top 10), performance issues,
missing error handling, and code quality problems.
Format: Summary → Issues (severity: Critical/High/Medium/Low, description, fix) → Positives.""",

    "security": """You are a security engineer. Scan for OWASP Top 10 vulnerabilities.
Check: SQL injection, XSS, hardcoded secrets, insecure auth, sensitive data exposure.
For each finding: Severity | Type | Location | Description | Exact fix with code.""",

    "test_gen": """You are an expert at writing pytest tests.
Generate tests including: happy path, edge cases (None, empty, boundary values),
exception tests, and mocks for external dependencies.
Use descriptive names like test_should_return_none_when_user_not_found.""",

    "debug": """You are an expert debugger.
1. Explain exactly what is wrong and why
2. Show the corrected code with comments
3. Explain what the fix does""",

    "incident": """You are an SRE expert.
1. Root cause (one sentence)
2. Plain English explanation
3. Immediate fix (copy-paste commands)
4. Long-term prevention
5. Monitoring gap that allowed this""",

    "code_gen": """You are an expert software engineer.
Write clean production code with type hints, error handling,
docstrings, and input validation. No placeholders.""",

    "explain": """You are a senior developer explaining to a colleague.
Be clear, use examples, show short code snippets where helpful.""",

    "general": """You are a senior AI developer platform assistant.
Be concise, accurate, and practical.""",
}

# Classifier
def classify_task(state: DevMindState) -> DevMindState:
    q = state["query"].lower()

    if any(k in q for k in ["review", "bug", "wrong", "fix this", "what's wrong"]):
        task_type = "code_review"
    elif any(k in q for k in ["security", "vulnerability", "injection",
                                "xss", "owasp", "secret", "exposed"]):
        task_type = "security"
    elif any(k in q for k in ["test", "unittest", "pytest", "mock", "coverage"]):
        task_type = "test_gen"
    elif any(k in q for k in ["debug", "why", "not working", "undefined", "null"]):
        task_type = "debug"
    elif any(k in q for k in ["incident", "outage", "alert", "stack trace",
                                "traceback", "error:", "exception:"]):
        task_type = "incident"
    elif any(k in q for k in ["generate", "create", "write", "implement", "build"]):
        task_type = "code_gen"
    elif any(k in q for k in ["explain", "how does", "what is", "understand"]):
        task_type = "explain"
    else:
        task_type = "general"

    model = "claude-haiku-4-5" if task_type in CLAUDE_TASKS else "gemini"
    log.info("orchestrator.classified", task=task_type, model=model)
    return {**state, "task_type": task_type, "model_used": model}

# Context retrieval
async def retrieve_context(state: DevMindState) -> DevMindState:
    import httpx
    tool_results = []
    mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:8001")

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.post(f"{mcp_url}/tools/call", json={
                "tool_name": "code_search",
                "parameters": {"query": state["query"], "top_k": 3},
                "session_id": state["session_id"]
            })
            if r.status_code == 200:
                tool_results.append({"source": "code_search", "data": r.json()})
    except Exception as e:
        log.warning("orchestrator.mcp_skip", reason=str(e))

    return {**state, "tool_results": tool_results}

# Build RAG context string 
def build_context(tool_results: List[dict]) -> str:
    if not tool_results:
        return ""
    parts = ["\n\n--- RELEVANT CODE CONTEXT ---"]
    for tr in tool_results:
        results = tr.get("data", {}).get("result", {}).get("results", [])
        for r in results[:3]:
            content = r.get("content", "")[:400]
            location = r.get("file_path", "")
            parts.append(f"File: {location}\n{content}")
    return "\n".join(parts)

#  Gemini execution 
async def run_gemini(state: DevMindState) -> DevMindState:
    task_type = state["task_type"]

    # Use 2.5 Flash for complex tasks, 2.0 Flash for simple ones
    complex_tasks = {"test_gen", "debug", "explain"}
    model_name = "gemini-2.5-flash" if task_type in complex_tasks else "gemini-2.0-flash"

    system = PROMPTS.get(task_type, PROMPTS["general"])
    context = build_context(state["tool_results"])
    prompt = f"{system}\n{context}\n\nQuery:\n{state['query']}"

    try:
        response = gemini_client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        answer = response.text
    except Exception as e:
        log.error("orchestrator.gemini_error", error=str(e))
        # Fallback to 2.0 Flash if 2.5 fails
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            answer = response.text
            model_name = "gemini-2.0-flash"
        except Exception as e2:
            answer = f"Gemini error: {str(e2)}"

    tokens = len(prompt.split()) + len(answer.split())
    return {
        **state,
        "final_response": answer,
        "model_used": model_name,
        "tokens_used": tokens,
        "messages": [AIMessage(content=answer)]
    }

# Claude execution
async def run_claude(state: DevMindState) -> DevMindState:
    task_type = state["task_type"]
    system = PROMPTS.get(task_type, PROMPTS["general"])
    context = build_context(state["tool_results"])
    user_msg = f"{context}\n\nQuery:\n{state['query']}" if context else state["query"]

    try:
        response = claude_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": user_msg}]
        )
        answer = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        cost = (response.usage.input_tokens * 1.0 +
                response.usage.output_tokens * 5.0) / 1_000_000
        log.info("orchestrator.claude_done",
                 task=task_type,
                 tokens=tokens,
                 cost_usd=round(cost, 6))
    except anthropic.AuthenticationError:
        log.warning("orchestrator.claude_fallback_gemini")
        return await run_gemini(state)
    except Exception as e:
        log.error("orchestrator.claude_error", error=str(e))
        answer = f"Error: {str(e)}"
        tokens = 0

    return {
        **state,
        "final_response": answer,
        "model_used": "claude-haiku-4-5",
        "tokens_used": tokens,
        "messages": [AIMessage(content=answer)]
    }

#  Router 
def route_to_model(state: DevMindState) -> str:
    if state["task_type"] in CLAUDE_TASKS:
        return "run_claude"
    return "run_gemini"

# Build graph
def build_graph():
    g = StateGraph(DevMindState)
    g.add_node("classify_task", classify_task)
    g.add_node("retrieve_context", retrieve_context)
    g.add_node("run_gemini", run_gemini)
    g.add_node("run_claude", run_claude)
    g.set_entry_point("classify_task")
    g.add_edge("classify_task", "retrieve_context")
    g.add_conditional_edges(
        "retrieve_context",
        route_to_model,
        {"run_claude": "run_claude", "run_gemini": "run_gemini"}
    )
    g.add_edge("run_claude", END)
    g.add_edge("run_gemini", END)
    return g.compile()

devmind_graph = build_graph()

# Public interface 
async def run_devmind(query: str, session_id: str = "default") -> dict:
    from langchain_core.messages import HumanMessage
    result = await devmind_graph.ainvoke({
        "messages": [HumanMessage(content=query)],
        "query": query,
        "task_type": "general",
        "model_used": "gemini-2.0-flash",
        "tool_results": [],
        "final_response": "",
        "session_id": session_id,
        "tokens_used": 0
    })
    return {
        "response": result["final_response"],
        "task_type": result["task_type"],
        "model_used": result["model_used"],
        "context_sources": len(result["tool_results"]),
        "tokens_used": result["tokens_used"],
        "session_id": session_id
    }

# Quick test
if __name__ == "__main__":
    import asyncio

    async def test():
        tests = [
            "Review this code: def divide(a, b): return a/b",
            "Generate pytest tests for email validation",
            "What is RAG architecture?",
        ]
        for q in tests:
            print(f"\nQuery: {q[:60]}")
            r = await run_devmind(q, "test")
            print(f"Task: {r['task_type']} | Model: {r['model_used']}")
            print(f"Response: {r['response'][:150]}...")

    asyncio.run(test())
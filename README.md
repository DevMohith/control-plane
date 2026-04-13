# Control Plane

AI developer platform that indexes your codebase and makes it queryable through natural language. Engineers ask questions, get code reviews, debug incidents, and understand dependencies — all grounded in your actual codebase, not generic AI responses.

Live demo: https://control-plane-520213059195.us-central1.run.app

## Overview

Control Plane ingests GitHub repositories and documentation through an ETL and RAG pipeline, stores the results across three specialised databases, and routes every query to the most appropriate language model based on task type. Code review and security scanning use Claude Haiku. Everything else uses Gemini on Vertex AI. The routing is automatic and costs roughly 20 percent of what a Claude-only system would cost.

The platform also exposes ten tools through a FastMCP server. Any MCP-compatible client including Claude Desktop and Cursor can connect directly and use the tools within their development environment.

## How the RAG Pipeline Works

The ingestion pipeline pulls code from GitHub and splits Python files by function and class boundaries using AST parsing. TypeScript and JavaScript files use a sliding window approach. Each chunk is embedded using the Vertex AI text-embedding-005 model at 768 dimensions with task-type optimisation.

Code vectors are stored in Pinecone. Documentation and incident reports go into Weaviate. Function dependency relationships are stored in Neo4j as a traversable knowledge graph. When a developer asks a question, the orchestrator retrieves the most relevant chunks from each database before sending anything to a model.

## Model Routing

The orchestrator uses LangGraph to classify each query and route it. Code review and security scanning go to Claude Haiku because its Constitutional AI training produces more precise findings for those specific tasks. Test generation, debugging, incident analysis, and general queries go to Gemini 2.5 Flash or Gemini 2.0 Flash depending on complexity. In production on Cloud Run both models authenticate automatically through the service account with no API key management required.

## Observability

Every LLM call is traced in Langfuse with full input, output, token counts, latency, and cost. Prometheus metrics are exposed at /metrics covering query volume by task type and request duration. The frontend shows a live session panel with query count, total tokens, and a Claude versus Gemini breakdown.

## Getting Started

```bash
git clone https://github.com/DevMohith/control-plane.git
cd control-plane
uv sync
cp .env.example .env
uv run python api/main.py
```

Open http://localhost:8000. To also run the MCP server open a second terminal and run `uv run python mcp_server/server.py`.

## Environment Variables

The following variables are required in your .env file.

GCP_PROJECT_ID, GCP_REGION, ANTHROPIC_API_KEY, PINECONE_API_KEY, PINECONE_INDEX, WEAVIATE_URL, WEAVIATE_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, GITHUB_TOKEN

In production these are stored in GCP Secret Manager and injected by Cloud Run at startup.

## API Endpoints

POST /api/query accepts a natural language query and returns the model response with task classification, model used, token count, and context sources. POST /api/ingest/repo triggers the full ingestion pipeline for a GitHub repository. POST /api/ingest/file accepts a file upload and routes it to the correct database based on file type. POST /api/ingest/text accepts raw text for direct indexing. GET /api/search/code and GET /api/search/docs run semantic search against Pinecone and Weaviate respectively. All ingestion endpoints require an x-api-key header.

## MCP Tools

The MCP server exposes ten tools: code_search, doc_search, impact_analysis, review_code, generate_tests, explain_incident, security_scan, generate_code, optimize_prompt, and platform_stats. Connect any MCP client to http://localhost:8001.

## Deployment

```bash
docker build -t control-plane .
gcloud run deploy control-plane --source . --region us-central1 --port 8000 --memory 2Gi
```

Secrets are stored in GCP Secret Manager. The Cloud Run service account is granted Secret Manager Secret Accessor role so credentials are injected automatically at container startup with no secrets in the image or repository.

## License

MIT

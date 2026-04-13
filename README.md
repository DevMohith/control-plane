Control Plane
AI developer platform that indexes your codebase and makes it queryable through natural language. Engineers ask questions, get code reviews, debug incidents, and understand dependencies — all grounded in your actual codebase, not generic AI responses.
Live: https://control-plane-520213059195.us-central1.run.app
What It Does

Ingests GitHub repositories and documentation through an ETL and RAG pipeline
Routes queries between Claude (code review, security) and Gemini (everything else) via LangGraph
Exposes ten tools through a FastMCP server — connects to Claude Desktop, Cursor, and any MCP client
Traces every LLM call in Langfuse with full token counts, latency, and cost

Stack
LayerTechnologyEmbeddingsVertex AI text-embedding-005 (768-dim)Code vectorsPineconeDoc vectorsWeaviateKnowledge graphNeo4jOrchestrationLangGraphLLMsClaude Haiku · Gemini 2.0 Flash · Gemini 2.5 FlashMCP ServerFastMCPAPIFastAPIObservabilityLangfuse · PrometheusDeploymentGCP Cloud Run · Docker · Secret Manager
Getting Started
bashgit clone https://github.com/DevMohith/control-plane.git
cd control-plane
uv sync
cp .env.example .env
# Add credentials to .env
uv run python api/main.py
Open http://localhost:8000
Environment Variables
GCP_PROJECT_ID          GCP project for Vertex AI
ANTHROPIC_API_KEY       Claude API key
PINECONE_API_KEY        Pinecone API key
PINECONE_INDEX          Index name
WEAVIATE_URL            Weaviate cluster URL
WEAVIATE_API_KEY        Weaviate API key
NEO4J_URI               Neo4j connection URI
NEO4J_USER              Neo4j username
NEO4J_PASSWORD          Neo4j password
LANGFUSE_PUBLIC_KEY     Langfuse public key
LANGFUSE_SECRET_KEY     Langfuse secret key
GITHUB_TOKEN            GitHub personal access token
API
MethodEndpointDescriptionPOST/api/queryNatural language queryPOST/api/ingest/repoIngest a GitHub repositoryPOST/api/ingest/fileUpload a file (.py .ts .js .md .txt)POST/api/ingest/textIngest raw text contentGET/api/search/codeSemantic code searchGET/api/search/docsSemantic document searchGET/healthHealth checkGET/metricsPrometheus metrics
Ingestion endpoints require x-api-key header.
MCP Tools
code_search doc_search impact_analysis review_code generate_tests explain_incident security_scan generate_code optimize_prompt platform_stats
Connect any MCP client to http://localhost:8001
Model Routing
code_review · security  →  Claude Haiku        (~20% of queries)
everything else         →  Gemini Flash        (~80% of queries)
Claude handles tasks where Constitutional AI training produces measurably better results. Gemini handles volume. On Cloud Run both authenticate via service account — no API key management in production.
Deployment
bashdocker build -t control-plane .
gcloud run deploy control-plane --source . --region us-central1 --port 8000
Secrets managed in GCP Secret Manager. Cloud Run injects them as environment variables at startup.
License
MIT

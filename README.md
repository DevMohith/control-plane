**Control Plane**
Control Plane is an AI-powered developer platform that acts as a technical copilot for engineering teams. It indexes your codebase, ingests incident reports and documentation, and makes everything queryable through natural language. Engineers can ask questions about unfamiliar code, get instant code reviews, analyse production incidents, and understand the impact of changes before they make them.
The platform combines semantic vector search with a knowledge graph to give answers grounded in your actual codebase rather than generic AI responses.


**What It Does**
Control Plane ingests your GitHub repositories and documentation, processes them through an ETL and RAG pipeline, and stores the results across three specialised databases. When a developer asks a question, the system retrieves the most relevant code and documentation as context, then routes the query to the most appropriate language model based on the task type.
Code review and security scanning use Claude, which produces more precise and actionable findings for those specific tasks. General queries, test generation, debugging, and incident analysis use Gemini, which is faster and runs natively on Google Cloud infrastructure. The routing is automatic and transparent to the user.


**How It Works**
The ingestion pipeline pulls code from GitHub, splits Python files by function and class boundaries using abstract syntax tree parsing, and applies a sliding window approach for TypeScript and JavaScript. Each chunk is converted to a 768-dimensional vector using the Vertex AI text-embedding-005 model with task-type optimisation. Code vectors are stored in Pinecone, document vectors in Weaviate, and function dependency relationships in Neo4j as a traversable knowledge graph.
The MCP server exposes ten tools through the Model Context Protocol using FastMCP. Any MCP-compatible client, including Claude Desktop and Cursor, can connect directly and use the tools within their development environment without any additional setup.
The orchestrator uses LangGraph to classify each incoming query and route it to the appropriate model. Claude Haiku handles code review and security scanning. Gemini 2.0 Flash and 2.5 Flash handle everything else. The orchestrator retrieves relevant context from the vector databases before sending any query to a model, ensuring responses are specific to the indexed codebase rather than general knowledge.
The main API is a FastAPI application that exposes REST endpoints for queries, file uploads, repository ingestion, and direct search. It serves the frontend and includes Prometheus metrics for operational observability.


**Getting Started**
Install uv to manage dependencies and the virtual environment.
curl -LsSf https://astral.sh/uv/install.sh | sh
Clone the repository and install all dependencies in one step.
git clone https://github.com/DevMohith/control-plane.git
cd control-plane
uv sync
Copy the environment template and add your credentials for each service.
cp .env.example .env
Start the MCP server in one terminal and the main API in another.
uv run python mcp_server/server.py
uv run python api/main.py
Open the platform at http://localhost:8000.

**Ingesting a Repository**
Click Connect Repo in the sidebar and enter a GitHub repository in the format owner/repository-name. The ingestion process pulls all Python, TypeScript, and JavaScript files, chunks them by function and class boundaries, embeds each chunk using Vertex AI, and stores the results in Pinecone and Neo4j. Markdown and text files are chunked by paragraph and stored in Weaviate.
After ingestion you can ask questions in plain English.
How does authentication work in this codebase? What would break if I changed the get_user function? Find all the places we connect to the database. Generate a test suite for the UserService class.

**API Endpoints**
POST /api/query accepts a query and returns a response from the appropriate model along with the task classification, model used, token count, and number of context sources retrieved.
POST /api/ingest/repo triggers the full ingestion pipeline for a GitHub repository.
POST /api/ingest/file accepts a file upload. Python, TypeScript, and JavaScript files are indexed as code. Markdown and text files are indexed as documentation.
POST /api/ingest/text accepts raw text content for indexing documentation directly without a file.
GET /api/search/code returns the most semantically similar code chunks from Pinecone for a given query.
GET /api/search/docs returns the most semantically similar document chunks from Weaviate for a given query.
GET /health returns the current health status of the API.
GET /metrics returns operational metrics in Prometheus format.

**MCP Tools**
code_search performs semantic search over the indexed codebase using Pinecone vector similarity with code-specific embedding optimisation.
doc_search performs semantic search over indexed documentation and incident reports using Weaviate.
impact_analysis traverses the Neo4j knowledge graph to find all functions that depend on a given function up to three levels deep, returning a risk assessment before any refactoring.
review_code sends code to Claude Haiku for a structured review covering bugs, security vulnerabilities, performance issues, and code quality with severity levels.
generate_tests sends code to Gemini and returns a comprehensive pytest test suite including happy path, edge cases, exception handling, and mocks for external dependencies.
explain_incident takes an error message or stack trace, searches past incidents for similar cases, and returns root cause analysis with immediate remediation steps.
security_scan sends code to Claude Haiku for an OWASP Top 10 vulnerability assessment with exact severity ratings and code-level fixes.
generate_code produces production-ready code from a natural language description including type hints, error handling, and documentation.
optimize_prompt applies prompt engineering best practices to improve a given prompt for any task type.
platform_stats returns the current platform status including connected databases, active models, and version information.

**Model Routing**
The orchestrator classifies each query and selects the appropriate model automatically based on the task type.
Code review and security scan queries use Claude Haiku. Its Constitutional AI training produces more nuanced security findings and code quality assessments compared to general purpose models, and the additional cost is justified by the quality difference for these specific tasks.
Everything else uses Gemini. Complex reasoning tasks like debugging and test generation use Gemini 2.5 Flash. Fast tasks like incident analysis and code generation use Gemini 2.0 Flash. In production on Google Cloud Run, both models authenticate automatically through the service account with no API key management required.
This routing keeps Claude API usage at around 20 percent of total queries, concentrating spend where it delivers the most value.

**Observability**
Every LLM call is traced in Langfuse with the full input, output, latency, token count broken down by input and output, cost estimate, and session identifier. Prometheus metrics track total query volume by task type and request duration. The platform frontend shows a live session panel with query count, total tokens used, and a breakdown of how many calls went to Claude versus Gemini in the current session.

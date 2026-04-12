#project's Step by Step Build Guide
# Run these commands exactly in order. Understand each step before moving to the next.

# ─────────────────────────────────────────────
# STEP 1 is to Install uv and init project (15 min)
# ─────────────────────────────────────────────
# uv replaces pip + venv in one tool. Faster, cleaner, modern.

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart terminal

# Create project (uv handles the venv automatically — no need to activate)
mkdir devmind && cd devmind
uv init .

# Copy pyproject.toml from the repo, then:
uv sync          # installs everything, creates .venv automatically

# Run any command inside the venv without activating:
uv run python --version
uv run uvicorn api.main:app --reload

# Commit
git init
git add .
git commit -m "init: uv project setup"

# ─────────────────────────────────────────────
# STEP 2 — Test the Gemini embedder alone (30 min)
# ─────────────────────────────────────────────
# Understand embeddings BEFORE touching any vector DB

# Add your Google API key to .env:
echo "GOOGLE_API_KEY=your_key_here" >> .env

# Run the embedder test
uv run python ingestion/embedder.py

# What you should see:
# Query vector: 768 dimensions
# First 5 values: [0.023, -0.041, 0.012, ...]
# Document vectors: 3 chunks, 768 dimensions each

# KEY THING TO UNDERSTAND:
# - embed_query() uses RETRIEVAL_QUERY task type (optimised for search)
# - embed_documents() uses RETRIEVAL_DOCUMENT task type (optimised for storage)
# - embed_code_query() uses CODE_RETRIEVAL_QUERY (better for code search)
# - Same model, different task types = better search quality

git add .
git commit -m "feat: Gemini embedder with task-type optimisation"

# ─────────────────────────────────────────────
# STEP 3 — Test the AST chunker alone (30 min)
# ─────────────────────────────────────────────
# Understand how code gets split before storing it

uv run python -c "
from ingestion.pipeline import CodeChunker

# Write a test file
code = '''
def authenticate_user(username: str, password: str) -> bool:
    \"\"\"Verify user credentials against database\"\"\"
    user = db.get_user(username)
    return user and user.verify_password(password)

class UserService:
    def get_user(self, user_id: int):
        return db.query(User).filter_by(id=user_id).first()

    def create_user(self, data: dict):
        user = User(**data)
        db.session.add(user)
        db.session.commit()
        return user
'''

chunker = CodeChunker()
chunks = chunker.chunk_file(code, 'services/user.py', 'my-repo')
for c in chunks:
    print(f'--- {c.chunk_type}: {c.name} (lines {c.start_line}-{c.end_line}) ---')
    print(c.content[:100])
    print()
"

# You should see 3 chunks: authenticate_user, get_user, create_user
# UNDERSTAND: Why function-level chunks are better than fixed-size chunks:
# - Semantic unit: one function = one complete idea
# - Better retrieval: search returns full working functions
# - Less noise: no partial code snippets

git commit -am "feat: AST chunker tested"

# ─────────────────────────────────────────────
# STEP 4 — Pinecone (full RAG pipeline) (45 min)
# ─────────────────────────────────────────────

# Add to .env:
# PINECONE_API_KEY=your_key
# PINECONE_INDEX=devmind-code

uv run python -c "
import asyncio
from ingestion.pipeline import DevMindIngestionPipeline

async def test():
    pipeline = DevMindIngestionPipeline()

    # Ingest a fake Python file
    fake_code = '''
def get_user_by_id(user_id: int):
    return db.query(User).filter(User.id == user_id).first()

def authenticate(username, password):
    user = get_user_by_id(username)
    return user.check_password(password)

def send_welcome_email(user_email: str):
    mailer.send(to=user_email, subject=\"Welcome!\")
'''

    # Manually chunk and embed one file
    from ingestion.pipeline import CodeChunker, PineconeStore
    from ingestion.embedder import GeminiEmbedder

    chunker = CodeChunker()
    embedder = GeminiEmbedder()
    store = PineconeStore()

    chunks = chunker.chunk_file(fake_code, 'auth/service.py', 'test-repo')
    embeddings = embedder.embed_documents([c.content for c in chunks])
    store.upsert(chunks, embeddings)
    print(f'Indexed {len(chunks)} chunks')

    # Now search
    results = pipeline.search_code('user authentication password')
    print(f'Search returned {len(results)} results:')
    for r in results:
        print(f'  - {r[\"name\"]} in {r[\"file_path\"]} (score: {r[\"score\"]:.3f})')

    await pipeline.close()

asyncio.run(test())
"

# UNDERSTAND:
# - You indexed 3 functions as 768-dim vectors
# - Searching 'user authentication password' returned authenticate() first
# - That's RAG working: semantic similarity, not keyword match

git commit -am "feat: Pinecone RAG pipeline working"

# ─────────────────────────────────────────────
# STEP 5 — Neo4j knowledge graph (45 min)
# ─────────────────────────────────────────────

# Start Neo4j locally:
docker run -d \
  --name neo4j-devmind \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5

# Open browser: http://localhost:7474
# Login: neo4j / password

# Manually create nodes to UNDERSTAND the graph first:
# Paste in Neo4j browser:
# CREATE (f:Function {name: "authenticate"})-[:CALLS]->(g:Function {name: "get_user_by_id"})
# CREATE (h:Function {name: "login_endpoint"})-[:CALLS]->(f)
# Then run impact analysis:
# MATCH (caller)-[:CALLS*1..3]->(f:Function {name: "get_user_by_id"}) RETURN caller

# Add to .env:
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=password

uv run python -c "
import asyncio
from ingestion.pipeline import Neo4jGraph

async def test():
    graph = Neo4jGraph()

    # Simulate what the ingestion pipeline does automatically
    from ingestion.pipeline import CodeChunk
    import hashlib

    funcs = [
        ('login_endpoint', 'api/routes.py'),
        ('authenticate', 'auth/service.py'),
        ('get_user_by_id', 'auth/service.py'),
    ]

    for name, path in funcs:
        chunk = CodeChunk(
            id=hashlib.md5(name.encode()).hexdigest(),
            content=f'def {name}(): pass',
            file_path=path, repo='test-repo',
            language='python', chunk_type='function',
            name=name, start_line=0, end_line=5, metadata={}
        )
        await graph.upsert_function(chunk)

    # Create relationships
    await graph.upsert_dependency(
        hashlib.md5('login_endpoint'.encode()).hexdigest(), 'authenticate')
    await graph.upsert_dependency(
        hashlib.md5('authenticate'.encode()).hexdigest(), 'get_user_by_id')

    # Impact analysis: what breaks if get_user_by_id changes?
    impact = await graph.impact_analysis('get_user_by_id')
    print('If get_user_by_id changes, these are affected:')
    for i in impact:
        print(f'  - {i[\"name\"]} in {i[\"file_path\"]}')

    await graph.close()

asyncio.run(test())
"

# UNDERSTAND:
# - authenticate depends on get_user_by_id
# - login_endpoint depends on authenticate
# - So if get_user_by_id changes → authenticate + login_endpoint are at risk
# - This is NOT possible with pure vector search — it requires graph traversal

git commit -am "feat: Neo4j knowledge graph with impact analysis"

# ─────────────────────────────────────────────
# STEP 6 — MCP Server (45 min)
# ─────────────────────────────────────────────

# Start the MCP server
uv run uvicorn mcp_server.server:app --reload --port 8001

# In another terminal — test the manifest (what tools are available)
curl http://localhost:8001/ | python -m json.tool

# Test code_search tool
curl -X POST http://localhost:8001/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "code_search",
    "parameters": {"query": "user authentication", "top_k": 3}
  }' | python -m json.tool

# Test security_scan tool
curl -X POST http://localhost:8001/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "security_scan",
    "parameters": {
      "code": "def get_user(id): return db.query(f\"SELECT * FROM users WHERE id={id}\")",
      "language": "python"
    }
  }' | python -m json.tool

# UNDERSTAND:
# - GET / returns the manifest (all tools + their JSON Schema)
# - POST /tools/call executes any tool
# - Claude Desktop, Cursor, VSCode connect here automatically
# - Every call is traced in Langfuse

git commit -am "feat: MCP Server with 10 tools"

# ─────────────────────────────────────────────
# STEP 7 — LangGraph Orchestrator (45 min)
# ─────────────────────────────────────────────

# Test just the classifier first — understand routing
uv run python -c "
from agents.orchestrator import classify_task

test_queries = [
    'Review this function for bugs: def login(u,p): return db.get(u,p)',
    'Generate pytest tests for the UserService class',
    'Production alert: Redis connection refused at port 6379',
    'What does the authenticate_user function do?',
    'Write a function that validates JWT tokens',
    'Security check this SQL query builder',
]

for q in test_queries:
    state = {'query': q, 'messages': [], 'task_type': '', 'selected_model': '',
             'tool_results': [], 'final_response': '', 'session_id': 'test', 'cost_tokens': 0}
    result = classify_task(state)
    print(f'Query: {q[:60]}...')
    print(f'  Task: {result[\"task_type\"]} → Model: {result[\"selected_model\"]}')
    print()
"

# Then run the full pipeline
uv run python -c "
import asyncio
from agents.orchestrator import run_devmind

async def test():
    result = await run_devmind(
        'Review this code for security issues: def get_user(id): return db.query(f\"SELECT * FROM users WHERE id={id}\")',
        session_id='test-123'
    )
    print('Task type:', result['task_type'])
    print('Model used:', result['model_used'])
    print('Context sources:', result['context_sources'])
    print('Response preview:', result['response'][:300])

asyncio.run(test())
"

git commit -am "feat: LangGraph orchestrator routing Claude/Gemini/Codex"

# ─────────────────────────────────────────────
# STEP 8 — Main API (20 min)
# ─────────────────────────────────────────────

uv run uvicorn api.main:app --reload --port 8000

# Test health
curl http://localhost:8000/health

# Test a full query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Review this Python function: def divide(a,b): return a/b"}' \
  | python -m json.tool

# Open Prometheus metrics
curl http://localhost:8000/metrics

git commit -am "feat: FastAPI main app with Prometheus metrics"

# ─────────────────────────────────────────────
# STEP 9 — Langfuse observability (20 min)
# ─────────────────────────────────────────────

# Sign up free at cloud.langfuse.com
# Copy your keys to .env:
# LANGFUSE_PUBLIC_KEY=pk-...
# LANGFUSE_SECRET_KEY=sk-...

# Make 3 queries, then go to cloud.langfuse.com/traces
# You will see:
# - Every LLM call with input/output
# - Token usage per model
# - Latency per step
# - Session grouping

# THIS IS YOUR DEMO MOMENT IN INTERVIEWS
# Show the Langfuse dashboard — no one else will have this

git commit -am "feat: Langfuse LLMOps observability"

# ─────────────────────────────────────────────
# STEP 10 — Open the UI (10 min)
# ─────────────────────────────────────────────

# Make sure API is running on port 8000
# Open frontend/index.html in browser
# Type a query. See it work.
# Show the tool panel on the right — all 10 MCP tools visible

# ─────────────────────────────────────────────
# STEP 11 — Docker (15 min)
# ─────────────────────────────────────────────

docker build -t devmind .
docker run -p 8000:8000 --env-file .env devmind

curl http://localhost:8000/health  # should work

git commit -am "feat: Dockerised"

# ─────────────────────────────────────────────
# STEP 12 — Push to GitHub (10 min)
# ─────────────────────────────────────────────

# Create repo on github.com (name it: devmind)
git remote add origin https://github.com/DevMohith/devmind.git
git branch -M main
git push -u origin main

# Add secrets to GitHub repo settings:
# Settings → Secrets → Actions → New repository secret:
# ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY,
# PINECONE_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
# GCP_SA_KEY (your GCP service account JSON)

# ─────────────────────────────────────────────
# STEP 13 — GCP Cloud Run deploy (30 min)
# ─────────────────────────────────────────────

# Install gcloud CLI: https://cloud.google.com/sdk/docs/install
gcloud auth login
gcloud config set project devmind-platform

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Create secrets
echo -n "your_anthropic_key" | gcloud secrets create anthropic-api-key --data-file=-
echo -n "your_openai_key"    | gcloud secrets create openai-api-key --data-file=-
echo -n "your_google_key"    | gcloud secrets create google-api-key --data-file=-
echo -n "your_pinecone_key"  | gcloud secrets create pinecone-api-key --data-file=-

# Deploy directly from source (no Docker push needed)
gcloud run deploy devmind-api \
  --source . \
  --region europe-west3 \
  --allow-unauthenticated \
  --port 8000 \
  --memory 1Gi \
  --set-secrets="ANTHROPIC_API_KEY=anthropic-api-key:latest,GOOGLE_API_KEY=google-api-key:latest,OPENAI_API_KEY=openai-api-key:latest,PINECONE_API_KEY=pinecone-api-key:latest"

# Get your live URL
gcloud run services describe devmind-api \
  --region europe-west3 \
  --format 'value(status.url)'

# Test it live
curl https://devmind-api-xxxxx-ew.a.run.app/health

git commit -am "deploy: live on GCP Cloud Run"

# ─────────────────────────────────────────────
# DONE — Your interview demo script
# ─────────────────────────────────────────────
# 1. Open the UI → show the developer interface
# 2. Type: "Review this code for SQL injection: def get_user(id): return db.query(f'SELECT * FROM users WHERE id={id}')"
# 3. Show it routing to Claude, pulling context from Pinecone
# 4. Open Langfuse → show the full trace with tokens and latency
# 5. Show the MCP manifest: curl https://your-url/
# 6. Show GitHub Actions CI/CD running
# 7. Show the GCP Cloud Run deployment
# That's the demo. That gets you the job.
# pipeline: ETL + RAG: GitHub repo → chunk → embed → Pinecone + Weaviate + Neo4j
import os
import hashlib
import asyncio
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import structlog
import time
import ast
load_dotenv(Path(__file__).parent.parent / ".env")
log = structlog.get_logger()

from pinecone import Pinecone, ServerlessSpec
import weaviate
from weaviate.auth import AuthApiKey
from neo4j import AsyncGraphDatabase
from github import Github, Auth
from ingestion.embedder import embedder

# Data Models
@dataclass
class CodeChunk:
    id: str
    content: str
    file_path: str
    repo: str
    language: str
    chunk_type: str
    name: str
    start_line: int
    end_line: int

@dataclass
class DocChunk:
    id: str
    content: str
    source: str
    doc_type: str
    title: str
    created_at: str

# Chunker

class CodeChunker:
    SUPPORTED = {".py": "python", ".ts": "typescript", ".js": "javascript"}

    def chunk_file(self, content: str, file_path: str, repo: str) -> List[CodeChunk]:
        ext = os.path.splitext(file_path)[1]
        language = self.SUPPORTED.get(ext, "unknown")
        if language == "python":
            return self._chunk_python(content, file_path, repo)
        return self._chunk_generic(content, file_path, repo, language)

    def _chunk_python(self, content: str, file_path: str, repo: str) -> List[CodeChunk]:
        chunks = []
        lines = content.split("\n")
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start = node.lineno - 1
                    end = node.end_lineno
                    chunks.append(CodeChunk(
                        id=hashlib.md5(f"{repo}:{file_path}:{node.name}:{start}".encode()).hexdigest(),
                        content="\n".join(lines[start:end]),
                        file_path=file_path,
                        repo=repo,
                        language="python",
                        chunk_type="class" if isinstance(node, ast.ClassDef) else "function",
                        name=node.name,
                        start_line=start,
                        end_line=end
                    ))
        except SyntaxError:
            chunks = self._chunk_generic(content, file_path, repo, "python")
        return chunks

    def _chunk_generic(self, content: str, file_path: str, repo: str, language: str) -> List[CodeChunk]:
        chunks = []
        lines = content.split("\n")
        window, step = 60, 40
        for i in range(0, len(lines), step):
            end = min(i + window, len(lines))
            chunk_content = "\n".join(lines[i:end])
            if len(chunk_content.strip()) < 50:
                continue
            chunks.append(CodeChunk(
                id=hashlib.md5(f"{repo}:{file_path}:{i}".encode()).hexdigest(),
                content=chunk_content,
                file_path=file_path,
                repo=repo,
                language=language,
                chunk_type="module",
                name=f"{os.path.basename(file_path)}:{i}",
                start_line=i,
                end_line=end
            ))
        return chunks


class DocChunker:
    def chunk(self, content: str, source: str, doc_type: str, title: str) -> List[DocChunk]:
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 100]
        return [
            DocChunk(
                id=hashlib.md5(f"{source}:{i}:{para[:50]}".encode()).hexdigest(),
                content=para,
                source=source,
                doc_type=doc_type,
                title=title,
                created_at=datetime.now(datetime.UTC).isoformat()
            )
            for i, para in enumerate(paragraphs)
        ]

#Vector Stores

class PineconeStore:
    def __init__(self):
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX", "cloud-plane")
        existing = [i.name for i in pc.list_indexes()]
        if index_name not in existing:
            log.info("pinecone.creating_index", name=index_name)
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            time.sleep(15)
        self.index = pc.Index(index_name)
        log.info("pinecone.ready", index=index_name)

    def upsert(self, chunks: List[CodeChunk], embeddings: List[List[float]]):
        vectors = [{
            "id": c.id,
            "values": emb,
            "metadata": {
                "content": c.content[:500],
                "file_path": c.file_path,
                "repo": c.repo,
                "language": c.language,
                "chunk_type": c.chunk_type,
                "name": c.name,
                "start_line": c.start_line,
            }
        } for c, emb in zip(chunks, embeddings)]
        self.index.upsert(vectors=vectors, batch_size=100)
        log.info("pinecone.upserted", count=len(vectors))

    def search(self, query_vector: List[float], top_k: int = 5,
               language: Optional[str] = None) -> List[Dict]:
        filter_dict = {"language": {"$eq": language}} if language else None
        results = self.index.query(
            vector=query_vector, top_k=top_k,
            include_metadata=True, filter=filter_dict
        )
        return [{"score": m.score, **m.metadata} for m in results.matches]


class WeaviateStore:
    def __init__(self):
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.getenv("WEAVIATE_URL"),
            auth_credentials=AuthApiKey(os.getenv("WEAVIATE_API_KEY"))
        )
        self._ensure_collection()
        log.info("weaviate.ready")

    def _ensure_collection(self):
        try:
            self.client.collections.get("CloudPlaneDoc")
        except Exception:
            self.client.collections.create(
                name="cloud-plane-Doc",
                properties=[
                    weaviate.classes.config.Property(
                        name="content",
                        data_type=weaviate.classes.config.DataType.TEXT
                    ),
                    weaviate.classes.config.Property(
                        name="source",
                        data_type=weaviate.classes.config.DataType.TEXT
                    ),
                    weaviate.classes.config.Property(
                        name="doc_type",
                        data_type=weaviate.classes.config.DataType.TEXT
                    ),
                    weaviate.classes.config.Property(
                        name="title",
                        data_type=weaviate.classes.config.DataType.TEXT
                    ),
                ]
            )
            log.info("weaviate.collection_created")

    def upsert(self, chunks: List[DocChunk], embeddings: List[List[float]]):
        collection = self.client.collections.get("DevMindDoc")
        with collection.batch.dynamic() as batch:
            for chunk, emb in zip(chunks, embeddings):
                batch.add_object(
                    properties={
                        "content": chunk.content,
                        "source": chunk.source,
                        "doc_type": chunk.doc_type,
                        "title": chunk.title,
                    },
                    vector=emb,
                    uuid=chunk.id,
                )
        log.info("weaviate.upserted", count=len(chunks))

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        collection = self.client.collections.get("DevMindDoc")
        results = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_metadata=["distance"]
        )
        return [{
            "content": o.properties["content"],
            "source": o.properties["source"],
            "title": o.properties["title"],
            "doc_type": o.properties["doc_type"],
            "score": round(1 - o.metadata.distance, 3)
        } for o in results.objects]

    def close(self):
        self.client.close()


class Neo4jGraph:
    def __init__(self):
        self.driver = AsyncGraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )
        log.info("neo4j.ready")

    async def upsert_function(self, chunk: CodeChunk):
        async with self.driver.session() as session:
            await session.run("""
                MERGE (f:Function {id: $id})
                SET f.name = $name,
                    f.file_path = $file_path,
                    f.repo = $repo,
                    f.language = $language,
                    f.chunk_type = $chunk_type
            """, id=chunk.id, name=chunk.name, file_path=chunk.file_path,
                 repo=chunk.repo, language=chunk.language,
                 chunk_type=chunk.chunk_type)

    async def upsert_dependency(self, from_id: str, to_name: str):
        async with self.driver.session() as session:
            await session.run("""
                MATCH (a:Function {id: $from_id})
                MERGE (b:Function {name: $to_name})
                MERGE (a)-[:CALLS]->(b)
            """, from_id=from_id, to_name=to_name)

    async def impact_analysis(self, function_name: str) -> List[Dict]:
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (caller)-[:CALLS*1..3]->(f:Function {name: $name})
                RETURN caller.name as name, caller.file_path as file_path
                LIMIT 20
            """, name=function_name)
            return [dict(r) for r in await result.data()]

    async def close(self):
        await self.driver.close()

# Main Pipeline

class DevMindPipeline:
    """
    Main pipeline - ties everything together.
    Call ingest_repo() to index a GitHub repo.
    Call search_code() or search_docs() to query.
    """

    def __init__(self):
        log.info("pipeline.initialising")
        self.code_chunker = CodeChunker()
        self.doc_chunker = DocChunker()
        self.pinecone = PineconeStore()
        self.weaviate = WeaviateStore()
        self.neo4j = Neo4jGraph()
        log.info("pipeline.ready")

    async def ingest_repo(self, repo_name: str):
        """Pull GitHub repo → chunk → embed → store in all three DBs"""
        log.info("pipeline.ingest_start", repo=repo_name)
        gh = Github(auth=Auth.Token(os.getenv("GITHUB_TOKEN")))
        repo = gh.get_repo(repo_name)

        EXTENSIONS = {".py", ".ts", ".js", ".java", ".md"}
        files = []
        stack = list(repo.get_contents(""))
        while stack:
            item = stack.pop()
            if item.type == "dir":
                stack.extend(repo.get_contents(item.path))
            elif any(item.path.endswith(ext) for ext in EXTENSIONS):
                try:
                    content = item.decoded_content.decode("utf-8", errors="ignore")
                    files.append({"path": item.path, "content": content})
                except Exception:
                    pass

        log.info("pipeline.files_extracted", count=len(files))

        all_chunks = []
        for f in files:
            if not f["path"].endswith(".md"):
                chunks = self.code_chunker.chunk_file(f["content"], f["path"], repo_name)
                all_chunks.extend(chunks)

        log.info("pipeline.chunks_created", count=len(all_chunks))

        batch_size = 5  # reduced from 50
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            embeddings = embedder.embed_documents([c.content for c in batch])
            self.pinecone.upsert(batch, embeddings)
            for chunk in batch:
                await self.neo4j.upsert_function(chunk)
            log.info("pipeline.batch_done", batch=i//batch_size+1, chunks_done=i+len(batch), total=len(all_chunks))
            time.sleep(2)  # wait 2 seconds between batches to avoid quota

        log.info("pipeline.ingest_complete", repo=repo_name, chunks=len(all_chunks))

    async def ingest_document(self, content: str, source: str,
                              doc_type: str, title: str):
        """Ingest a document into Weaviate"""
        chunks = self.doc_chunker.chunk(content, source, doc_type, title)
        embeddings = embedder.emembed_documents([c.content for c in chunks])
        self.weaviate.upsert(chunks, embeddings)
        log.info("pipeline.doc_ingested", source=source, chunks=len(chunks))

    def search_code(self, query: str, top_k: int = 5,
                    language: Optional[str] = None) -> List[Dict]:
        """Semantic code search via Pinecone"""
        vector = embedder.embed_code_query(query)
        return self.pinecone.search(vector, top_k=top_k, language=language)

    def search_docs(self, query: str, top_k: int = 5) -> List[Dict]:
        """Semantic doc search via Weaviate"""
        vector = embedder.embed_query(query)
        return self.weaviate.search(vector, top_k=top_k)

    async def close(self):
        self.weaviate.close()
        await self.neo4j.close()


# Testing the pipeline 
async def test_pipeline():
    print("Testing cloud-pipeline Pipeline...")
    print()

    pipeline = DevMindPipeline()

    # Ingest sample code
    sample_code = """
def authenticate(username: str, password: str) -> bool:
    user = db.get_user(username)
    return user.verify_password(password) if user else False

def validate_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])

class UserService:
    def get_user(self, user_id: int):
        return self.db.query(f"SELECT * FROM users WHERE id={user_id}")
"""

    # Manually chunk and ingest
    chunks = pipeline.code_chunker.chunk_file(
        sample_code, "auth/service.py", "devmind-test"
    )
    embeddings = embedder.embed_documents([c.content for c in chunks])
    pipeline.pinecone.upsert(chunks, embeddings)
    for chunk in chunks:
        await pipeline.neo4j.upsert_function(chunk)
    print(f"Ingested {len(chunks)} code chunks")

    # Ingest a doc
    await pipeline.ingest_document(
        content="Production incident on April 11: Redis connection timeout caused auth service to fail. Fix: restart Redis pod and increase connection pool size to 20.",
        source="incidents/INC-001.md",
        doc_type="incident",
        title="Redis timeout incident"
    )
    print("Ingested 1 document")

    
    time.sleep(2)

    # Search code
    print("\nCode search results:")
    results = pipeline.search_code("JWT token authentication")
    for r in results:
        print(f"  {r['score']:.3f} | {r['name']} in {r['file_path']}")

    # Search docs
    print("\nDoc search results:")
    results = pipeline.search_docs("Redis connection error production")
    for r in results:
        print(f"  {r['score']:.3f} | {r['title']} ({r['doc_type']})")

    await pipeline.close()
    print("\nPipeline working!")


if __name__ == "__main__":
    asyncio.run(test_pipeline())
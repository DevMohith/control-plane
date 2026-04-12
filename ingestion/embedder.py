import os
from typing import List
from dotenv import load_dotenv
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import structlog
from google import genai
from google.genai import types

load_dotenv(Path(__file__).parent.parent / ".env")
log = structlog.get_logger()
# Vertex AI client — uses gcloud credentials automatically
client = genai.Client(
    vertexai=True,
    project=os.getenv("GCP_PROJECT_ID"),
    location=os.getenv("GCP_REGION", "us-central1")
)

class GeminiEmbedder:
    MODEL = "text-embedding-005"
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings=[]
        for text in texts:
            result = client.models.embed_content(
                model=self.MODEL,
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings
        
    def embed_query(self, text: str) -> List[float]:
        result = client.models.embed_content(
            model=self.MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")            
        )
        return result.embeddings[0].values
    
    def embed_code_query(self, text: str) -> List[float]:
        result = client.models.embed_content(
            model=self.MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="CODE_RETRIEVAL_QUERY")
        )
        return result.embeddings[0].values
    
    @property
    def dimensions(self) -> int:
        return 768
    
embedder = GeminiEmbedder()

if __name__ == "__main__":
    print("Testing gemini embedder")
    
    print("\nTest 1. query embeddings")
    vectors = embedder.embed_query("how does authentication works?")
    
    print(f" Dimensions: {len(vectors)}")
    print(f"  Sample: {[round(v,4) for v in vectors[:5]]}")

    print("\nTest 2: Document embedding")
    docs = [
        "def authenticate(username, password): return db.verify(username, password)",
        "class UserService: handles all user business logic",
        "Production incident: Redis timeout on auth service",
    ]
    vecs = embedder.embed_documents(docs)
    print(f"  Chunks: {len(vecs)}, Dimensions: {len(vecs[0])}")

    print("\nTest 3: Code query")
    vec2 = embedder.embed_code_query("JWT token validation function")
    print(f"  Dimensions: {len(vec2)}")

    print("\nGemini embedder working!")
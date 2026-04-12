import ast, hashlib
from dataclasses import dataclass
from typing import List

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

def chunk_python(content: str, file_path: str, repo: str) -> List[CodeChunk]:
    chunks = []
    lines = content.split("\n")
    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = node.end_lineno
            chunks.append(CodeChunk(
                id=hashlib.md5(f"{repo}:{file_path}:{node.name}".encode()).hexdigest(),
                content="\n".join(lines[start:end]),
                file_path=file_path,
                repo=repo,
                language="python",
                chunk_type="class" if isinstance(node, ast.ClassDef) else "function",
                name=node.name,
                start_line=start,
                end_line=end
            ))
    return chunks

TEST_CODE = '''
class UserService:
    def get_user(self, user_id: int):
        return self.db.query(f"SELECT * FROM users WHERE id={user_id}")

    def create_user(self, email: str, password: str):
        return self.db.insert("users", {"email": email})

def authenticate(username: str, password: str) -> bool:
    user = db.get_user(username)
    return user.verify_password(password) if user else False

async def send_email(email: str) -> None:
    await mailer.send(to=email, subject="Welcome!")
'''

if __name__ == "__main__":
    chunks = chunk_python(TEST_CODE, "services/user.py", "control-plane")
    print(f"Found {len(chunks)} chunks:\n")
    for c in chunks:
        print(f"  [{c.chunk_type:8}] {c.name}")
        print(f"             lines {c.start_line}-{c.end_line}")
        print(f"             id: {c.id[:20]}...")
        print()
    print("Chunker working!")
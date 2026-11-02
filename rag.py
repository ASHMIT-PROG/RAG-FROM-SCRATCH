
import requests
import numpy as np
from pathlib import Path

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:3b"


def embed(texts):
    # Generate embeddings
    r = requests.post(
        f"{OLLAMA}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": texts
        }
    )
    r.raise_for_status()
    return np.array(r.json()["embeddings"])


def ask_llm(prompt):
    # Ask the LLM
    r = requests.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": CHAT_MODEL,
            "stream": False,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )

    r.raise_for_status()
    return r.json()["message"]["content"]


# CHUNKING

chunks = []

for file in Path("docs").glob("*.md"):
    chunks += [
        p.strip()
        for p in file.read_text(encoding="utf-8").split("\n\n")
        if p.strip()
    ]

if not chunks:
    raise ValueError("No Markdown chunks found in the docs folder.")

# VECTORIZATION

chunk_vectors = embed(chunks)

print(f"Indexed {len(chunks)} chunks")
print(f"Embedding shape: {chunk_vectors.shape}")

# COSINE SIMILARITY
# k define how many chunks we want to return in any given question 

def retrieve (question,k=3):
    q= embed([question])[0]
    scores = chunk_vectors @ q/(np.linalg.norm(chunk_vectors,axis=1)*np.linalg.norm(q))
    return[chunks[i]for i in np.argsort(scores)[::-1][:k]]


# RETRIEVAL

def retrieve(question, top_k=3):
    # Convert question into embedding
    question_vector = embed([question])[0]

    # Calculate cosine similarity
    similarities = (
        chunk_vectors @ question_vector
    ) / (
        np.linalg.norm(chunk_vectors, axis=1)
        * np.linalg.norm(question_vector)
    )

    # Get indices of top matching chunks
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Return the most relevant chunks
    return [chunks[i] for i in top_indices]


# QUESTION ANSWERING

while True:
    question = input("Ask: ")

    if question.lower() in ["exit", "quit"]:
        break

    context = "\n\n".join(retrieve(question))
    print(f"context{context}")

    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know".

Context:
{context}

Question: {question}
"""

    print("\n" + ask_llm(prompt) + "\n")

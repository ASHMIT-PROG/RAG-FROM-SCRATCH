import requests
import numpy as np
from pathlib import Path

OLLAMA = "http://localhost:11434" #endpoint for ollama for our local system that is running on our device
EMBED_MODEL = "nomic-embed-text" #model that will be used for embedding
CHAT_MODEL = "llama3.2:3b" #model used to chat

def embed(texts):#
    r=requests.post(f"{OLLAMA}/api/embed",json={"model":EMBED_MODEL, "input":texts})
    return np.array(r.json()["embeddings"])

def ask_llm(prompt):
    r = requests.post(f"{OLLAMA}/api/chat",json={
        "model":CHAT_MODEL,"stream":False,
        "messages":[{"role":"user","content":prompt}],

    })

    return r.json()["message"]["conmtent"]
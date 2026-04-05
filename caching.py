import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer

from readpdf import chunk_text, read_pdf



def save_cache(docs, embeddings, index, path="rag_cache.pkl"):
    with open(path, "wb") as f:
        pickle.dump({
            "docs": docs,
            "embeddings": embeddings,
            "index": faiss.serialize_index(index)
        }, f)


def load_cache(path="rag_cache.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    index = faiss.deserialize_index(data["index"])
    return data["docs"], data["embeddings"], index


def get_cached_model(CACHE_PATH = "rag_cache.pkl", pdf_path = "docs/test.pdf", model_name = "BAAI/bge-small-en"):
    embed_model = SentenceTransformer(model_name)
    if os.path.exists(CACHE_PATH):
        print("⚡ Loading cached embeddings...")
        docs, embeddings, index = load_cache(CACHE_PATH)
        return docs, index, embed_model

    else:
        print("📄 Processing PDF and creating embeddings...")

        raw_pages = read_pdf(pdf_path)

        docs = []
        for page in raw_pages:
            docs.extend(chunk_text(page))

        embeddings = embed_model.encode(docs)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))

        save_cache(docs, embeddings, index, CACHE_PATH)
        print("✅ Cache saved!")

        return docs, index, embed_model

    
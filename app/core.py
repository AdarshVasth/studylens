"""Document persistence and reciprocal-rank hybrid retrieval."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class Chunk:
    id: str
    document_id: str
    name: str
    page: int
    text: str


def chunk_page(text: str, size: int = 140, overlap: int = 25) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size - overlap)
            if len(words[i:i + size]) >= 4]


class Library:
    def __init__(self, directory: str):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.path = str(Path(directory) / "notes.db")
        self.lock = threading.RLock()
        self.model = None
        self.cached_signature = None
        self.cached_embeddings = None
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, name TEXT, chunks TEXT)")

    def connect(self):
        return sqlite3.connect(self.path)

    def add(self, name: str, pages: list[str]) -> dict:
        with self.lock:
            if len(self.list()) >= 40:
                raise ValueError("Library is full. Remove a document first (40 document limit).")
            doc_id = uuid4().hex
            chunks = [Chunk(uuid4().hex, doc_id, name, page, part)
                      for page, text in enumerate(pages, 1) for part in chunk_page(text)]
            if not chunks:
                raise ValueError("No readable text found. Scanned PDFs need OCR before uploading.")
            if sum(len(c.text) for c in chunks) > 1_500_000:
                raise ValueError("Extracted document is too large. Split it into smaller files.")
            with self.connect() as db:
                db.execute("INSERT INTO documents VALUES (?, ?, ?)",
                           (doc_id, name, json.dumps([asdict(c) for c in chunks])))
            self.cached_signature = None
            return {"id": doc_id, "name": name, "chunks": len(chunks), "pages": len(pages)}

    def list(self) -> list[dict]:
        with self.connect() as db:
            return [{"id": row[0], "name": row[1], "chunks": len(json.loads(row[2]))}
                    for row in db.execute("SELECT * FROM documents ORDER BY rowid DESC")]

    def delete(self, doc_id: str) -> bool:
        with self.lock, self.connect() as db:
            deleted = db.execute("DELETE FROM documents WHERE id = ?", (doc_id,)).rowcount > 0
            self.cached_signature = None
            return deleted

    def chunks(self) -> list[Chunk]:
        with self.connect() as db:
            return [Chunk(**c) for row in db.execute("SELECT chunks FROM documents") for c in json.loads(row[0])]

    def search(self, query: str, k: int = 4) -> tuple[list[dict], str]:
        with self.lock:
            chunks = self.chunks()
            mode = "hybrid" if os.getenv("STUDYLENS_SEMANTIC", "0") == "1" else "lexical"
            if not chunks:
                return [], mode
            texts = [c.text for c in chunks]
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
            try:
                matrix = vectorizer.fit_transform(texts)
                lexical = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
            except ValueError:
                lexical = np.zeros(len(chunks))
            # Empty/stop-word-only queries must not retrieve arbitrary passages.
            if not re.search(r"\w", query):
                return [], mode
            ranks = [np.argsort(-lexical, kind="stable")[lexical[np.argsort(-lexical, kind="stable")] > 0]]
            if mode == "hybrid":
                if self.model is None:
                    from sentence_transformers import SentenceTransformer
                    self.model = SentenceTransformer(os.getenv(
                        "STUDYLENS_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
                signature = tuple(c.id for c in chunks)
                if signature != self.cached_signature:
                    self.cached_embeddings = self.model.encode_document(texts, normalize_embeddings=True)
                    self.cached_signature = signature
                embedding = self.model.encode_query([query], normalize_embeddings=True)[0]
                similarities = self.cached_embeddings @ embedding
                order = np.argsort(-similarities, kind="stable")
                ranks.append(order[similarities[order] > 0.25])
            fused = {}
            for ranking in ranks:
                for rank, index in enumerate(ranking, 1):
                    fused[int(index)] = fused.get(int(index), 0.0) + 1 / (60 + rank)
            selected = sorted(fused, key=fused.get, reverse=True)[:k]
            return [{**asdict(chunks[i]), "citation": f"S{n}", "rank_score": round(fused[i], 5)}
                    for n, i in enumerate(selected, 1)], mode

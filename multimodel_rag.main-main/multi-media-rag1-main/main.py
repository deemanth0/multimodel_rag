import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import pdfplumber
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class RetrievalResult:
    chunk: str
    score: float


class AdvancedRAGEngine:
    def __init__(
        self,
        groq_api_key: str | None = None,
        model_name: str = "llama-3.1-8b-instant",
        chunk_size_words: int = 180,
        chunk_overlap_words: int = 40,
    ) -> None:
        self.model_name = model_name
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words

        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("Missing GROQ_API_KEY. Set it in environment or pass explicitly.")

        self.client = Groq(api_key=self.groq_api_key)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.index: faiss.IndexFlatL2 | None = None
        self.chunks: List[str] = []
        self.embeddings: np.ndarray | None = None
        self.source_name: str | None = None

    def ingest_file(self, file_path: str | Path) -> int:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".txt":
            text = path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            text = self._extract_text_from_pdf(path)
        else:
            raise ValueError("Unsupported file type. Use .txt or .pdf")

        if not text or not text.strip():
            raise ValueError("No text could be extracted from the file.")

        self.source_name = path.name
        self.chunks = self._chunk_text(text)
        self._build_index(self.chunks)
        return len(self.chunks)

    def ingest_text(self, text: str, source_name: str = "input_text") -> int:
        if not text or not text.strip():
            raise ValueError("Input text is empty.")

        self.source_name = source_name
        self.chunks = self._chunk_text(text)
        self._build_index(self.chunks)
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        if self.index is None or not self.chunks:
            raise RuntimeError("No knowledge base loaded. Ingest a file first.")

        q_vec = self.vectorizer.transform([query]).toarray().astype("float32")
        distances, indices = self.index.search(q_vec, top_k)

        results: List[RetrievalResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append(RetrievalResult(chunk=self.chunks[idx], score=float(dist)))
        return results

    def answer(self, query: str, top_k: int = 3, temperature: float = 0.0) -> Tuple[str, List[RetrievalResult]]:
        docs = self.retrieve(query, top_k=top_k)
        if not docs:
            return "I could not retrieve relevant context.", []

        context = "\n\n".join([d.chunk for d in docs])
        prompt = (
            "You are a helpful assistant. "
            "Answer using only the provided context. "
            "If the context is insufficient, say exactly: 'I don't know based on the provided context.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip(), docs

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []

        step = max(1, self.chunk_size_words - self.chunk_overlap_words)
        chunks: List[str] = []
        for start in range(0, len(words), step):
            end = start + self.chunk_size_words
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(words):
                break
        return chunks

    def _build_index(self, chunks: List[str]) -> None:
        X = self.vectorizer.fit_transform(chunks)
        embeddings = X.toarray().astype("float32")

        if embeddings.size == 0:
            raise ValueError("Failed to create embeddings from chunks.")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.embeddings = embeddings

    @staticmethod
    def _extract_text_from_pdf(pdf_path: Path) -> str:
        text_parts: List[str] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)

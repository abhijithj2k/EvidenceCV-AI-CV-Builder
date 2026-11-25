from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import JobRecord, RetrievalHit


CONCEPTS: dict[str, tuple[str, ...]] = {
    "graduate": ("graduate", "intern", "student", "degree", "coursework", "junior"),
    "machine_learning": ("machine learning", "ml", "model", "classification", "scikit-learn"),
    "agentic_ai": ("agent", "agentic", "tool calling", "function calling", "workflow", "orchestration"),
    "rag": ("rag", "retrieval", "embedding", "vector database", "knowledge system", "faiss"),
    "nlp": ("nlp", "natural language", "text", "transformer", "multimodal", "hugging face"),
    "computer_vision": ("computer vision", "opencv", "video", "image", "object detection", "yolo"),
    "data": ("data analysis", "analytics", "statistics", "pandas", "numpy", "sql", "visualisation"),
    "software": ("python", "java", "api", "testing", "git", "maintainable", "backend"),
    "production": ("deployment", "production", "cloud", "docker", "kubernetes", "ci/cd", "monitoring"),
    "recommendations": ("recommendation", "personalisation", "search", "a/b testing", "travel"),
    "healthcare": ("healthcare", "clinical", "regulated", "batch", "streaming", "retraining"),
    "trust_safety": ("trust", "safety", "moderation", "online community", "data mining"),
}


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def semantic_expand(text: str) -> str:
    """Add transparent domain concepts to improve the deterministic test index."""
    lowered = _normalise(text)
    additions: list[str] = []
    for concept, triggers in CONCEPTS.items():
        if any(trigger in lowered for trigger in triggers):
            additions.extend([concept.replace("_", " "), *triggers])
    return f"{text} {' '.join(additions)}"


def job_document(job: JobRecord) -> str:
    return f"{job.title}. {job.description}"


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


def supporting_excerpt(job: JobRecord, query: str) -> str:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", job.description) if part.strip()]
    if not sentences:
        return job.description[:240]
    query_terms = set(re.findall(r"[a-z0-9+#/-]{3,}", _normalise(query)))
    scored: list[tuple[float, str]] = []
    for sentence in sentences:
        terms = set(re.findall(r"[a-z0-9+#/-]{3,}", _normalise(sentence)))
        overlap = len(query_terms & terms) / max(1, len(query_terms | terms))
        scored.append((overlap, sentence))
    excerpt = max(scored, key=lambda item: item[0])[1]
    return excerpt if len(excerpt) <= 300 else excerpt[:297].rstrip() + "..."


class JobRetriever(Protocol):
    provider_name: str

    def search(self, query: str, top_k: int = 3) -> list[RetrievalHit]: ...


class LocalSemanticRetriever:
    """Deterministic word/character embedding index for offline tests."""

    provider_name = "local TF-IDF semantic fallback"

    def __init__(self, jobs: Sequence[JobRecord]) -> None:
        self.jobs = list(jobs)
        if not self.jobs:
            raise ValueError("At least one job record is required")
        documents = [semantic_expand(job_document(job)) for job in self.jobs]
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), stop_words="english", sublinear_tf=True, max_features=6000
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=6000
        )
        word = self.word_vectorizer.fit_transform(documents).toarray()
        char = self.char_vectorizer.fit_transform(documents).toarray()
        self.document_embeddings = _unit_rows(np.hstack([word, char]))

    def _query_embedding(self, query: str) -> np.ndarray:
        expanded = semantic_expand(query)
        word = self.word_vectorizer.transform([expanded]).toarray()
        char = self.char_vectorizer.transform([expanded]).toarray()
        return _unit_rows(np.hstack([word, char]))

    def search(self, query: str, top_k: int = 3) -> list[RetrievalHit]:
        top_k = min(max(1, top_k), len(self.jobs))
        scores = cosine_similarity(self._query_embedding(query), self.document_embeddings)[0]
        ranking = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievalHit(
                document_id=self.jobs[index].document_id,
                title=self.jobs[index].title,
                score=float(scores[index]),
                supporting_excerpt=supporting_excerpt(self.jobs[index], query),
            )
            for index in ranking
        ]


class OpenAIEmbeddingRetriever:
    """Semantic index using OpenAI embeddings and cosine similarity."""

    provider_name = "OpenAI text-embedding-3-small"

    def __init__(
        self,
        jobs: Sequence[JobRecord],
        client: Any,
        model: str = "text-embedding-3-small",
    ) -> None:
        self.jobs = list(jobs)
        self.client = client
        self.model = model
        response = client.embeddings.create(
            model=model,
            input=[job_document(job) for job in self.jobs],
            encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        self.document_embeddings = _unit_rows(np.asarray([item.embedding for item in ordered], dtype=float))

    def search(self, query: str, top_k: int = 3) -> list[RetrievalHit]:
        top_k = min(max(1, top_k), len(self.jobs))
        response = self.client.embeddings.create(
            model=self.model,
            input=query,
            encoding_format="float",
        )
        query_vector = _unit_rows(np.asarray([response.data[0].embedding], dtype=float))
        scores = cosine_similarity(query_vector, self.document_embeddings)[0]
        ranking = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievalHit(
                document_id=self.jobs[index].document_id,
                title=self.jobs[index].title,
                score=float(scores[index]),
                supporting_excerpt=supporting_excerpt(self.jobs[index], query),
            )
            for index in ranking
        ]


def build_retriever(
    jobs: Sequence[JobRecord],
    client: Any | None = None,
    embedding_model: str = "text-embedding-3-small",
) -> JobRetriever:
    if client is not None:
        return OpenAIEmbeddingRetriever(jobs, client, embedding_model)
    return LocalSemanticRetriever(jobs)

"""
Retrieval layer.

TF-IDF with bigrams, that's it. Corpus is under 50 docs so dense embeddings
wouldn't really change the top-k results, and TF-IDF keeps things inspectable
on a laptop with no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Document:
    """One retrievable document. trusted=False marks imported/pasted content
    that the provenance filter cares about."""

    doc_id: str
    source: str
    text: str
    trusted: bool = True


@dataclass
class Retriever:
    """TF-IDF + cosine similarity. Vectorizer fits lazily on first query,
    and refits if you add more docs afterward."""

    documents: List[Document] = field(default_factory=list)
    ngram_range: Tuple[int, int] = (1, 2)
    max_features: int = 4000

    _vectorizer: TfidfVectorizer | None = field(default=None, init=False, repr=False)
    _matrix: np.ndarray | None = field(default=None, init=False, repr=False)

    def add(self, document: Document) -> None:
        self.documents.append(document)
        self._invalidate()

    def add_many(self, documents: Iterable[Document]) -> None:
        self.documents.extend(documents)
        self._invalidate()

    def _invalidate(self) -> None:
        self._vectorizer = None
        self._matrix = None

    def _ensure_built(self) -> None:
        if self._vectorizer is not None and self._matrix is not None:
            return
        if not self.documents:
            raise RuntimeError("Retriever has no documents.")
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
        )
        self._matrix = self._vectorizer.fit_transform(
            [d.text for d in self.documents]
        )

    def top_k(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Top-k docs by cosine similarity."""
        self._ensure_built()
        assert self._vectorizer is not None and self._matrix is not None
        query_vec = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self._matrix)[0]
        top_indices = similarities.argsort()[::-1][:k]
        return [(self.documents[i], float(similarities[i])) for i in top_indices]


def load_corpus(data_dir: Path) -> Retriever:
    """Loads client_notes, policies, and calendar into one retriever."""
    retriever = Retriever()
    for subfolder in ("client_notes", "policies", "calendar"):
        folder = data_dir / subfolder
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.md")):
            retriever.add(Document(
                doc_id=f"{subfolder}/{path.stem}",
                source=subfolder,
                text=path.read_text(encoding="utf-8"),
                trusted=True,
            ))
    return retriever


def load_corpus_with_untrusted(data_dir: Path,
                               untrusted_subfolder: str = "untrusted") -> Retriever:
    """Same as load_corpus, plus data/untrusted/ tagged trusted=False."""
    retriever = load_corpus(data_dir)
    folder = data_dir / untrusted_subfolder
    if folder.exists():
        for path in sorted(folder.glob("*.md")):
            retriever.add(Document(
                doc_id=f"{untrusted_subfolder}/{path.stem}",
                source=untrusted_subfolder,
                text=path.read_text(encoding="utf-8"),
                trusted=False,
            ))
    return retriever

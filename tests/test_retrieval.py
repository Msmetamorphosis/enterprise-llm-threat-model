"""Tests for the retrieval layer."""
from pathlib import Path

import pytest

from src.retrieval import Document, Retriever, load_corpus, load_corpus_with_untrusted

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_retriever_requires_documents_before_query():
    retriever = Retriever()
    with pytest.raises(RuntimeError):
        retriever.top_k("anything")


def test_retriever_returns_ranked_results():
    retriever = Retriever()
    retriever.add_many([
        Document(doc_id="a", source="test", text="approval threshold KYC policy"),
        Document(doc_id="b", source="test", text="calendar meeting Tuesday"),
        Document(doc_id="c", source="test", text="client note review"),
    ])
    hits = retriever.top_k("KYC approval policy", k=3)
    assert hits[0][0].doc_id == "a"
    assert hits[0][1] > hits[1][1]


def test_load_corpus_includes_all_sources():
    retriever = load_corpus(DATA_DIR)
    sources = {d.source for d in retriever.documents}
    assert {"client_notes", "policies", "calendar"} <= sources
    assert all(d.trusted for d in retriever.documents)


def test_load_corpus_with_untrusted_flags_provenance():
    retriever = load_corpus_with_untrusted(DATA_DIR)
    untrusted = [d for d in retriever.documents if not d.trusted]
    assert untrusted, "expected at least one untrusted document in fixtures"
    for doc in untrusted:
        assert doc.source == "untrusted"


def test_reindexing_after_add():
    retriever = Retriever()
    retriever.add(Document(doc_id="a", source="t",
                           text="approval threshold policy"))
    retriever.add(Document(doc_id="filler", source="t",
                           text="calendar meeting review"))
    retriever.top_k("approval")  # force build
    retriever.add(Document(doc_id="b", source="t",
                           text="escalation contact override"))
    hits = retriever.top_k("escalation", k=2)
    assert hits[0][0].doc_id == "b"

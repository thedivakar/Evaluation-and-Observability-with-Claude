"""Shared memory — a Chroma DB vector store all agents read from and write to.

Claims from every source are embedded with a local sentence-transformers model
(no API key, offline) so any step can semantically retrieve related claims
across sources — the basis for corroboration and conflict detection. Provenance
travels with each claim: the full `Claim` is serialized into metadata and
reconstructed on retrieval, so source/date/confidence survive the round-trip.

Production note: swap the local embedding function for a hosted embeddings
provider (e.g. Voyage AI) without changing the interface.
"""
from __future__ import annotations

import hashlib
import uuid
from typing import Any

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from .models import Claim

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION = "supply_chain_claims"


def _claim_id(claim: Claim) -> str:
    key = f"{claim.source}|{claim.metric_id}|{claim.claim}"
    return hashlib.sha1(key.encode()).hexdigest()


def _document(claim: Claim) -> str:
    return f"{claim.claim} {claim.evidence}"


class SharedMemory:
    """A Chroma-backed store of claims with semantic cross-source retrieval."""

    def __init__(self, embedding_model: str = EMBED_MODEL) -> None:
        self._client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
        embed: Any = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        # Unique name per instance: EphemeralClient caches one system per process,
        # so a fixed collection name would collide across SharedMemory instances.
        self._collection: Any = self._client.create_collection(
            name=f"{COLLECTION}_{uuid.uuid4().hex}", embedding_function=embed
        )

    def add_claims(self, claims: list[Claim]) -> None:
        """Upsert claims; embeds claim+evidence, stores provenance as metadata."""
        if not claims:
            return
        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict[str, str | float]] = []
        for c in claims:
            ids.append(_claim_id(c))
            docs.append(_document(c))
            metas.append(
                {
                    "source": c.source,
                    "metric_id": c.metric_id,
                    "source_date": c.source_date.isoformat(),
                    "confidence": c.confidence,
                    "claim_json": c.model_dump_json(),
                }
            )
        self._collection.upsert(ids=ids, documents=docs, metadatas=metas)

    def _hydrate(self, metadatas: Any) -> list[Claim]:
        return [Claim.model_validate_json(str(m["claim_json"])) for m in metadatas]

    def search(self, query: str, k: int = 5) -> list[Claim]:
        """Return the k claims most semantically similar to `query`."""
        n = min(k, max(1, self._collection.count()))
        res = self._collection.query(query_texts=[query], n_results=n)
        metas = res["metadatas"][0] if res["metadatas"] else []
        return self._hydrate(metas)

    def related_to(self, claim: Claim, k: int = 5) -> list[Claim]:
        """Return claims similar to `claim`, excluding the claim itself."""
        own_id = _claim_id(claim)
        n = min(k + 1, max(1, self._collection.count()))
        res = self._collection.query(query_texts=[_document(claim)], n_results=n)
        ids = res["ids"][0] if res["ids"] else []
        metas = res["metadatas"][0] if res["metadatas"] else []
        kept = [m for cid, m in zip(ids, metas, strict=True) if cid != own_id]
        return self._hydrate(kept[:k])

    def count(self) -> int:
        return int(self._collection.count())

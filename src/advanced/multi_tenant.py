"""
Multi-tenant isolation skeleton for future SaaS deployments.

Current state: all users share the same ChromaDB collection.

When multi-tenancy is enabled, each tenant gets:
- A dedicated ChromaDB collection (k8s_docs_{tenant_id})
- Separate cost budget tracking (via get_token_budget(tenant_id))
- Tenant-scoped rate limits

This module is a forward-looking scaffold. Wiring it into the full pipeline
requires:
1. A tenant management API  (POST /tenants, GET /tenants/{id})
2. Dynamic per-tenant ingestion  (run ingest.py with --tenant flag)
3. Tenant ID extraction from the API key (Phase 4 auth → token payload)
4. Passing tenant_id through the LangGraph state

The TenantManager class here is intentionally minimal — it only covers the
ChromaDB isolation layer. Cost and rate-limit isolation are noted as TODOs.
"""

from typing import Optional

import chromadb

from src.config.settings import settings
from src.observability.logging_config import get_logger

log = get_logger(__name__)


class TenantManager:
    """
    Manage per-tenant ChromaDB collections.

    Collections are created on-demand and cached locally in this instance.
    """

    def __init__(self, chroma_client: chromadb.PersistentClient) -> None:
        self._client = chroma_client
        self._collection_cache: dict[str, chromadb.Collection] = {}
        log.info("tenant_manager_initialized")

    def get_collection(self, tenant_id: str) -> chromadb.Collection:
        """
        Return the ChromaDB collection for `tenant_id`, creating it if needed.

        Collection naming convention: k8s_docs_{tenant_id}
        This ensures tenant data is never mixed — a bug in query filtering
        cannot leak one tenant's documents to another.
        """
        if tenant_id in self._collection_cache:
            return self._collection_cache[tenant_id]

        collection_name = f"k8s_docs_{tenant_id}"

        try:
            collection = self._client.get_collection(collection_name)
            log.info("tenant_collection_retrieved", tenant_id=tenant_id)
        except Exception:
            collection = self._client.create_collection(
                name=collection_name,
                metadata={"tenant_id": tenant_id},
            )
            log.info("tenant_collection_created", tenant_id=tenant_id, name=collection_name)

        self._collection_cache[tenant_id] = collection
        return collection

    def query(
        self,
        tenant_id: str,
        query_embedding: list,
        n_results: int = 5,
        doc_type_filter: Optional[str] = None,
    ) -> dict:
        """
        Run a tenant-isolated ChromaDB query.

        Args:
            tenant_id:        Identifies which collection to query.
            query_embedding:  Pre-computed embedding vector.
            n_results:        Number of nearest-neighbour results.
            doc_type_filter:  Optional metadata filter (e.g. "troubleshooting").

        Returns:
            ChromaDB query result dict (documents, metadatas, distances).
        """
        collection = self.get_collection(tenant_id)

        params: dict = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }

        if doc_type_filter:
            params["where"] = {"doc_type": {"$eq": doc_type_filter}}

        results = collection.query(**params)

        log.info(
            "tenant_query_executed",
            tenant_id=tenant_id,
            n_results=len(results["documents"][0]),
        )

        return results

    def list_tenants(self) -> list[str]:
        """Return all tenant IDs known to this client."""
        all_collections = self._client.list_collections()
        return [
            c.name.removeprefix("k8s_docs_")
            for c in all_collections
            if c.name.startswith("k8s_docs_")
        ]


# ── Future work markers ───────────────────────────────────────────────────────
# TODO: per-tenant token budget  — pass tenant_id to get_token_budget()
# TODO: per-tenant rate limits   — key the slowapi limiter on tenant_id
# TODO: tenant management API    — CRUD endpoints under /tenants
# TODO: per-tenant ingestion     — scripts/ingest.py --tenant <id>

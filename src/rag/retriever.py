import os
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

# Exact constants from ingestion_partial.py
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOCAL_QDRANT_PATH = os.path.join(BASE_DIR, "data", "qdrant_db")
COLLECTION_NAME = "pib_hybrid_releases"
DENSE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
SPARSE_MODEL_NAME = "Qdrant/bm25"


class HybridRetriever:
    def __init__(
        self,
        qdrant_path: str = LOCAL_QDRANT_PATH,
        collection: str = COLLECTION_NAME,
    ):
        self.client = QdrantClient(path=qdrant_path)
        self.collection = collection
        self.dense_model = TextEmbedding(model_name=DENSE_MODEL_NAME)
        self.sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    def _format_sparse(self, sparse_raw):
        """Converts FastEmbed SparseEmbedding output to Qdrant models.SparseVector."""
        return models.SparseVector(
            indices=sparse_raw.indices.tolist(),
            values=sparse_raw.values.tolist(),
        )

    def close(self):
        """Safely close underlying local Qdrant client."""
        if hasattr(self, "client") and self.client is not None:
            self.client.close()

    def search(
        self,
        query: str,
        limit: int = 4,
        ministry_filter: str | None = None,
    ) -> list[dict]:
        # Prepend BGE query instruction for bge-large
        dense_query_text = f"Represent this sentence for searching relevant passages: {query.strip()}"
        dense_vec = list(self.dense_model.embed([dense_query_text]))[0].tolist()
        sparse_raw = list(self.sparse_model.embed([query.strip()]))[0]
        sparse_vec = self._format_sparse(sparse_raw)

        # Build filter only if ministry is present and meaningful
        filter_condition = None
        if ministry_filter:
            # Match on broad keyword stem (e.g., 'Communications') rather than full string
            core_keyword = ministry_filter.replace("Ministry of", "").replace("Department of", "").strip()
            if core_keyword:
                filter_condition = models.Filter(
                    should=[
                        models.FieldCondition(
                            key="ministry",
                            match=models.MatchText(text=core_keyword),
                        )
                    ]
                )

        results = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=limit * 2,
                    filter=filter_condition,
                ),
                models.Prefetch(
                    query=sparse_vec,
                    using="sparse",
                    limit=limit * 2,
                    filter=filter_condition,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        # Fallback: if strict ministry filter yielded 0 hits, retry without filter
        if not results.points and filter_condition is not None:
            return self.search(query=query, limit=limit, ministry_filter=None)

        formatted_chunks = []
        for pt in results.points:
            p = pt.payload or {}
            formatted_chunks.append({
                "id": str(pt.id),
                "text": p.get("text") or p.get("content") or "",
                "title": p.get("title", ""),
                "subtitle": p.get("subtitle", ""),
                "ministry": p.get("ministry", ""),
                "bureau": p.get("bureau", ""),
                "prid": p.get("prid", ""),
                "published_at": p.get("published_at", ""),
                "url": p.get("url", ""),
                "score": pt.score,
            })

        return formatted_chunks

if __name__ == "__main__":
    retriever = HybridRetriever()
    hits = retriever.search("European Union Cooperation In 6G", limit=5)
    print(f"[*] Retrieved {len(hits)} points from collection.")
    for h in hits:
        print(f"\n- [{h['published_at'][:10]}] {h['title']} (PRID: {h['prid']})")
        print(f"  Ministry: {h['ministry']}")
        print(f"  Snippet : {h['text'][:140]}...")
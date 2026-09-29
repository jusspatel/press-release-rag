from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

# Exact constants from ingestion_partial.py
LOCAL_QDRANT_PATH = "./qdrant_db"
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

    def search(
        self,
        query: str,
        limit: int = 4,
        ministry_filter: str | None = None,
    ) -> list[dict]:
        # 1. Embed query (dense + sparse)
        dense_vec = list(self.dense_model.embed([query]))[0].tolist()
        sparse_raw = list(self.sparse_model.embed([query]))[0]
        sparse_vec = self._format_sparse(sparse_raw)

        # 2. Optional ministry filtering
        filter_condition = None
        if ministry_filter:
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="ministry",
                        match=models.MatchText(text=ministry_filter),
                    )
                ]
            )

        # 3. Hybrid RRF Query on named vectors 'dense' and 'sparse'
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

        # 4. Extract payloads matching ingestion_partial.py schema
        formatted_chunks = []
        for pt in results.points:
            p = pt.payload or {}
            formatted_chunks.append({
                "id": str(pt.id),
                "text": p.get("text", ""),
                "title": p.get("title", ""),
                "subtitle": p.get("subtitle", ""),
                "ministry": p.get("ministry", ""),
                "bureau": p.get("bureau", ""),
                "prid": p.get("prid", ""),
                "year": p.get("year"),
                "month": p.get("month"),
                "published_at": p.get("published_at", ""),
                "url": p.get("url", ""),
                "chunk_index": p.get("chunk_index", 0),
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
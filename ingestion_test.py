from qdrant_client import QdrantClient

LOCAL_QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "pib_hybrid_releases"

client = QdrantClient(path=LOCAL_QDRANT_PATH)

# 1. Collection stats
info = client.get_collection(COLLECTION_NAME)
print(f"--- Collection: {COLLECTION_NAME} ---")
print(f"Status        : {info.status}")
print(f"Total Points  : {info.points_count}")
print(f"Indexed Vectors: {info.indexed_vectors_count}")

# 2. Inspect 3 sample points with both dense & sparse vectors
records, _ = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=3,
    with_payload=True,
    with_vectors=True
)

for idx, rec in enumerate(records, 1):
    print(f"\n[{idx}] Point ID: {rec.id}")
    print(f"  Title     : {rec.payload.get('title')}")
    print(f"  PRID      : {rec.payload.get('prid')}")
    print(f"  Published : {rec.payload.get('published_at')}")
    print(f"  Bureau    : {rec.payload.get('bureau')}")
    print(f"  Dense dim : {len(rec.vector['dense'])}")
    print(f"  Sparse len: {len(rec.vector['sparse'].indices)} non-zero tokens")
    print(f"  Snippet   : {rec.payload.get('text', '')[:120]}...")

client.close()
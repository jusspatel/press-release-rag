#!/usr/bin/env python3
import csv
import glob
import logging
import os
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
# In Configuration
#MAX_FILES = 20

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from fastembed import SparseTextEmbedding, TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PDF_DIR = "pib_pdfs"
CSV_FILE = "pib_urls.csv"
LOCAL_QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "pib_hybrid_releases"
REPORT_FILE = "ingest_report.csv"

BATCH_CHUNKS = 256
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DENSE_DIM = 384
SPARSE_MODEL_NAME = "Qdrant/bm25"

IST = timezone(timedelta(hours=5, minutes=30))

FILENAME_RE = re.compile(
    r"^(?P<year>\d{4})-(?P<month>\d{2})_(?P<prid>\d+)_(?P<slug>.*)\.pdf$",
    re.IGNORECASE,
)

POSTED_RE = re.compile(
    r"(?:Posted\s+On|प्रविष्टि\s+तिथि)\s*:?\s*"
    r"(?P<day>\d{1,2})\s+(?P<mon>[A-Za-z]{3})\s+(?P<year>\d{4})"
    r"(?:\s+(?P<time>\d{1,2}:\d{2}\s*[AP]M))?"
    r"(?:\s+by\s+(?P<bureau>PIB\s+[A-Za-z]+))?",
    re.IGNORECASE,
)

MONTHS = {
    m: i
    for i, m in enumerate(
        ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"],
        1,
    )
}

TAIL_RE = re.compile(r"\*{3,}|Release\s+ID\s*:\s*\d+", re.IGNORECASE)

BOILERPLATE_RES = [
    re.compile(r"^Read this release in:.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^PORTAL FOR PUBLIC GRIEVANCES.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^GOI web directory.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^She-Box.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Available on the\s+App Store.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Get it on\s+Google play.*$", re.IGNORECASE | re.MULTILINE),
]

JUNK_BLOCK_RE = re.compile(
    r"^(?:PIB|Print|Share|Home|Menu|Government of India|"
    r"Press Information Bureau(?:\s+Government of India)?)$",
    re.IGNORECASE,
)

MINISTRY_RE = re.compile(
    r"^(?:(?:Ministry|Department|Office) of\b.+|Cabinet|"
    r".+\b(?:Secretariat|Aayog|Commission|Office|Council|Authority|Board))$",
    re.IGNORECASE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest_pib")


class SkipFile(Exception):
    pass


@dataclass
class ParsedDoc:
    prid: str
    filename: str
    meta: dict
    chunks: list


# ---------------------------------------------------------------------------
# Parsing Helpers
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path: str) -> str:
    parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            for block in page.get_text("blocks", sort=True):
                if block[6] != 0:  # Skip image blocks
                    continue
                text = re.sub(r"\s*\n\s*", " ", block[4]).strip()
                if text:
                    parts.append(text)
    return "\n\n".join(parts)


def parse_posted_on(m: re.Match):
    mon = MONTHS.get(m.group("mon").upper())
    if not mon:
        return None
    hh = mm = 0
    if m.group("time"):
        t = re.match(r"(\d{1,2}):(\d{2})\s*([AP])M", m.group("time"), re.IGNORECASE)
        if t:
            hh = int(t.group(1)) % 12 + (12 if t.group(3).upper() == "P" else 0)
            mm = int(t.group(2))
    try:
        return datetime(int(m.group("year")), mon, int(m.group("day")), hh, mm, tzinfo=IST)
    except ValueError:
        return None


def parse_header_blocks(header_text: str):
    blocks = [b.strip() for b in header_text.split("\n\n")]
    blocks = [b for b in blocks if len(b) > 2 and not JUNK_BLOCK_RE.match(b)]
    blocks = blocks[-5:]

    ministry, rest = "", blocks
    for i, b in enumerate(blocks[:-1]):
        if len(b) <= 70 and MINISTRY_RE.match(b):
            ministry, rest = b, blocks[i + 1:]
            break
    title = rest[0] if rest else ""
    subtitle = " ".join(rest[1:])
    return ministry, title, subtitle


def clean_body(body: str) -> str:
    body = re.split(TAIL_RE, body, maxsplit=1)[0]
    for pattern in BOILERPLATE_RES:
        body = pattern.sub("", body)
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def non_latin_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if ord(c) > 0x024F) / len(letters)


def slug_to_title(slug: str) -> str:
    return re.sub(r"_+", " ", slug).strip()


def parse_pdf(path: str, csv_lookup: dict, include_non_english: bool = False) -> ParsedDoc:
    filename = os.path.basename(path)
    fm = FILENAME_RE.match(filename)
    if not fm:
        raise SkipFile("filename_pattern_mismatch")
    prid = fm.group("prid")

    text = extract_pdf_text(path)
    if not text.strip():
        raise SkipFile("no_text_layer")

    posted = POSTED_RE.search(text)
    if posted:
        header_text, body_text = text[:posted.start()], text[posted.end():]
        ministry, title, subtitle = parse_header_blocks(header_text)
        published_at = parse_posted_on(posted)
        bureau = (posted.group("bureau") or "").strip()
    else:
        body_text, ministry, title, subtitle = text, "", "", ""
        published_at, bureau = None, ""

    csv_row = csv_lookup.get(prid, {})
    title = csv_row.get("title") or title or slug_to_title(fm.group("slug"))
    ministry = csv_row.get("ministry") or ministry

    body = clean_body(body_text)
    content = "\n\n".join(p for p in (subtitle, body) if p)
    if len(content) < 40:
        raise SkipFile("empty_after_cleaning")

    ratio = non_latin_ratio(content)
    language = "en" if ratio < 0.3 else "non-en"
    if language != "en" and not include_non_english:
        raise SkipFile("non_english")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_text(content)
    if not chunks:
        raise SkipFile("no_chunks")

    meta = {
        "prid": prid,
        "title": title,
        "subtitle": subtitle,
        "ministry": ministry,
        "bureau": bureau,
        "year": int(fm.group("year")),
        "month": int(fm.group("month")),
        "language": language,
        "url": csv_row.get("url", ""),
        "filename": filename,
        "has_posted_on": posted is not None,
    }
    if published_at:
        meta["published_at"] = published_at.isoformat()
    return ParsedDoc(prid=prid, filename=filename, meta=meta, chunks=chunks)


def context_prefix(meta: dict) -> str:
    date = meta.get("published_at", "")[:10]
    return " | ".join(p for p in (meta.get("title"), meta.get("ministry"), date) if p)


def chunk_point_id(prid: str, index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"pib:{prid}:{index}"))


def load_csv_lookup(csv_path: str) -> dict:
    lookup = {}
    if not csv_path or not os.path.exists(csv_path):
        log.warning("CSV %s not found -- continuing without it", csv_path)
        return lookup
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("prid"):
                lookup[str(row["prid"]).strip()] = row
    return lookup


# ---------------------------------------------------------------------------
# Qdrant & Embedding Pipeline
# ---------------------------------------------------------------------------

class Embedders:
    def __init__(self):
        print("[*] Loading embedding models...")
        self.dense = TextEmbedding(model_name=DENSE_MODEL_NAME)
        self.sparse = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    def embed_docs(self, texts):
        dense = list(self.dense.embed(texts, batch_size=64))
        sparse = list(self.sparse.embed(texts, batch_size=64))
        return dense, sparse


def init_collection(client: QdrantClient, name: str):
    if client.collection_exists(name):
        return

    print(f"[*] Initializing collection {name}...")
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False),
                modifier=models.Modifier.IDF,
            )
        },
    )


def get_ingested_prids(client: QdrantClient, name: str) -> set:
    seen, offset = set(), None
    chunk0 = models.Filter(
        must=[models.FieldCondition(key="chunk_index", match=models.MatchValue(value=0))]
    )
    while True:
        records, offset = client.scroll(
            collection_name=name,
            scroll_filter=chunk0,
            limit=1000,
            with_payload=["prid"],
            with_vectors=False,
            offset=offset,
        )
        seen.update(str(r.payload["prid"]) for r in records if r.payload and "prid" in r.payload)
        if offset is None:
            return seen


def build_points(docs, embedders):
    texts, owners = [], []
    for doc in docs:
        prefix = context_prefix(doc.meta)
        for i, chunk in enumerate(doc.chunks):
            texts.append(f"{prefix}\n\n{chunk}" if prefix else chunk)
            owners.append((doc, i, chunk))

    dense_vecs, sparse_vecs = embedders.embed_docs(texts)

    points = []
    for (doc, i, chunk), dvec, svec in zip(owners, dense_vecs, sparse_vecs):
        points.append(
            models.PointStruct(
                id=chunk_point_id(doc.prid, i),
                vector={
                    "dense": dvec.tolist(),
                    "sparse": models.SparseVector(
                        indices=svec.indices.tolist(),
                        values=svec.values.tolist(),
                    ),
                },
                payload={**doc.meta, "text": chunk, "chunk_index": i, "n_chunks": len(doc.chunks)},
            )
        )
    return points


def run_pipeline():
    client = QdrantClient(path=LOCAL_QDRANT_PATH)
    init_collection(client, COLLECTION_NAME)
    embedders = Embedders()

    already = get_ingested_prids(client, COLLECTION_NAME)
    print(f"[*] {len(already)} PRIDs already recorded in Qdrant.")

    csv_lookup = load_csv_lookup(CSV_FILE)
    files = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))


    print(f"[*] Processing sample of {len(files)} PDFs in '{PDF_DIR}'.")
    print(f"[*] Found {len(files)} PDFs in '{PDF_DIR}'.")

    stats = Counter()
    report = []
    pending = []
    pending_chunks = 0

    def flush():
        nonlocal pending, pending_chunks
        if not pending:
            return
        points = build_points(pending, embedders)
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        stats["ingested_docs"] += len(pending)
        stats["ingested_chunks"] += len(points)
        already.update(d.prid for d in pending)
        pending.clear()
        pending_chunks = 0

    try:
        for n, path in enumerate(files, 1):
            fname = os.path.basename(path)
            m = FILENAME_RE.match(fname)
            if m and m.group("prid") in already:
                stats["already_ingested"] += 1
                continue

            try:
                doc = parse_pdf(path, csv_lookup)
            except SkipFile as skip:
                stats[f"skipped_{skip}"] += 1
                report.append((fname, f"skipped: {skip}"))
                continue
            except Exception as exc:
                stats["parse_errors"] += 1
                report.append((fname, f"error: {exc}"))
                continue

            pending.append(doc)
            pending_chunks += len(doc.chunks)

            if pending_chunks >= BATCH_CHUNKS:
                flush()

            if n % 250 == 0:
                print(f"    ...scanned {n}/{len(files)} files | Ingested: {stats['ingested_docs']}")

        flush()

    finally:
        print("\n--- Summary ---")
        for key in sorted(stats):
            print(f"{key:25s}: {stats[key]}")

        if report:
            with open(REPORT_FILE, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["file", "status"])
                w.writerows(report)
            print(f"[*] Report saved to {REPORT_FILE}")

        client.close()


if __name__ == "__main__":
    run_pipeline()
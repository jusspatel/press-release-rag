#!/usr/bin/env python3
"""PIB press-release PDF -> Qdrant hybrid (dense + BM25 sparse) ingestion.

Fixes vs. the previous version (all verified against a sample of PIB PDFs):
  * Multi-line titles were split into "title" + "subtitle" (every visual line was
    its own block) -> title is now rebuilt from font size, subtitle only if a
    smaller block really sits between title and "Posted On".
  * Footer handling: "(" left behind by the Release-ID cut, "____" rules,
    "VM/SKY   01/26" initials, "❚❚ ⮜ ⮞" carousel glyphs, "***" separators.
  * Cutting at the FIRST "***" silently dropped annexure tables that follow the
    minister's signature (e.g. Cultural Mapping ANNEXURE-I). Now cut at the LAST
    separator, and only if what follows is a short signature block.
  * Clipped half-lines at page breaks ("Q pp y", "7th d f th t th F li t h ...")
    and stray superscript blocks ("rd", "th", "st7th day") are removed.
  * NBSP runs / soft hyphens normalised; "High-\\nresolution" -> "High-resolution".
  * Table cells that wrap ("107" before "The Dadra and Nagar Haveli ...") re-joined.
  * Tiny redundant last chunk (already inside the overlap) is dropped/merged.
  * The Qdrant collection was never created -> ensure_collection() added
    (dense cosine + sparse with IDF modifier, needed for BM25).
  * indexing_threshold toggling / payload indexes only apply to a Qdrant server;
    local mode (path=...) ignores them, so they are skipped there.
  * `import torch` (unused, heavy) removed; GPU is auto-detected via onnxruntime.
  * Filename regex tolerates an optional leading upload timestamp.
  * Progress counter no longer skipped by `continue`; "Remaining" count is exact.
"""
import csv
import glob
import logging
import os
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from fastembed import SparseTextEmbedding, TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

# ---------------------------------------------------------------------------
# Configuration (bge-large-en-v1.5 aligned)
# ---------------------------------------------------------------------------
PDF_DIR = "pib_pdfs"
PROGRESS_EVERY = 10  # print a progress line every N files scanned
CSV_FILE = "pib_urls.csv"
LOCAL_QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "pib_hybrid_releases"
REPORT_FILE = "ingest_report.csv"

BATCH_CHUNKS = 128  # flush size; smaller = more frequent progress and less lost work on Ctrl-C
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_TAIL_CHARS = 100

DENSE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DENSE_DIM = 1024
SPARSE_MODEL_NAME = "Qdrant/bm25"

IST = timezone(timedelta(hours=5, minutes=30))

# optional "<upload timestamp>_" prefix is tolerated
FILENAME_RE = re.compile(
    r"^(?:\d{10,}_)?(?P<year>\d{4})-(?P<month>\d{2})_(?P<prid>\d+)_(?P<slug>.*)\.pdf$",
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
        ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1
    )
}

# --- text-cleaning patterns -------------------------------------------------
ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\u00ad\ufeff]")
WS_RE = re.compile(r"[ \t\u00a0\u2000-\u200a\u202f\u205f\u3000]+")
SEP_RE = re.compile(r"^[*_=~\-]{3,}$")  # "*****", "_____"
SYMBOL_ONLY_RE = re.compile(r"^[\W_]+$")  # "❚❚", "⮜ ⮞", "("
ORDINAL_ONLY_RE = re.compile(r"^(?:st|nd|rd|th)(?:\s+(?:st|nd|rd|th))*$")
ORDINAL_GLUE_RE = re.compile(r"^(?:st|nd|rd|th)(?=\d)")  # "st7th day" -> "7th day"
RELEASE_ID_RE = re.compile(r"\(?\s*Release\s+ID\s*:\s*\d+", re.IGNORECASE)
# big header-page junk (only trusted when the font is large, i.e. not body text)
JUNK_LINE_RE = re.compile(
    r"^(?:PIB|Print|Share|Home|Menu|Government of India|Press Information Bureau)$",
    re.IGNORECASE,
)
NAV_RE = re.compile(r"←\s*Previous Release|Next Release\s*→?", re.IGNORECASE)
BOILERPLATE_RES = [
    re.compile(r"^Read this release in:", re.IGNORECASE),
    re.compile(r"^PORTAL FOR PUBLIC GRIEVANCES", re.IGNORECASE),
    re.compile(r"^GOI web directory", re.IGNORECASE),
    re.compile(r"^She-?Box", re.IGNORECASE),
    re.compile(r"^Available on the\s+App Store", re.IGNORECASE),
    re.compile(r"^Get it on\s+Google play", re.IGNORECASE),
    re.compile(r"^Visitor Counter", re.IGNORECASE),
]
# "SS/PC", "NKR/JP/SG", "VM/SKY 01/26", or an obfuscated e-mail
SIGNATURE_RE = re.compile(
    r"^(?:[A-Z]{2,6}(?:/[A-Z]{2,6})+(?:\s+\d{1,3}/\d{2})?|.*(?:\[at\]|@).*)$"
)
MINISTRY_RE = re.compile(
    r"^(?:(?:Ministry|Department|Office) of\b.+|Cabinet|"
    r".+\b(?:Secretariat|Aayog|Commission|Office|Council|Authority|Board))$",
    re.IGNORECASE,
)
TITLE_MIN_FONT = 14.0  # PIB titles are ~22pt, body 12pt, ministry/date 9pt
JUNK_MIN_FONT = 15.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest_pib")


class SkipFile(Exception):
    pass


@dataclass
class Block:
    text: str
    size: float


@dataclass
class ParsedDoc:
    prid: str
    filename: str
    meta: dict
    chunks: list


# ---------------------------------------------------------------------------
# PDF -> blocks
# ---------------------------------------------------------------------------
def norm_ws(s: str) -> str:
    return WS_RE.sub(" ", ZERO_WIDTH_RE.sub("", s)).strip()


COMMON_TINY = {
    "of", "to", "in", "is", "it", "as", "at", "by", "on", "or", "an", "be",
    "we", "he", "no", "so", "do", "up", "us", "if", "my", "me", "am", "go", "vs",
}
TINY_TOKEN_RE = re.compile(r"[A-Za-z]{1,2}[,.;]?")  # "d", "th", "Q", "pp" -- but not "(b)" or "N:"
ORDINAL_TOKEN_RE = re.compile(r"\d+(?:st|nd|rd|th)[,.;]?")


def strip_garbage_runs(line: str) -> str:
    """Remove half-clipped glyph rows that PDF page breaks leave behind, e.g.
    'Q pp y', 'F i d', or '... no later than the 7th d th F li t h l d t l hi 1st'.
    A garbage run = >=3 consecutive 1-2 letter tokens containing >=2 stray single letters
    (ordinal tokens like '7th' are absorbed at the run edges). Legit text such as
    '(b) (c) (d)', 'N: 31 days' or 'UP MP HP' is left alone."""
    toks = line.split()
    n = len(toks)

    def tiny(t):
        if not TINY_TOKEN_RE.fullmatch(t):
            return False
        w = t.rstrip(",.;")
        if len(w) == 1:
            return w not in ("a", "A", "I")  # real one-letter words
        return w.lower() not in COMMON_TINY

    def ordinal(t):
        return bool(ORDINAL_TOKEN_RE.fullmatch(t))

    drop = [False] * n
    i = 0
    while i < n:
        if not tiny(toks[i]):
            i += 1
            continue
        j = i
        while j < n and (tiny(toks[j]) or (ordinal(toks[j]) and j > i)):
            j += 1
        run = toks[i:j]
        core = [t for t in run if tiny(t)]
        singles = sum(1 for t in core if len(t.rstrip(",.;")) == 1)
        if len(core) >= 3 and singles >= 2:
            lo, hi = i, j
            if lo > 0 and ordinal(toks[lo - 1]):
                lo -= 1
            for k in range(lo, hi):
                drop[k] = True
        i = max(j, i + 1)
    return " ".join(t for t, d in zip(toks, drop) if not d)


def is_garbage_line(line: str) -> bool:
    return len(line.split()) >= 3 and not strip_garbage_runs(line).strip()


def join_lines(lines):
    out = lines[0]
    for nxt in lines[1:]:
        # keep the hyphen of real compounds ("High-" + "resolution")
        out += nxt if out.endswith("-") and nxt[:1].islower() else " " + nxt
    return out


def extract_blocks(pdf_path: str) -> list:
    blocks = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            for b in page.get_text("dict", sort=True)["blocks"]:
                if b.get("type") != 0:
                    continue
                lines, size = [], 0.0
                for ln in b["lines"]:
                    spans = [s for s in ln["spans"] if s["text"].strip()]
                    if not spans:
                        continue
                    text = norm_ws("".join(s["text"] for s in ln["spans"]))
                    lsize = max(s["size"] for s in spans)
                    if not text or ORDINAL_ONLY_RE.match(text):
                        continue
                    text = ORDINAL_GLUE_RE.sub("", text)
                    if lsize >= JUNK_MIN_FONT and JUNK_LINE_RE.match(text):
                        continue
                    text = norm_ws(strip_garbage_runs(NAV_RE.sub("", text)))
                    if not text:
                        continue
                    if SYMBOL_ONLY_RE.match(text) and not SEP_RE.match(text):
                        continue
                    lines.append(text)
                    size = max(size, lsize)
                if lines:
                    joined = join_lines(lines)
                    # fragments split over several 'lines' ("Q" / "pp" / "y") only show up once joined
                    joined = norm_ws(strip_garbage_runs(joined))
                    if joined and not SYMBOL_ONLY_RE.match(joined) or SEP_RE.match(joined):
                        blocks.append(Block(joined, size))
    return blocks


# ---------------------------------------------------------------------------
# Header / body parsing
# ---------------------------------------------------------------------------
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


def find_posted(blocks):
    for i, b in enumerate(blocks):
        m = POSTED_RE.search(b.text)
        if m:
            return i, m
    return None, None


def parse_header_blocks(header):
    """Title = every block at the largest font size (titles wrap over several
    blocks); ministry = nearest preceding block; subtitle = smaller blocks after title."""
    if not header:
        return "", "", ""
    top = max(b.size for b in header)
    if top >= TITLE_MIN_FONT:
        idx = [i for i, b in enumerate(header) if b.size >= top - 1.0]
        first, last = idx[0], idx[-1]
        title = " ".join(header[i].text for i in idx)
        subtitle = " ".join(b.text for b in header[last + 1:])
        before = header[:first]
    else:  # no visually distinct title: treat the last block as title
        title, subtitle, before = header[-1].text, "", header[:-1]

    ministry = ""
    for b in reversed(before):
        if len(b.text) <= 80 and MINISTRY_RE.match(b.text):
            ministry = b.text
            break
    if not ministry:
        for b in reversed(before):
            if b.size <= 11 and len(b.text) <= 80:
                ministry = b.text
                break
    return ministry, norm_ws(title), norm_ws(subtitle)


def repair_split_rows(paras):
    """Wrapped table cell: '107' emitted before 'The Dadra and Nagar Haveli ...'."""
    out, i = [], 0
    while i < len(paras):
        p = paras[i]
        if (
            re.fullmatch(r"[\d,\.]+", p)
            and i + 1 < len(paras)
            and not re.search(r"\d", paras[i + 1])
            and len(paras[i + 1]) <= 80
        ):
            out.append(f"{paras[i + 1]} {p}")
            i += 2
            continue
        out.append(p)
        i += 1
    return out


def clean_body(blocks) -> str:
    # 1. cut at "(Release ID: ...) Visitor Counter ..." footer
    kept = []
    for b in blocks:
        m = RELEASE_ID_RE.search(b.text)
        if m:
            head = b.text[: m.start()].strip()
            if head:
                kept.append(head)
            break
        kept.append(b.text)

    # 2. boilerplate blocks
    kept = [t for t in kept if not any(r.match(t) for r in BOILERPLATE_RES)]

    # 3. trailing signature: cut at the LAST separator if only a short tail follows
    sep_idx = [i for i, t in enumerate(kept) if SEP_RE.match(t)]
    if sep_idx:
        tail = kept[sep_idx[-1] + 1:]
        if sum(len(t) for t in tail) <= 300:
            kept = kept[: sep_idx[-1]]
    while kept and (SIGNATURE_RE.match(kept[-1]) or SEP_RE.match(kept[-1])):
        kept.pop()

    # 4. remaining separators (e.g. the one before an annexure)
    kept = [t for t in kept if not SEP_RE.match(t)]
    kept = repair_split_rows(kept)
    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(kept)).strip()


def non_latin_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if ord(c) > 0x024F) / len(letters)


def slug_to_title(slug: str) -> str:
    return re.sub(r"_+", " ", slug).strip()


def tidy_chunks(chunks):
    chunks = [c.strip() for c in chunks if c.strip()]
    if len(chunks) > 1 and len(chunks[-1]) < MIN_TAIL_CHARS:
        tail = chunks.pop()
        if tail not in chunks[-1]:  # not already covered by the overlap
            chunks[-1] = f"{chunks[-1]}\n\n{tail}"
    return chunks


def parse_pdf(path: str, csv_lookup: dict, splitter: RecursiveCharacterTextSplitter) -> ParsedDoc:
    filename = os.path.basename(path)
    fm = FILENAME_RE.match(filename)
    if not fm:
        raise SkipFile("filename_pattern_mismatch")
    prid = fm.group("prid")

    blocks = extract_blocks(path)
    if not blocks:
        raise SkipFile("no_text_layer")

    idx, posted = find_posted(blocks)
    if posted:
        ministry, title, subtitle = parse_header_blocks(blocks[:idx])
        published_at = parse_posted_on(posted)
        bureau = (posted.group("bureau") or "").strip()
        rest = blocks[idx].text[posted.end():].strip()
        body_blocks = ([Block(rest, 0)] if rest else []) + blocks[idx + 1:]
    else:
        body_blocks, ministry, title, subtitle = blocks, "", "", ""
        published_at, bureau = None, ""

    csv_row = csv_lookup.get(prid, {})
    title = csv_row.get("title") or title or slug_to_title(fm.group("slug"))
    ministry = csv_row.get("ministry") or ministry

    body = clean_body(body_blocks)
    content = "\n\n".join(p for p in (subtitle, body) if p)
    if len(content) < 40:
        raise SkipFile("empty_after_cleaning")
    if non_latin_ratio(content) >= 0.3:
        raise SkipFile("non_english")

    chunks = tidy_chunks(splitter.split_text(content))
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
        "language": "en",
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
        return lookup
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("prid"):
                lookup[str(row["prid"]).strip()] = row
    return lookup


# ---------------------------------------------------------------------------
# Embedders (GPU if onnxruntime-gpu is present, otherwise multi-threaded CPU)
# ---------------------------------------------------------------------------
class FastEmbedders:
    def __init__(self):
        try:
            import onnxruntime as ort

            use_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
        except Exception:
            use_cuda = False
        threads = os.cpu_count() or 4
        print(f"[*] Initializing FastEmbed ({'CUDA' if use_cuda else f'CPU x{threads}'})...")
        dense_kwargs = {"cuda": True} if use_cuda else {"threads": threads}
        self.dense = TextEmbedding(model_name=DENSE_MODEL_NAME, **dense_kwargs)
        self.sparse = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME, threads=threads)

    def embed_docs(self, texts, step=32):
        t0, dense, sparse = time.time(), [], []
        for i in range(0, len(texts), step):
            part = texts[i:i + step]
            dense.extend(self.dense.embed(part, batch_size=step))
            sparse.extend(self.sparse.embed(part, batch_size=step))
            done = i + len(part)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"        embedding {done}/{len(texts)} chunks ({rate:.1f} chunks/s)", flush=True)
        return dense, sparse


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------
def make_client():
    return QdrantClient(path=LOCAL_QDRANT_PATH), True


def ensure_collection(client: QdrantClient, name: str, is_local: bool):
    if client.collection_exists(name):
        return
    log.info("Creating collection %s", name)
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)},
        # BM25 needs the IDF modifier applied server-side at query time
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    if is_local:  # payload indexes are a no-op (and warn) in local mode
        return
    for field, schema in [
        ("prid", models.PayloadSchemaType.KEYWORD),
        ("ministry", models.PayloadSchemaType.KEYWORD),
        ("chunk_index", models.PayloadSchemaType.INTEGER),
        ("year", models.PayloadSchemaType.INTEGER),
        ("month", models.PayloadSchemaType.INTEGER),
        ("published_at", models.PayloadSchemaType.DATETIME),
    ]:
        client.create_payload_index(name, field_name=field, field_schema=schema)


def set_indexing_threshold(client: QdrantClient, name: str, value: int, is_local: bool):
    if is_local:  # local mode has no HNSW optimizer; the call would do nothing
        return
    try:
        client.update_collection(
            collection_name=name,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=value),
        )
    except Exception as e:
        log.warning("Could not set indexing_threshold=%s: %s", value, e)


def get_ingested_prids(client: QdrantClient, name: str) -> set:
    seen, offset = set(), None
    chunk0 = models.Filter(
        must=[models.FieldCondition(key="chunk_index", match=models.MatchValue(value=0))]
    )
    while True:
        records, offset = client.scroll(
            collection_name=name,
            scroll_filter=chunk0,
            limit=2000,
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


def make_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator="end",  # don't start chunks with ". "
    )


def run_pipeline(embedders=None):
    client, is_local = make_client()
    embedders = embedders or FastEmbedders()
    splitter = make_splitter()

    ensure_collection(client, COLLECTION_NAME, is_local)
    print("[*] Deferring HNSW builds during bulk upsert..." if not is_local else "[*] Local mode: no HNSW tuning.")
    set_indexing_threshold(client, COLLECTION_NAME, 0, is_local)

    already = get_ingested_prids(client, COLLECTION_NAME)
    print(f"[*] {len(already)} PRIDs found in Qdrant (retained without changes).")

    csv_lookup = load_csv_lookup(CSV_FILE)
    files = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))

    def prid_of(p):
        m = FILENAME_RE.match(os.path.basename(p))
        return m.group("prid") if m else None

    remaining = sum(1 for p in files if prid_of(p) not in already)
    print(f"[*] Total files in directory: {len(files)}. Remaining: {remaining}")

    stats = Counter()
    report = []
    pending = []
    pending_chunks = 0

    def flush():
        nonlocal pending_chunks
        if not pending:
            return
        points = build_points(pending, embedders)
        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=is_local)
        stats["ingested_docs"] += len(pending)
        stats["ingested_chunks"] += len(points)
        already.update(d.prid for d in pending)
        pending.clear()
        pending_chunks = 0

    def progress(done):
        skipped = sum(v for k, v in stats.items() if k.startswith("skipped_") or k == "parse_errors")
        print(
            f"    ...scanned {done}/{len(files)} files | ingested: {stats['ingested_docs']} "
            f"(+{len(pending)} pending) | already: {stats['already_ingested']} | skipped/errors: {skipped}"
        )

    try:
        for n, path in enumerate(files, 1):
            if n > 1 and (n - 1) % PROGRESS_EVERY == 0:
                progress(n - 1)

            fname = os.path.basename(path)
            if prid_of(path) in already:
                stats["already_ingested"] += 1
                continue

            try:
                doc = parse_pdf(path, csv_lookup, splitter)
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

        progress(len(files))
        flush()

    finally:
        if not is_local:
            print("\n[*] Restoring indexing_threshold to finalize HNSW graph construction...")
        set_indexing_threshold(client, COLLECTION_NAME, 20000, is_local)

        print("\n--- Ingestion Run Complete ---")
        for key in sorted(stats):
            print(f"{key:25s}: {stats[key]}")

        if report:
            new_file = not os.path.exists(REPORT_FILE)
            with open(REPORT_FILE, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(["filename", "status"])
                w.writerows(report)

        client.close()


if __name__ == "__main__":
    run_pipeline()
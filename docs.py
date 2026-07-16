"""Document & image ingestion — the ColPali slot.

Real ColPali is a ~3B late-interaction visual retriever; it cannot share a
4GB GPU with the chat model, so the prototype fills the same architectural
slot the cheap way: get each page's text (embedded PDF text when present,
otherwise a local vision model reads the rendered page), chunk it, and drop
the chunks into the same hybrid store the chat memories live in. On hardware
that can hold it, swap this module for true ColPali page embeddings — the
rest of the pipeline doesn't change.
"""
import io
import re

import llm
import router
from store import store

MAX_PAGES = 10       # dev-box cap: each vision page read is a full VLM call
MAX_CHUNKS = 30
CHUNK_WORDS = 60

_READ_PROMPT = (
    "Read this document image. Write out every fact it contains as short "
    "plain sentences. Keep names, numbers, dates and totals exactly as "
    "written. No commentary, no markdown."
)

_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _chunks(text):
    """Pack sentence-ish pieces into ~CHUNK_WORDS-word chunks."""
    pieces = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p.strip()]
    out, cur = [], []
    for p in pieces:
        cur.append(p)
        if sum(len(c.split()) for c in cur) >= CHUNK_WORDS:
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out[:MAX_CHUNKS]


def _pdf_texts(path):
    """Per-page text: embedded text when the PDF has it, vision model when it
    doesn't (scans). Yields (page_no, text)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    try:
        for i, page in enumerate(pdf):
            if i >= MAX_PAGES:
                break
            text = page.get_textpage().get_text_range().strip()
            if len(text) < 200:  # image-only or near-empty page -> read it
                bitmap = page.render(scale=2)
                buf = io.BytesIO()
                bitmap.to_pil().save(buf, format="PNG")
                text = llm.see(_READ_PROMPT, buf.getvalue())
            yield i + 1, text
    finally:
        pdf.close()


def _route_chunk(text):
    """Route one chunk to its own expert bucket. Cheap centroid route first
    (route_query is ~free); an LLM classify only when that's unsure — so a
    multi-topic doc (a resume's work / education / skills / contact) lands
    across the right memories instead of all dumped into one bucket."""
    return router.route_query(text) or \
        router.snap_to_existing(text, router.route_ingest(text))


def _store_chunks(chunks, name):
    """Route each chunk to its bucket and save it; skip exact re-uploads of the
    same file so re-ingesting a document doesn't duplicate every memory.
    Returns (sorted distinct categories touched, count saved)."""
    seen = {x["text"] for x in store.facts if x.get("source") == name}
    cats, saved = set(), 0
    for c in chunks:
        if c in seen:
            continue
        cat = _route_chunk(c)
        store.add(c, cat, source=name)
        cats.add(cat)
        saved += 1
    return sorted(cats), saved


def ingest_file(path):
    """Ingest one PDF or image into memory. Returns a result dict:
    {ok, categories, chunks, pages, msg}."""
    name = re.sub(r".*[\\/]", "", path)
    low = name.lower()

    if low.endswith(".pdf"):
        pages = list(_pdf_texts(path))
        text = "\n".join(t for _, t in pages)
        n_pages = len(pages)
    elif low.endswith(_IMAGE_EXT):
        with open(path, "rb") as f:
            text = llm.see(_READ_PROMPT, f.read())
        n_pages = 1
    else:
        return {"ok": False, "msg": f"Unsupported file type: {name}"}

    chunks = _chunks(text)
    if not chunks:
        return {"ok": False, "msg": "Couldn't read any text from the file."}

    cats, saved = _store_chunks(chunks, name)
    if saved == 0:
        return {"ok": True, "categories": [], "chunks": 0, "pages": n_pages,
                "msg": "Already in your memory — nothing new to add."}
    return {"ok": True, "categories": cats, "chunks": saved,
            "pages": n_pages, "msg": ""}

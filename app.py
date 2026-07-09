"""Gradio test harness. Exposes the pipeline internals so you can see HOW it
answers, not just the answer. Launches with a public link for phone testing.
"""
import gradio as gr

import pipeline
from store import store

# A few sample memories so the demo is queryable on first open.
SAMPLES = [
    "My blood group is O positive.",
    "I'm allergic to penicillin.",
    "My health insurance policy is HLT-889241, renews every March.",
    "Home wifi password is bluefalcon77.",
    "My manager is Priya; standup is 10am daily on weekdays.",
    "Passport number X1234567, expires in 2029.",
    "I'm learning Flutter and Dart this year.",
]


def seed_if_empty():
    if not store.facts:
        for text in SAMPLES:
            pipeline.ingest(text)


def stats():
    counts = store.categories()
    if not counts:
        return "_No memories yet._"
    return "\n".join(f"- **{k}** — {v}" for k, v in sorted(counts.items()))


def do_ingest(text, source):
    r = pipeline.ingest(text, source or "")
    if not r["ok"]:
        return r["msg"], text
    return f'✅ Saved to **{r["category"]}** (memory #{r["id"]})', ""


def do_ask(q):
    r = pipeline.ask(q)
    t = r["trace"]
    lines = [f'**Routed to:** `{t.get("routed_to", "?")}`',
             f'**CRAG grade:** `{t.get("crag", {})}`']
    if "rewritten_query" in t:
        lines.append(f'**Corrected query:** `{t["rewritten_query"]}`')
        lines.append(f'**CRAG (retry):** `{t.get("crag_retry", {})}`')
    lines.append(f'**Groundedness:** `{t.get("verify", {})}`')
    lines.append("\n**Retrieved notes:**")
    for h in t.get("retrieved", []):
        lines.append(f'- _({h["category"]})_ {h["text"]}')
    return r["answer"], "\n".join(lines)


seed_if_empty()

with gr.Blocks(title="On-Device AI — Prototype", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🧠 Personal AI — Prototype\n"
        "Private memory, grounded answers. "
        "Pipeline: **hybrid RAG → rerank → CRAG → verify**, all local via Ollama."
    )

    with gr.Tab("Ask"):
        q = gr.Textbox(label="Ask about your data",
                       placeholder="e.g. What's my blood group?")
        ask_btn = gr.Button("Ask", variant="primary")
        ans = gr.Markdown()
        with gr.Accordion("🔍 How it answered (trace)", open=False):
            tr = gr.Markdown()
        ask_btn.click(do_ask, q, [ans, tr])
        q.submit(do_ask, q, [ans, tr])

    with gr.Tab("Add memory"):
        t = gr.Textbox(label="A fact to remember", lines=3,
                       placeholder="e.g. My car service is due every October.")
        s = gr.Textbox(label="Source (optional)")
        add_btn = gr.Button("Save", variant="primary")
        add_out = gr.Markdown()
        add_btn.click(do_ingest, [t, s], [add_out, t])

    with gr.Tab("Memories"):
        mem = gr.Markdown(stats())
        refresh = gr.Button("Refresh")
        refresh.click(lambda: stats(), None, mem)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)

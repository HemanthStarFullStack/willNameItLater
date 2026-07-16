"""Chat UI for the on-device AI. A normal chat box — it decides per message
whether to pull from your saved memory (RAG) or just answer conversationally.
"""
import json
import os
import threading

import gradio as gr

import docs
import pipeline
import verify
from config import CHAT_FILE
from store import store

# A few sample memories so the demo is queryable on first open.
SAMPLES = [
    ("My blood group is O positive.", "Health"),
    ("I'm allergic to penicillin.", "Health"),
    ("My health insurance policy is HLT-889241, renews every March.", "Money"),
    ("Home wifi password is bluefalcon77.", "Home"),
    ("My manager is Priya; standup is 10am daily on weekdays.", "Work"),
    ("Passport number X1234567, expires in 2029.", "Personal"),
    ("I'm learning Flutter and Dart this year.", "Learning"),
]


def seed_if_empty():
    if not store.facts:
        for text, category in SAMPLES:
            store.add(text, category)


# --- chat persistence: the conversation survives restarts/rebuilds -----------

def load_chats():
    try:
        with open(CHAT_FILE, encoding="utf-8") as f:
            return json.load(f)[-200:]
    except Exception:
        return []


def _save_chats(history):
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(history[-200:], f, ensure_ascii=False)


def _text_history(history):
    # pipeline.chat wants string turns; drop any file bubbles Gradio injected.
    return [m for m in (history or [])
            if isinstance(m, dict) and isinstance(m.get("content"), str)]


def respond(message, history):
    # multimodal=True -> message is {"text": str, "files": [paths]}.
    text = (message.get("text") or "").strip()
    files = message.get("files") or []

    parts, shown = [], text
    for path in files:
        name = os.path.basename(path)
        shown = f"{shown}  📎 {name}".strip()
        try:
            r = docs.ingest_file(path)
        except ConnectionError:
            parts.append("⚠️ Can't reach the local model (Ollama isn't running).")
            continue
        if not r["ok"]:
            parts.append(f"❌ {name}: {r['msg']}")
        elif r["chunks"] == 0:
            parts.append(f"📎 {name}: {r['msg']}")
        else:
            where = ", ".join(f"**{c}**" for c in r["categories"])
            parts.append(f"📎 **{name}** → read {r['pages']} page(s), saved "
                         f"{r['chunks']} chunk(s) to {where}. Ask me about it.")
    if text:
        try:
            parts.append(pipeline.chat(text, _text_history(history)))
        except ConnectionError:
            parts.append("⚠️ I can't reach the local model (Ollama isn't running).")

    reply = "\n\n".join(parts) if parts else "Send a message or attach a PDF/photo."
    _save_chats((history or []) + [
        {"role": "user", "content": shown or "📎 (file)"},
        {"role": "assistant", "content": reply}])
    return reply


def clear_chats():
    if os.path.exists(CHAT_FILE):
        os.remove(CHAT_FILE)


def do_ingest(text):
    r = pipeline.ingest(text, "")
    if not r["ok"]:
        return r["msg"], text
    return f'✅ Saved to **{r["category"]}** (memory #{r["id"]})', ""


def list_facts():
    """Everything the AI believes, grouped by expert — visible and deletable."""
    if not store.facts:
        return "_No memories yet._"
    lines = [f"**{len(store.facts)} memories** — this is everything the AI "
             "knows about you. Delete anything that's wrong.", ""]
    for cat in sorted({x["category"] for x in store.facts}):
        lines.append(f"#### {cat}")
        lines += [f"- `#{x['id']}` {x['text']}" +
                  (f" _(from {x['source']})_" if x.get("source") else "")
                  for x in store.facts_in(cat)]
        lines.append("")
    return "\n".join(lines)


def do_delete(fid):
    store.delete(int(fid or 0))
    return list_facts()


seed_if_empty()
# Load the HHEM verifier off the critical path so the first answer isn't slow.
threading.Thread(target=verify.warmup, daemon=True).start()

with gr.Blocks(title="On-Device AI", fill_height=True) as demo:
    gr.Markdown("### 🧠 Personal AI — private, on-device")

    with gr.Tab("Chat"):
        gr.ChatInterface(
            fn=respond,
            type="messages",
            multimodal=True,  # the input box takes a PDF/photo attachment too
            textbox=gr.MultimodalTextbox(
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp"],
                placeholder="Ask me anything — or attach a PDF/photo to remember it."),
            # callable -> re-read on every page load, so a refresh shows the
            # latest saved history, not the history as of app start
            chatbot=gr.Chatbot(value=load_chats, type="messages",
                               label="Chat", height=450),
            examples=[{"text": "What's my blood group?"},
                      {"text": "What's my wifi password?"},
                      {"text": "Am I allergic to anything?"},
                      {"text": "Tell me a joke"}],
            cache_examples=False,
        )

    with gr.Tab("Add memory"):
        t = gr.Textbox(label="A fact to remember", lines=2,
                       placeholder="e.g. My car service is due every October.")
        add_btn = gr.Button("Save", variant="primary")
        add_out = gr.Markdown()
        add_btn.click(do_ingest, t, [add_out, t])

    with gr.Tab("Memories"):
        # callable -> re-read per page load, not frozen at app start
        mem = gr.Markdown(list_facts)
        with gr.Row():
            del_id = gr.Number(label="Memory # to delete", precision=0)
            del_btn = gr.Button("Delete")
        gr.Button("Refresh").click(lambda: list_facts(), None, mem)
        del_btn.click(do_delete, del_id, mem)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

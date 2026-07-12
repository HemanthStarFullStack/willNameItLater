"""Chat UI for the on-device AI. A normal chat box — it decides per message
whether to pull from your saved memory (RAG) or just answer conversationally.
"""
import gradio as gr

import pipeline
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


def respond(message, history):
    return pipeline.chat(message, history)


def do_ingest(text):
    r = pipeline.ingest(text, "")
    if not r["ok"]:
        return r["msg"], text
    return f'✅ Saved to **{r["category"]}** (memory #{r["id"]})', ""


def stats():
    counts = store.categories()
    if not counts:
        return "_No memories yet._"
    total = sum(counts.values())
    lines = [f"**{total} memories saved**", ""]
    lines += [f"- {k} — {v}" for k, v in sorted(counts.items())]
    return "\n".join(lines)


seed_if_empty()

with gr.Blocks(title="On-Device AI", fill_height=True) as demo:
    gr.Markdown("### 🧠 Personal AI — private, on-device")

    with gr.Tab("Chat"):
        gr.ChatInterface(
            fn=respond,
            examples=["What's my blood group?", "What's my wifi password?",
                      "Am I allergic to anything?", "Tell me a joke",
                      "What's the capital of France?"],
            cache_examples=False,
        )

    with gr.Tab("Add memory"):
        t = gr.Textbox(label="A fact to remember", lines=2,
                       placeholder="e.g. My car service is due every October.")
        add_btn = gr.Button("Save", variant="primary")
        add_out = gr.Markdown()
        add_btn.click(do_ingest, t, [add_out, t])

    with gr.Tab("Memories"):
        mem = gr.Markdown(stats())
        gr.Button("Refresh").click(lambda: stats(), None, mem)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

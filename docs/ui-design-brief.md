# On-Device AI — UI Design Brief

## What it is
A fully private personal AI that lives on your phone. All models run on-device;
memories and chats are AES-encrypted locally and never leave the phone.
Positioning: **"One brain, many expert memories — a personal AI team on your phone."**
Differentiator: it shows its work and **never lies** — every answer carries a
visible trust trace (where the answer came from, how confident the match was).

## Tone / personality
Warm, private, trustworthy. A vault that talks. NOT a generic chatbot skin —
the trust/provenance UI is the hero. Dark mode first-class. Feels calm on a
cheap phone (works down to 2GB RAM devices).

## Screens (current app has all of these, redesign freely)

1. **Model picker (first run)**
   - Shows device RAM ("Your device: 6 GB RAM — 3 of 5 models fit, best one auto-selected")
   - Radio list of models: name, params, download size, quality tier, family
   - Models that don't fit are disabled with honest reason ("needs 6+ GB RAM")
   - Slow-but-better models get a 🐢 hint; recommended one gets ★ "best for you"
   - One big CTA: "Download & start · <model>" with progress % while downloading
   - Line: "Everything runs on-device; no account or token."
   - (Coming: optional "sign in to unlock accelerated models" tier — design a slot for it)

2. **Chat tab**
   - Streaming assistant bubbles (token-by-token)
   - Under each assistant reply, two artifacts:
     a. collapsed **"🧭 thought process"** expander — step list of what the brain
        did (routed to X memories → match 0.84 → verified grounded → saved)
     b. **footer badge line** — the route the answer took, e.g.:
        `🧠 Health memory · match 0.84` / `🌐 web search · no memory match (0.43)` /
        `💬 general reply · 2 note(s) as context` / `💾 saved to Work · updated` /
        `🗑️ memory` (forgot something)
   - A live status line above the input while generating (same trace, streaming)
   - Input placeholder: "Ask anything — it stays on this device"
   - Send button disabled while generating

3. **Add memory tab** — simple: text field + category auto-suggest, save confirmation showing which expert bucket it landed in

4. **Memories tab** (the trust page)
   - Facts grouped by expert category (Health, Money, Work, Home, Personal, Learning, Travel…)
   - Each fact: text, category chip, delete button
   - This page is the "see & correct what the AI believes" moat — make it feel like the user's vault, not a settings list

5. **States to design**
   - First-run download (with size, %), model loading spinner, generating,
     web-search in flight, error bubble (⚠️ prefix), empty memories, empty chat greeting

## Hard constraints
- Flutter Material 3, phone-first portrait
- Must remain legible/fast on low-end devices — no heavy blur/video
- Route badges use these concepts (icons can change): memory 🧠 / web 🌐 / chat 💬 / saved 💾 / forgot 🗑️
- Match scores (0.00–1.00) are shown — design them as a confidence element, not raw decimals if you have a better idea

## Attach with this brief
Screenshots of the current (functional, undesigned) UI: model picker, chat with
trace + footers, grounded answer. Use them as the "before".

/// The end-to-end brain — the port of pipeline.py. chat() splits compound
/// questions, routes each to the right expert memory, answers strictly from
/// notes behind a groundedness gate, falls back to web/general chat, and
/// quietly saves/updates/deletes personal facts. Every step records a trace so
/// the UI can show its work. (ask()/_crag from the Python file aren't ported:
/// the app UI never calls them.)
///
/// Groundedness gate: the desktop prototype's primary check is HHEM (a torch
/// NLI cross-encoder) — no torch on-device, so this uses pipeline.py's own
/// documented fallback: lexical coverage + an LLM judge for low-overlap cases.
/// Upgrade path: an ONNX NLI model via fonnx.
library;

import 'llm.dart';
import 'router.dart';
import 'store.dart';
import 'web.dart';

// --- config.py values -------------------------------------------------------
// Cosine gates calibrated for EmbeddingGemma-300M on the desktop harness
// (2026-07-17, test/harness/embedder_test.dart): on-topic queries scored
// ≥0.61 against their fact, off-topic questions peaked at 0.49 — so 0.55
// separates cleanly. Re-run that test whenever the embedder changes.
const kSearchK = 8;
const kTopK = 5;
const kRagThreshold = 0.55;
const kChatContextFloor = 0.42;

class ChatReply {
  final String answer;
  final String footer;
  final List<String> steps;
  const ChatReply(this.answer, this.footer, this.steps);
}

final _webHints = RegExp(
  r'\b(latest|current(ly)?|today|tonight|now|recent(ly)?|this (week|month|year)|'
  r'news|headline|price|prices|cost of|weather|forecast|score|scores|result|'
  r'results|who won|winner|standings|stock|shares|released?|launch(ed|ing)?|'
  r'update|version|20(2[4-9]|3\d))\b',
  caseSensitive: false,
);

final _stop = Set.of(
    ('a an the is are was were am i my me you your of to in on at for and or '
            'with that this it its as be been being have has had do does did '
            'what when where who why how which not no yes can will would '
            'should may might')
        .split(' '));

List<String> _contentWords(String s) => [
      for (final m in RegExp(r'[a-z0-9]+').allMatches(s.toLowerCase()))
        if (!_stop.contains(m.group(0))) m.group(0)!,
    ];

double _coverage(String answer, String context) {
  final words = _contentWords(answer);
  if (words.isEmpty) return 1.0;
  final ctx = _contentWords(context).toSet();
  return words.where(ctx.contains).length / words.length;
}

// --- duration math: decisions are code, the model only reads numbers out ----

const _months = {
  'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
  'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
};

final _range = RegExp(
  r'\b([a-z]{3})[a-z]*\.?\s+(\d{4})\s*(?:-|–|—|to)\s*'
  r'(?:([a-z]{3})[a-z]*\.?\s+(\d{4})|(present|current|now|today|ongoing))\b',
  caseSensitive: false,
);

String _span(int n) {
  final y = n ~/ 12, mo = n % 12;
  final parts = [
    if (y > 0) '$y year${y == 1 ? '' : 's'}',
    if (mo > 0) '$mo month${mo == 1 ? '' : 's'}',
  ];
  return parts.join(' ');
}

List<String> _durations(String text) {
  final out = <String>[];
  final today = DateTime.now();
  for (final m in _range.allMatches(text)) {
    final a = m.group(1)!.toLowerCase();
    if (!_months.containsKey(a)) continue;
    final ya = int.parse(m.group(2)!);
    final (int, int) end;
    if (m.group(5) != null) {
      end = (today.year, today.month);
    } else if (m.group(3) != null &&
        _months.containsKey(m.group(3)!.toLowerCase())) {
      end = (int.parse(m.group(4)!), _months[m.group(3)!.toLowerCase()]!);
    } else {
      continue;
    }
    final n = (end.$1 - ya) * 12 + (end.$2 - _months[a]!);
    if (n > 0 && n < 1200) {
      out.add('${m.group(0)} = $n months (${_span(n)})');
    }
  }
  return out.toSet().toList(); // same range in repeated notes -> once
}

String _fmt(List<Fact> hits) {
  var body = [
    for (var i = 0; i < hits.length; i++)
      '[${i + 1}] (${hits[i].category}) ${hits[i].text}',
  ].join('\n');
  final spans = _durations(body);
  if (spans.isNotEmpty) {
    body += '\n\nDurations already calculated for you — use these exact '
        'numbers, never recompute them:\n${spans.map((s) => '- $s').join('\n')}';
  }
  return body;
}

// --- regex gates (verbatim from pipeline.py) --------------------------------

const _greetings = {
  'hi', 'hii', 'hey', 'yo', 'hello', 'hello there', 'hey there', 'sup',
  'hola', 'howdy', 'good morning', 'good afternoon', 'good evening',
  'hi there', 'thanks', 'thank you', 'ok', 'okay',
};

final _factHint = RegExp(
  r"\b(my|mine|i am|i'm|im|i have|i've|i work|i live|i like|i love|i hate|"
  r'i prefer|i was|i use|i drive|i own|call me)\b',
  caseSensitive: false,
);
final _notFactStart = RegExp(
  r"^(what|whats|who|whos|whom|whose|when|whens|where|wheres|why|how|hows|"
  r'is|are|am|was|were|do|does|did|can|could|will|would|should|shall|may|'
  r'might|tell|show|give|list|search|find|explain|write|make|create|help|'
  r"please)('s)?\b",
  caseSensitive: false,
);
final _updateCue = RegExp(
  r'\b(now|no longer|not anymore|anymore|instead|changed?( to)?|new|'
  r'moved to|renamed|switched)\b',
  caseSensitive: false,
);
final _forget = RegExp(r'^(please\s+)?(forget|delete|remove|erase)\b',
    caseSensitive: false);
final _followup = RegExp(
  r'\b(he|she|her|hers|his|him|it|its|they|them|their|that|this|those)\b'
  r'|^(and|also|what about|how about)\b',
  caseSensitive: false,
);
final _otherPerson =
    RegExp("\\bmy\\s+(\\w+)['’`]s\\b", caseSensitive: false);
final _lookup = RegExp(
  r'^(what|whats|which|when|whens|where|wheres|who|whos|whose|'
  r'how (much|many|old|long)|am i|do i|does my|is my|are my|'
  r'tell me my|list my|show my)\b',
  caseSensitive: false,
);
final _advice = RegExp(
  r'\b(should|could|would|recommend|suggest|advice|advise|think|opinion|'
  r'idea|ideas|tips|help me|motivate|encourage|feel|feeling|worried|'
  r'nervous|excited|stressed|plan|prepare|next|better|improve)\b',
  caseSensitive: false,
);

class Pipeline {
  final Llm llm;
  final Router router;
  final MemoryStore store;

  /// Live progress line for the UI ("routed to Work memories"…).
  void Function(String step)? onStep;

  /// Streams the visible text of the final user-facing answer call.
  void Function(String partial)? onPartial;

  Pipeline(this.llm, this.router, this.store);

  // --- ingest (Add-memory tab) ----------------------------------------------

  Future<({bool ok, String category, int id, String msg})> ingest(String text,
      {String source = ''}) async {
    text = text.trim();
    if (text.isEmpty) {
      return (ok: false, category: '', id: 0, msg: 'Nothing to save.');
    }
    var category = await router.routeIngest(text);
    if (!store.categories().containsKey(category)) {
      // Don't let the classifier fragment the buckets ("Fitness" vs Health).
      category = await router.snapToExisting(text, category);
    }
    final id = await store.add(text, category, source: source);
    return (ok: true, category: category, id: id, msg: '');
  }

  // --- groundedness gate ----------------------------------------------------

  Future<({bool grounded, String reason})> _verify(
      String answer, String context) async {
    if (answer.toLowerCase().contains("don't have that in your data")) {
      return (grounded: true, reason: 'honest no-answer');
    }
    final covered = _coverage(answer, context);
    if (covered >= 0.6) {
      return (
        grounded: true,
        reason: 'lexical coverage ${(covered * 100).round()}%'
      );
    }
    // Low overlap -> the answer added new terms; let the model judge.
    final out = await llm.chatJson(
      'NOTES:\n$context\n\nANSWER:\n$answer\n\n'
      'Does every fact in ANSWER also appear in NOTES? Rephrasing is fine. '
      'Answer false ONLY if ANSWER states a fact that is nowhere in NOTES. '
      'JSON: {"grounded": true}',
      system:
          'Fact-check the answer against the notes. When unsure, say true. '
          'Output only JSON.',
    );
    final grounded = out['grounded'] is bool ? out['grounded'] as bool : true;
    return (
      grounded: grounded,
      reason: 'lexical coverage ${(covered * 100).round()}%'
    );
  }

  // --- remembering / forgetting ---------------------------------------------

  bool _looksLikeFact(String message) {
    if (message.contains('?')) return false;
    if (_notFactStart.hasMatch(message.trim())) return false;
    return _factHint.hasMatch(message);
  }

  /// Every lasting personal fact stated in the message, as short notes.
  Future<List<String>> _extractFacts(String message) async {
    final out = await llm.chatJson(
      'Message from the user: "$message"\n\n'
      'List every lasting personal fact the user states (name, health, work, '
      'home, preferences, possessions, relationships, plans) as short '
      'first-person notes, e.g. "My name is Hemanth.", "My manager is Raj." '
      'Keep WHO each fact is about exactly as the user said it. '
      'If it is a question, a request, small talk, or a temporary situation '
      '(feeling sick, running late), return an empty list.\n'
      'JSON: {"facts": ["...", "..."]}  or  {"facts": []}',
      system: 'You extract durable personal facts. Output only JSON.',
    );
    var raw = out['facts'];
    if (raw is String) raw = [raw];
    final clean = <String>[];
    for (final f in (raw is List ? raw : const [])) {
      if (f is! String) continue;
      // Small models sometimes echo placeholder tokens from the prompt.
      final s = f.trim().replaceFirst(RegExp(r'^\s*<[^>]{0,30}>\s*'), '');
      if (s.isNotEmpty &&
          !{'the note', 'null', 'none', '...'}.contains(s.toLowerCase())) {
        clean.add(s);
      }
      if (clean.length == 4) break;
    }

    // Guard against the extractor changing WHOSE fact it is ("my manager is
    // now Raj" -> "My name is Raj."): every fact must be made of words the
    // user actually said. If the model proposed facts but mangled them all,
    // save the raw message instead — retrieval handles natural phrasing fine.
    final ok = [
      for (final f in clean)
        if (_coverage(f, message) >= 0.6) f,
    ];
    // Also never DROP a correction: if the extractor returned nothing for an
    // update-cued statement ("my manager is NOW Raj" — greedy decode can flip
    // between runs), an old, now-wrong fact would silently survive. Saving the
    // raw message is always safer than keeping stale truth.
    if (ok.isEmpty && (clean.isNotEmpty || _updateCue.hasMatch(message))) {
      var msg = message.trim();
      while (msg.endsWith('.')) {
        msg = msg.substring(0, msg.length - 1);
      }
      return ['$msg.'];
    }
    return ok;
  }

  /// Add / update / replace one extracted fact. Returns a footer label.
  /// A stored fact is only ever REPLACED on an explicit update cue — an LLM
  /// judge was tried in the prototype and replaced "allergic to penicillin"
  /// with "allergic to peanuts"; a small model must not make delete decisions.
  Future<String?> _saveFact(String fact, String message) async {
    final (hits, score) = await store.search(fact, k: 1);

    // Same fact restated -> refresh it in place, no duplicate.
    if (hits.isNotEmpty && score >= 0.92) {
      await store.update(hits.first.id, fact);
      return '${hits.first.category} · updated';
    }

    // Explicit correction on the same topic -> merge into the old note.
    // Without a cue word, always add: a wrong add is a duplicate the user can
    // delete; a wrong replace silently loses data.
    if (hits.isNotEmpty && score >= 0.60 && _updateCue.hasMatch(message)) {
      final old = hits.first;
      String merged;
      try {
        merged = (await llm.chat(
          'OLD: "${old.text}"\nNEW: "$fact"\n\n'
          'NEW is the current truth and overrides OLD where they disagree. '
          'Write ONE short note stating only the CURRENT state: keep details '
          'from OLD that NEW does not contradict, drop what it contradicts, '
          'and never mention the change itself (no "now", "instead", '
          '"changed"). Output only the note.',
        ))
            .trim();
      } catch (_) {
        merged = fact; // merge failed -> keep the new statement as-is
      }
      merged = merged.replaceAll(RegExp(r'^"|"$'), '');
      final covered = _coverage(merged, '${old.text} $fact');
      // The merged note must actually carry the NEW information — a small
      // model sometimes rewrites OLD and drops the correction entirely
      // (kept "Priya" after "my manager is now Raj").
      final keepsNew = _coverage(fact, merged) >= 0.6;
      await store.update(old.id, covered >= 0.6 && keepsNew ? merged : fact);
      return '${old.category} · updated';
    }

    final r = await ingest(fact, source: 'chat');
    return r.ok ? r.category : null;
  }

  /// If the user just told us lasting personal facts, keep them all.
  Future<String?> _maybeRemember(String message) async {
    if (!_looksLikeFact(message)) return null;
    final labels = <String>[];
    for (final f in await _extractFacts(message)) {
      final lbl = await _saveFact(f, message);
      if (lbl != null) labels.add(lbl);
    }
    return labels.isEmpty ? null : labels.join(', ');
  }

  /// Natural-language delete: "forget my wifi password".
  Future<String?> _maybeForget(String message, List<String> steps) async {
    if (!_forget.hasMatch(message)) return null;
    final (hits, score) = await store.search(message, k: 1);
    if (hits.isEmpty || score < 0.55) {
      return "I couldn't find a saved memory matching that.";
    }
    final h = hits.first;
    await store.delete(h.id);
    steps.add('🗑️ deleted memory #${h.id} from ${h.category}');
    return 'Forgot: "${h.text}"';
  }

  // --- question shaping ------------------------------------------------------

  String _recent(List<({String role, String content})> history, [int n = 4]) {
    final tail =
        history.length <= n ? history : history.sublist(history.length - n);
    return [
      for (final h in tail)
        '${h.role[0].toUpperCase()}${h.role.substring(1)}: ${h.content}',
    ].join('\n');
  }

  /// Make a follow-up self-contained ("what time is her standup?" ->
  /// "what time is Priya's standup?") so retrieval finds the right note.
  Future<String> _rewriteFollowup(String message,
      List<({String role, String content})> history, List<String> steps) async {
    if (history.isEmpty || !_followup.hasMatch(message)) return message;
    final convo = _recent(history);
    String out;
    try {
      out = (await llm.chat(
        'Conversation:\n$convo\n\nLatest user message: "$message"\n\n'
        'Rewrite the latest message as ONE standalone question, replacing '
        'pronouns with what they refer to. Output only the question.',
      ))
          .trim()
          .replaceAll(RegExp(r'^"|"$'), '');
    } catch (_) {
      return message; // rewrite is an optimisation, never a hard dependency
    }
    if (out.isNotEmpty && out.length < 200) {
      steps.add('rewrote follow-up → $out');
      return out;
    }
    return message;
  }

  /// Split a compound message into separately-answerable questions.
  /// Regex, not an LLM: the prototype's LLM splitter flip-flopped between
  /// identical runs and silently disabled the feature.
  List<String> _splitQuestions(String message) {
    var body = message.trim();
    while (body.endsWith('?')) {
      body = body.substring(0, body.length - 1);
    }
    if (body.contains('?')) {
      final parts = [
        for (final p in message.split('?'))
          if (p.trim().isNotEmpty) p.trim(),
      ];
      if (parts.length >= 2) {
        return [for (final p in parts.take(3)) '$p?'];
      }
    }
    // "A and B" splits only when the message reads as a question — statements
    // ("I drive a Honda and live in Hyderabad") go to the fact extractor.
    if (!_notFactStart.hasMatch(message.trim())) return const [];
    final parts = message.split(RegExp(r',?\s+and\s+', caseSensitive: false));
    if (parts.length == 2 &&
        parts.every((p) => p.trim().split(RegExp(r'\s+')).length >= 2)) {
      return [for (final p in parts) p.trim()];
    }
    return const [];
  }

  /// Merge per-question answers into one natural reply, without letting the
  /// merge step invent anything.
  Future<String> _combine(String message, List<String> parts) async {
    final joined = parts.join(' ');
    final String out;
    try {
      out = await llm.chat(
        'QUESTION: $message\n\nFACTS:\n${parts.map((p) => '- $p').join('\n')}'
        '\n\nWrite one short, natural reply addressed to the user (say '
        '"your", not "my") answering the question using ONLY these facts. '
        'Do not add anything new.',
        system: 'You merge partial answers into one reply. Never add facts.',
      );
    } catch (_) {
      return joined; // the per-question answers are already usable
    }
    return _coverage(out, joined) >= 0.6 ? out : joined;
  }

  /// Drop notes about OTHER people unless the question mentions that person.
  /// Embeddings can't split "my name" from "my mom's name"; a relation word in
  /// the note but not in the question = wrong entity.
  List<Fact> _entityFilter(String query, List<Fact> hits) {
    final q = query.toLowerCase().replaceAll(RegExp(r'[^a-z ]'), '');
    final out = <Fact>[];
    for (final h in hits) {
      final m = _otherPerson.firstMatch(h.text);
      if (m != null && !q.contains(m.group(1)!.toLowerCase())) continue;
      out.add(h);
    }
    return out.isEmpty ? hits : out;
  }

  bool _isLookup(String message) =>
      _lookup.hasMatch(message.trim()) && !_advice.hasMatch(message);

  // Personal fact statements share words with the live-info cues ("my manager
  // is NOW Raj") — they must be remembered, never sent to the web.
  bool _needsWeb(String message) =>
      _webHints.hasMatch(message) && !_looksLikeFact(message);

  // --- answering -------------------------------------------------------------

  Future<(String, String)> _answerOne(String message,
      List<({String role, String content})> history, List<String> steps) async {
    final lookup = _isLookup(message);
    var expert = await router.routeQuery(message);
    _step(steps, 'routed to ${expert ?? 'all'} memories');
    var (hits, score) =
        await store.search(message, category: expert, k: kTopK);
    if (expert != null && score < kRagThreshold) {
      // Routed to the wrong expert? Check all memories before giving up.
      expert = null;
      (hits, score) = await store.search(message, k: kTopK);
      _step(steps,
          'expert scored low → re-searched all (${score.toStringAsFixed(2)})');
    }

    if (lookup && score >= kRagThreshold && hits.isNotEmpty) {
      // Only near-top notes reach the model: a small answerer given loosely
      // related notes quotes the wrong one (asked passport, said the name).
      hits = [for (final h in hits) if (h.cos >= score - 0.12) h];
      hits = _entityFilter(message, hits);
      _step(
          steps,
          'match ${score.toStringAsFixed(2)} ≥ $kRagThreshold → memory path, '
          'kept ${hits.length} note(s) after dropping distractors');
      final context = _fmt(hits);
      var answer = await llm.chat(
        'NOTES about the user:\n$context\n\n'
        'QUESTION: $message\n\n'
        'Answer in ONE short, natural sentence addressed to the user — '
        'start with "Your" or "You" where it fits, never with "My". '
        'State the actual value from the NOTES; never just repeat the '
        'question, and never copy a note word-for-word as the whole reply. '
        'Use a note even if its wording differs. Do not add anything not '
        'asked; no reference numbers like [1] or category tags. Never do '
        'arithmetic: use durations exactly as listed and never add them '
        'together or work out a new figure. If the '
        'NOTES do not actually contain the answer, reply exactly: "I don\'t '
        'have that in your data yet." — never substitute a different fact.',
        system: "You answer strictly from the user's notes. Never invent facts.",
        onPartial: onPartial,
      );
      final verdict = await _verify(answer, context);
      _step(
          steps,
          'verify: ${verdict.grounded ? '✔ grounded' : '✘ unverified'} '
          '(${verdict.reason})');
      if (!verdict.grounded) {
        // Don't state a flagged claim and apologise under it — "never lies"
        // is the point. Withhold, and show what IS known.
        final spans = _durations(context);
        answer = "I can't back that up from your notes, so I won't guess."
            '${spans.isNotEmpty ? '\n\nWhat I do have:\n${spans.map((s) => '- $s').join('\n')}' : ''}'
            '\n\n(unverified draft: $answer)';
      }
      final which = expert ?? hits.first.category;
      return (answer, '🧠 $which memory · match ${score.toStringAsFixed(2)}');
    }

    if (_needsWeb(message)) {
      _step(steps, 'live-info cue → web search');
      final (answer, _) = await _chatOrSearch(message, history, const []);
      return (
        answer,
        '🌐 web search · no memory match (${score.toStringAsFixed(2)})'
      );
    }

    // Conversational reply. The notes ride along as background so it can talk
    // about the user's actual life. No groundedness gate here on purpose —
    // encouragement isn't extractive, the gate would flag every friendly
    // sentence.
    final notes = [for (final h in hits) if (h.cos >= kChatContextFloor) h];
    final why = !lookup
        ? 'not a value lookup'
        : 'no memory match (${score.toStringAsFixed(2)})';
    _step(
        steps,
        '$why → conversational reply'
        '${notes.isNotEmpty ? ', ${notes.length} note(s) as background' : ''}');
    final (answer, _) = await _chatOrSearch(message, history, notes);
    final tag = notes.isNotEmpty ? ' · ${notes.length} note(s) as context' : '';
    return (answer, '💬 general reply$tag');
  }

  Future<(String, bool)> _chatOrSearch(String message,
      List<({String role, String content})> history, List<Fact> notes) async {
    if (!_needsWeb(message)) {
      final convo = _recent(history);
      final known = notes.isNotEmpty
          ? 'What you already know about them (use only what fits, '
              'ignore the rest):\n${_fmt(notes)}\n\n'
          : '';
      final answer = await llm.chat(
        '$known$convo\nUser: $message\nAssistant:',
        system:
            "You are the user's personal assistant and you know them well. "
            'Talk like a warm, straight-talking friend — natural sentences, '
            'no lists, no reference numbers. Draw on what you know about them '
            'when it genuinely helps: to encourage them, give advice that '
            'fits their life, or connect things. Never recite their details '
            'back at them, and never mention facts that aren\'t relevant. '
            'Never do arithmetic: use any duration exactly as given and '
            'never add durations together. If they correct you, accept it '
            'and reply afresh — never repeat your previous answer. Two to '
            'four sentences.',
        temperature: 0.7, // conversation, not extraction
        onPartial: onPartial,
      );
      return (answer, false);
    }

    final results = await searchWeb(message, k: 4);
    if (results.isEmpty) {
      return (
        "I tried to search the web but couldn't reach it right now.",
        true
      );
    }
    var answer = await llm.chat(
      'Web results:\n${formatResults(results)}\n\n'
      'Question: $message\n\n'
      'Answer the question using the web results above, briefly. State actual '
      'values from the results; never write placeholders like [insert ...].',
      system: 'You answer from the provided web results. Be concise and factual.',
      onPartial: onPartial,
    );
    answer += '\n\nSources: ${[
      for (var i = 0; i < results.length; i++) '[${i + 1}] ${results[i].url}',
    ].join(', ')}';
    return (answer, true);
  }

  void _step(List<String> steps, String s) {
    steps.add(s);
    onStep?.call(s);
  }

  // --- entry point ------------------------------------------------------------

  Future<ChatReply> chat(String message,
      List<({String role, String content})> history) async {
    message = message.trim();
    if (message.isEmpty) return const ChatReply('Ask me something.', '', []);

    final steps = <String>[];
    final forgot = await _maybeForget(message, steps);
    if (forgot != null) {
      return ChatReply(forgot, '🗑️ memory', steps);
    }

    // Pure greetings carry no data intent — answer instantly instead of
    // running retrieval (and tripping the groundedness gate).
    final bare = message.toLowerCase().replaceAll(RegExp(r'[^a-z ]'), '').trim();
    if (_greetings.contains(bare)) {
      return const ChatReply(
          'Hi! Ask me anything about your saved data.', '💬 greeting', []);
    }

    final question = await _rewriteFollowup(message, history, steps);
    final subs = _splitQuestions(question);
    String answer, footer;
    if (subs.isNotEmpty) {
      _step(steps, 'split into ${subs.length} questions: ${subs.join(' / ')}');
      final parts = <String>[], tags = <String>[];
      for (final q in subs) {
        final (a, tag) = await _answerOne(q, history, steps);
        parts.add(a);
        tags.add(tag);
      }
      answer = await _combine(question, parts);
      _step(steps, 'combined part answers into one reply');
      footer = tags.join(' | ');
    } else {
      final (a, tag) = await _answerOne(question, history, steps);
      answer = a;
      footer = tag;
    }

    // The answer above is already final — a failure while saving facts must
    // never surface as an error and destroy it.
    try {
      final saved = await _maybeRemember(message);
      if (saved != null) {
        footer += ' · 💾 saved to $saved';
        _step(steps, '💾 new fact detected → saved to $saved');
      }
    } catch (_) {}

    return ChatReply(answer, footer, steps);
  }
}

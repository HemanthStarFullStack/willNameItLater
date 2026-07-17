import 'dart:convert';
import 'dart:io';

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:system_info_plus/system_info_plus.dart';

import 'brain/embedder.dart';
import 'brain/llm.dart';
import 'brain/pipeline.dart';
import 'brain/router.dart' as brain;
import 'brain/store.dart';

/// One on-device model. `minRamGB` = the device RAM floor to run it at all;
/// `quality` is a hand-curated 0–100 benchmark proxy used to rank models that
/// fit — like the desktop "which LLM fits" tools, scaled to phones.
/// Current-generation (2025-26) models only; every URL is an open GGUF
/// (official/ggml-org/unsloth mirrors, verified 302-public) — no HuggingFace
/// token. `thinks` marks models that emit <think>…</think> reasoning blocks.
class Model {
  final String name, family, file, url;
  final double params, sizeGB, minRamGB;
  final int quality, year;
  final bool thinks;
  const Model(this.name, this.family, this.params, this.sizeGB, this.minRamGB,
      this.quality, this.year, this.file, this.url,
      {this.thinks = false});
}

const _catalog = <Model>[
  Model('Qwen3 0.6B', 'Qwen3', 0.6, 0.5, 0, 44, 2025,
      'Qwen3-0.6B-Q4_K_M.gguf',
      'https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf',
      thinks: true),
  Model('Gemma 3 1B', 'Gemma', 1.0, 0.8, 2.5, 52, 2025,
      'gemma-3-1b-it-Q4_K_M.gguf',
      'https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf'),
  Model('DeepSeek R1 1.5B', 'DeepSeek', 1.5, 1.1, 3.0, 58, 2025,
      'DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf',
      'https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf',
      thinks: true),
  Model('Qwen3 1.7B', 'Qwen3', 1.7, 1.3, 3.0, 63, 2025,
      'Qwen3-1.7B-Q4_K_M.gguf',
      'https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf',
      thinks: true),
  Model('SmolLM3 3B', 'SmolLM3', 3.0, 1.9, 5.0, 72, 2025,
      'SmolLM3-3B-Q4_K_M.gguf',
      'https://huggingface.co/unsloth/SmolLM3-3B-GGUF/resolve/main/SmolLM3-3B-Q4_K_M.gguf'),
  Model('Gemma 3 4B', 'Gemma', 4.0, 2.5, 6.0, 79, 2025,
      'gemma-3-4b-it-Q4_K_M.gguf',
      'https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf'),
  Model('Phi-4 mini', 'Phi', 3.8, 2.5, 6.0, 80, 2025,
      'Phi-4-mini-instruct-Q4_K_M.gguf',
      'https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q4_K_M.gguf'),
  Model('Qwen3 4B (2507)', 'Qwen3', 4.0, 2.5, 6.0, 84, 2025,
      'Qwen3-4B-Instruct-2507-Q4_K_M.gguf',
      'https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf'),
];

/// The on-device embedder (MiniLM-L6-v2 ONNX) — powers hybrid retrieval and
/// bucket routing. Ships in the fonnx repo; one-time ~20 MB download.
const _embedderUrl =
    'https://github.com/Telosnex/fonnx/raw/main/example/assets/models/miniLmL6V2/miniLmL6V2.onnx';
const _embedderFile = 'miniLmL6V2.onnx';

int _fit(Model m, double ramGB) =>
    ramGB >= m.minRamGB * 1.4 ? 2 : (ramGB >= m.minRamGB ? 1 : 0);

/// CPU decode speed falls roughly with parameter count, and one chat turn
/// runs several generations (route/answer/verify/save). Models above ~2B are
/// answers-in-minutes on a phone — real quality, unusable latency.
bool _fastOnPhone(Model m) => m.params <= 2.0;

List<Model> _ranked(double ramGB) {
  final list = [..._catalog];
  list.sort((a, b) {
    final fa = _fit(a, ramGB) > 0 ? 1 : 0;
    final fb = _fit(b, ramGB) > 0 ? 1 : 0;
    if (fa != fb) return fb - fa;
    final sa = _fastOnPhone(a) ? 1 : 0;
    final sb = _fastOnPhone(b) ? 1 : 0;
    if (sa != sb) return sb - sa; // responsive models first
    return b.quality.compareTo(a.quality);
  });
  return list;
}

Model _best(double ramGB) => _ranked(ramGB).firstWhere(
      (m) => _fit(m, ramGB) > 0,
      orElse: () => _catalog.first,
    );

String _tier(int q) =>
    q >= 75 ? 'excellent' : (q >= 62 ? 'strong' : (q >= 48 ? 'good' : 'basic'));

/// A few sample memories so the app is queryable on first open (app.py).
const _samples = [
  ('My blood group is O positive.', 'Health'),
  ("I'm allergic to penicillin.", 'Health'),
  ('My health insurance policy is HLT-889241, renews every March.', 'Money'),
  ('Home wifi password is bluefalcon77.', 'Home'),
  ('My manager is Priya; standup is 10am daily on weekdays.', 'Work'),
  ('Passport number X1234567, expires in 2029.', 'Personal'),
  ("I'm learning Flutter and Dart this year.", 'Learning'),
];

void main() => runApp(const App());

class App extends StatelessWidget {
  const App({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'On-Device AI',
        theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
        home: const Home(),
      );
}

class Msg {
  final String role;
  String content;
  String footer;
  List<String> steps;
  Msg(this.role, this.content, {this.footer = '', this.steps = const []});

  Map<String, dynamic> toJson() =>
      {'role': role, 'content': content, 'footer': footer, 'steps': steps};
  static Msg fromJson(Map<String, dynamic> j) => Msg(
        j['role'] as String,
        j['content'] as String,
        footer: (j['footer'] ?? '') as String,
        steps: [for (final s in (j['steps'] ?? const [])) s as String],
      );
}

class Home extends StatefulWidget {
  const Home({super.key});

  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  // Brain
  final _llm = Llm();
  final _embedder = Embedder();
  MemoryStore? _store;
  Pipeline? _pipeline;

  // Setup state
  Model? _selected;
  double? _ramGB;
  double? _progress;
  String? _status;
  bool _booted = false;

  // Chat state
  final _messages = <Msg>[];
  final _input = TextEditingController();
  final _scroll = ScrollController();
  bool _generating = false;
  String _liveStep = '';
  late String _dataDir;

  @override
  void initState() {
    super.initState();
    _restore();
  }

  Future<String> _path(String file) async => '$_dataDir/$file';

  Future<void> _restore() async {
    _dataDir = (await getApplicationDocumentsDirectory()).path;
    final mb = await SystemInfoPlus.physicalMemory ?? 0;
    final ramGB = mb / 1024.0;
    final best = _best(ramGB);
    setState(() {
      _ramGB = ramGB > 0 ? ramGB : null;
      _selected = best;
    });
    if (File(await _path(best.file)).existsSync() &&
        File(await _path(_embedderFile)).existsSync()) {
      await _boot(best);
    }
  }

  Future<void> _download() async {
    final m = _selected;
    if (m == null) return;
    try {
      // Embedder first (small); then the chat model. .part-then-rename so an
      // interrupted download is never mistaken for a complete file.
      final ePath = await _path(_embedderFile);
      if (!File(ePath).existsSync()) {
        setState(() {
          _progress = 0;
          _status = 'Downloading memory engine (~20 MB)…';
        });
        await Dio().download(_embedderUrl, '$ePath.part',
            onReceiveProgress: (r, t) {
          if (t > 0) setState(() => _progress = r / t);
        });
        await File('$ePath.part').rename(ePath);
      }
      final gPath = await _path(m.file);
      if (!File(gPath).existsSync()) {
        setState(() {
          _progress = 0;
          _status = 'Downloading ${m.name} '
              '(${m.sizeGB.toStringAsFixed(1)} GB)… first time only.';
        });
        await Dio().download(m.url, '$gPath.part',
            onReceiveProgress: (r, t) {
          if (t > 0) setState(() => _progress = r / t);
        });
        await File('$gPath.part').rename(gPath);
      }
      setState(() => _status = 'Preparing your memories…');
      await _boot(m);
    } catch (e) {
      setState(() => _status = 'Download failed: $e');
    } finally {
      setState(() => _progress = null);
    }
  }

  Future<void> _boot(Model m) async {
    _embedder.load(await _path(_embedderFile));
    await _llm.load(await _path(m.file));
    _llm.noThinkSuffix =
        (m.family == 'Qwen3' && m.thinks) ? ' /no_think' : '';

    final store = MemoryStore(_embedder, await _path('memories.json'));
    await store.load();
    if (store.facts.isEmpty) {
      for (final (text, cat) in _samples) {
        await store.add(text, cat);
      }
    }
    final router = brain.Router(_llm, _embedder, store);
    final pipeline = Pipeline(_llm, router, store);
    pipeline.onStep = (s) => setState(() => _liveStep = s);
    pipeline.onPartial = (partial) => setState(() {
          if (_messages.isNotEmpty && _messages.last.role == 'assistant') {
            _messages.last.content = partial;
          }
        });

    // Restore chat history (last 200 messages survive restarts).
    try {
      final f = File(await _path('chats.json'));
      if (f.existsSync()) {
        final raw = json.decode(await f.readAsString()) as List;
        _messages.addAll(
            raw.map((e) => Msg.fromJson(e as Map<String, dynamic>)));
      }
    } catch (_) {}

    setState(() {
      _store = store;
      _pipeline = pipeline;
      _booted = true;
      _status = null;
    });
  }

  Future<void> _saveChats() async {
    final tail = _messages.length <= 200
        ? _messages
        : _messages.sublist(_messages.length - 200);
    await File(await _path('chats.json'))
        .writeAsString(json.encode([for (final m in tail) m.toJson()]));
  }

  Future<void> _send() async {
    final text = _input.text.trim();
    final pipeline = _pipeline;
    if (text.isEmpty || pipeline == null || _generating) return;
    _input.clear();
    final history = [
      for (final m in _messages) (role: m.role, content: m.content),
    ];
    setState(() {
      _messages.add(Msg('user', text));
      _messages.add(Msg('assistant', ''));
      _generating = true;
      _liveStep = 'thinking…';
    });
    try {
      final reply = await pipeline.chat(text, history);
      setState(() {
        _messages.last
          ..content = reply.answer
          ..footer = reply.footer
          ..steps = reply.steps;
      });
    } catch (e) {
      // Never wipe an answer that already streamed in — flag it instead.
      setState(() {
        final partial = _messages.last.content;
        if (partial.isEmpty || partial == '💭 thinking…') {
          _messages.last.content = '⚠️ $e';
        } else {
          _messages.last.footer = '⚠️ interrupted: $e';
        }
      });
    } finally {
      setState(() {
        _generating = false;
        _liveStep = '';
      });
      await _saveChats();
      if (_scroll.hasClients) {
        _scroll.animateTo(_scroll.position.maxScrollExtent + 200,
            duration: const Duration(milliseconds: 300), curve: Curves.easeOut);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!_booted) return Scaffold(body: SafeArea(child: _setup()));
    return DefaultTabController(
      length: 3,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('🧠 On-Device AI'),
          actions: [
            IconButton(
              tooltip: 'Switch model',
              icon: const Icon(Icons.tune),
              onPressed:
                  _generating ? null : () => setState(() => _booted = false),
            ),
          ],
          bottom: const TabBar(tabs: [
            Tab(text: 'Chat'),
            Tab(text: 'Add memory'),
            Tab(text: 'Memories'),
          ]),
        ),
        body: TabBarView(children: [_chatTab(), _addTab(), _memoriesTab()]),
      ),
    );
  }

  // --- setup (model picker) --------------------------------------------------

  Widget _setup() {
    final ram = _ramGB;
    if (ram == null) return const Center(child: Text('Scanning your device…'));
    final ranked = _ranked(ram);
    final fitCount = _catalog.where((m) => _fit(m, ram) > 0).length;
    final best = _best(ram);
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text('Your device: ${ram.toStringAsFixed(1)} GB RAM — '
              '$fitCount of ${_catalog.length} models fit, best one '
              'auto-selected. Everything runs on-device; no account or token.'),
          const SizedBox(height: 8),
          Expanded(
            child: ListView(
              children: [for (final m in ranked) _modelTile(m, ram, best)],
            ),
          ),
          const SizedBox(height: 8),
          if (_progress != null) ...[
            LinearProgressIndicator(value: _progress),
            const SizedBox(height: 6),
            Text('${((_progress ?? 0) * 100).toStringAsFixed(0)}%'),
          ] else
            FilledButton(
              onPressed: _selected == null ? null : _download,
              child: Text(_selected == null
                  ? 'No model fits this device'
                  : (_booted
                          ? 'Switch to'
                          : 'Download & start') +
                      '  ·  ${_selected!.name}'),
            ),
          if (_status != null) ...[
            const SizedBox(height: 10),
            Text(_status!, style: const TextStyle(color: Colors.orange)),
          ],
        ],
      ),
    );
  }

  Widget _modelTile(Model m, double ram, Model best) {
    final fit = _fit(m, ram);
    final badge = fit == 2 ? '🟢' : (fit == 1 ? '🟡' : '🔴');
    if (fit == 0) {
      return ListTile(
        enabled: false,
        leading: Text(badge, style: const TextStyle(fontSize: 18)),
        title: Text('${m.name}  ·  ${m.params}B'),
        subtitle: Text('needs ${m.minRamGB.toStringAsFixed(0)}+ GB RAM'),
      );
    }
    final rec = identical(m, best);
    return RadioListTile<String>(
      value: m.file,
      groupValue: _selected?.file,
      onChanged:
          _progress != null ? null : (_) => setState(() => _selected = m),
      secondary: Text(badge, style: const TextStyle(fontSize: 18)),
      title: Row(children: [
        Flexible(child: Text(m.name)),
        if (rec)
          const Padding(
            padding: EdgeInsets.only(left: 8),
            child: Text('★ best for you',
                style: TextStyle(fontSize: 12, color: Colors.indigo)),
          ),
      ]),
      subtitle: Text('${m.params}B · ${m.sizeGB.toStringAsFixed(1)} GB · '
          '${_tier(m.quality)} · ${m.family}'
          '${_fastOnPhone(m) ? '' : ' · 🐢 smarter but SLOW on a phone'}'),
    );
  }

  // --- Chat tab ----------------------------------------------------------------

  Widget _chatTab() {
    return Column(
      children: [
        Expanded(
          child: ListView.builder(
            controller: _scroll,
            padding: const EdgeInsets.all(12),
            itemCount: _messages.length,
            itemBuilder: (_, i) => _bubble(_messages[i]),
          ),
        ),
        if (_generating && _liveStep.isNotEmpty)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
            child: Row(children: [
              const SizedBox(
                  width: 14,
                  height: 14,
                  child: CircularProgressIndicator(strokeWidth: 2)),
              const SizedBox(width: 8),
              Expanded(
                child: Text('🧭 $_liveStep',
                    style: const TextStyle(fontSize: 12, color: Colors.grey),
                    overflow: TextOverflow.ellipsis),
              ),
            ]),
          ),
        SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(12, 4, 12, 12),
            child: Row(children: [
              Expanded(
                child: TextField(
                  controller: _input,
                  onSubmitted: (_) => _send(),
                  decoration: const InputDecoration(
                    hintText: 'Ask anything — it stays on this device',
                    border: OutlineInputBorder(),
                  ),
                ),
              ),
              const SizedBox(width: 8),
              IconButton.filled(
                onPressed: _generating ? null : _send,
                icon: const Icon(Icons.arrow_upward),
              ),
            ]),
          ),
        ),
      ],
    );
  }

  Widget _bubble(Msg m) {
    final user = m.role == 'user';
    return Align(
      alignment: user ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.all(12),
        constraints: const BoxConstraints(maxWidth: 520),
        decoration: BoxDecoration(
          color: user
              ? Theme.of(context).colorScheme.primaryContainer
              : Theme.of(context).colorScheme.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(14),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(m.content.isEmpty ? '…' : m.content),
            if (m.steps.isNotEmpty)
              Theme(
                data: Theme.of(context)
                    .copyWith(dividerColor: Colors.transparent),
                child: ExpansionTile(
                  tilePadding: EdgeInsets.zero,
                  dense: true,
                  title: const Text('🧭 thought process',
                      style: TextStyle(fontSize: 12, color: Colors.grey)),
                  children: [
                    for (var i = 0; i < m.steps.length; i++)
                      Align(
                        alignment: Alignment.centerLeft,
                        child: Text('${i + 1}. ${m.steps[i]}',
                            style: const TextStyle(
                                fontSize: 11, color: Colors.grey)),
                      ),
                  ],
                ),
              ),
            if (m.footer.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Text(m.footer,
                    style: const TextStyle(fontSize: 11, color: Colors.grey)),
              ),
          ],
        ),
      ),
    );
  }

  // --- Add memory tab ------------------------------------------------------------

  final _addCtl = TextEditingController();
  String _addResult = '';
  bool _adding = false;

  Widget _addTab() {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          TextField(
            controller: _addCtl,
            maxLines: 2,
            decoration: const InputDecoration(
              labelText: 'A fact to remember',
              hintText: 'e.g. My car service is due every October.',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 12),
          FilledButton(
            onPressed: _adding
                ? null
                : () async {
                    final t = _addCtl.text.trim();
                    if (t.isEmpty || _pipeline == null) return;
                    setState(() {
                      _adding = true;
                      _addResult = 'Saving…';
                    });
                    final r = await _pipeline!.ingest(t);
                    setState(() {
                      _adding = false;
                      _addResult = r.ok
                          ? '✅ Saved to ${r.category} (memory #${r.id})'
                          : r.msg;
                      if (r.ok) _addCtl.clear();
                    });
                  },
            child: const Text('Save'),
          ),
          const SizedBox(height: 12),
          Text(_addResult),
        ],
      ),
    );
  }

  // --- Memories tab ----------------------------------------------------------------

  Widget _memoriesTab() {
    final store = _store;
    if (store == null || store.facts.isEmpty) {
      return const Center(child: Text('No memories yet.'));
    }
    final cats = store.categories().keys.toList()..sort();
    return ListView(
      padding: const EdgeInsets.all(12),
      children: [
        Text(
          '${store.facts.length} memories — this is everything the AI knows '
          'about you. Delete anything that\'s wrong.',
          style: const TextStyle(fontSize: 13, color: Colors.grey),
        ),
        for (final cat in cats) ...[
          Padding(
            padding: const EdgeInsets.only(top: 12, bottom: 4),
            child: Text(cat,
                style:
                    const TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
          ),
          for (final f in store.factsIn(cat))
            ListTile(
              dense: true,
              contentPadding: EdgeInsets.zero,
              title: Text(f.text),
              subtitle: f.source.isNotEmpty
                  ? Text('from ${f.source}',
                      style: const TextStyle(fontSize: 11))
                  : null,
              trailing: IconButton(
                icon: const Icon(Icons.delete_outline, size: 20),
                onPressed: () async {
                  await store.delete(f.id);
                  setState(() {});
                },
              ),
            ),
        ],
      ],
    );
  }
}

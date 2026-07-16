import 'dart:io';

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:lib_llama_cpp/lib_llama_cpp.dart';
import 'package:path_provider/path_provider.dart';
import 'package:system_info_plus/system_info_plus.dart';

/// One on-device model. Everything the picker needs to score it against a
/// phone. `minRamGB` = the device RAM floor to run it at all; `quality` is a
/// hand-curated 0–100 benchmark proxy (small-model instruct reputation), used
/// only to *rank* models that fit — like the desktop "which LLM fits" tools,
/// scaled down to what actually runs on a phone. All URLs are open GGUF (Qwen
/// official or bartowski mirrors) — no HuggingFace token.
class Model {
  final String name, family, file, url;
  final double params, sizeGB, minRamGB;
  final int quality, year;
  const Model(this.name, this.family, this.params, this.sizeGB, this.minRamGB,
      this.quality, this.year, this.file, this.url);
}

const _catalog = <Model>[
  Model('SmolLM2 360M', 'SmolLM2', 0.36, 0.3, 1.5, 34, 2024,
      'SmolLM2-360M-Instruct-Q4_K_M.gguf',
      'https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf'),
  Model('Qwen2.5 0.5B', 'Qwen2.5', 0.5, 0.4, 2.0, 44, 2024,
      'qwen2.5-0.5b-instruct-q4_k_m.gguf',
      'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf'),
  Model('Llama 3.2 1B', 'Llama', 1.0, 0.8, 3.0, 52, 2024,
      'Llama-3.2-1B-Instruct-Q4_K_M.gguf',
      'https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf'),
  Model('Qwen2.5 1.5B', 'Qwen2.5', 1.5, 1.0, 3.0, 62, 2024,
      'qwen2.5-1.5b-instruct-q4_k_m.gguf',
      'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf'),
  Model('SmolLM2 1.7B', 'SmolLM2', 1.7, 1.1, 3.0, 56, 2024,
      'SmolLM2-1.7B-Instruct-Q4_K_M.gguf',
      'https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf'),
  Model('Gemma 2 2B', 'Gemma', 2.6, 1.7, 4.0, 67, 2024,
      'gemma-2-2b-it-Q4_K_M.gguf',
      'https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf'),
  Model('Llama 3.2 3B', 'Llama', 3.0, 2.0, 6.0, 72, 2024,
      'Llama-3.2-3B-Instruct-Q4_K_M.gguf',
      'https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf'),
  Model('Qwen2.5 3B', 'Qwen2.5', 3.0, 2.0, 6.0, 74, 2024,
      'qwen2.5-3b-instruct-q4_k_m.gguf',
      'https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf'),
  Model('Phi-3.5 mini', 'Phi', 3.8, 2.3, 6.0, 78, 2024,
      'Phi-3.5-mini-instruct-Q4_K_M.gguf',
      'https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf'),
];

// Fit against a device's RAM: 2 = comfortable, 1 = tight, 0 = won't run.
int _fit(Model m, double ramGB) =>
    ramGB >= m.minRamGB * 1.4 ? 2 : (ramGB >= m.minRamGB ? 1 : 0);

// Whole catalog ranked for this device: models that fit first (best quality on
// top), then the ones that don't — kept visible so you can see what a bigger
// phone would unlock, exactly like the desktop hardware-scan tools.
List<Model> _ranked(double ramGB) {
  final list = [..._catalog];
  list.sort((a, b) {
    final fa = _fit(a, ramGB) > 0 ? 1 : 0;
    final fb = _fit(b, ramGB) > 0 ? 1 : 0;
    if (fa != fb) return fb - fa;
    return b.quality.compareTo(a.quality);
  });
  return list;
}

// The single best model this device can run (highest quality that fits).
Model _best(double ramGB) => _ranked(ramGB).firstWhere(
      (m) => _fit(m, ramGB) > 0,
      orElse: () => _catalog.first,
    );

String _tier(int q) =>
    q >= 75 ? 'excellent' : (q >= 62 ? 'strong' : (q >= 48 ? 'good' : 'basic'));

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

class Home extends StatefulWidget {
  const Home({super.key});

  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  LlamaOpenAIClient? _client;
  final _messages = <({String text, bool isUser})>[];
  final _turns = <LlamaResponseInputItem>[]; // role-tagged chat history
  final _input = TextEditingController();
  Model? _selected;
  double? _ramGB;
  double? _progress;
  String? _status;
  bool _generating = false;

  @override
  void initState() {
    super.initState();
    _restore();
  }

  Future<void> _restore() async {
    // Scan the hardware (RAM) and auto-select the best model that fits.
    final mb = await SystemInfoPlus.physicalMemory ?? 0;
    final ramGB = mb / 1024.0;
    final best = _best(ramGB);
    setState(() {
      _ramGB = ramGB > 0 ? ramGB : null;
      _selected = best;
    });
    final path = await _modelPath(best.file);
    if (File(path).existsSync()) _load(path);
  }

  Future<String> _modelPath(String file) async {
    final dir = await getApplicationDocumentsDirectory();
    return '${dir.path}/$file';
  }

  void _load(String path) {
    setState(() {
      _client = LlamaOpenAIClient(
        models: {'local': LlamaModelConfig(modelPath: path)},
      );
      _status = null;
    });
  }

  Future<void> _download() async {
    final m = _selected;
    if (m == null) return;
    final path = await _modelPath(m.file);
    if (File(path).existsSync()) {
      _load(path);
      return;
    }
    setState(() {
      _progress = 0;
      _status = 'Downloading ${m.name} (${m.sizeGB.toStringAsFixed(1)} GB)… '
          'first time only.';
    });
    try {
      // .part then rename, so an interrupted download isn't taken for complete.
      await Dio().download(m.url, '$path.part',
          onReceiveProgress: (rec, total) {
        if (total > 0) setState(() => _progress = rec / total);
      });
      await File('$path.part').rename(path);
      _load(path);
    } catch (e) {
      setState(() => _status = 'Download failed: $e');
    } finally {
      setState(() => _progress = null);
    }
  }

  Future<void> _send() async {
    final text = _input.text.trim();
    final client = _client;
    if (text.isEmpty || client == null || _generating) return;
    _input.clear();
    // Role-tagged history. The `name` field is what forces lib_llama_cpp down
    // its chat-template path (generateMessages) instead of raw completion —
    // without it the model gets an un-templated prompt and rambles into
    // nonsense. Standard chat templates ignore `name`, so it's harmless.
    _turns.add(LlamaResponseInputItem(role: 'user', content: text, name: 'u'));
    setState(() {
      _messages.add((text: text, isUser: true));
      _messages.add((text: '', isUser: false));
      _generating = true;
    });
    final reply = StringBuffer();
    try {
      await for (final event in client.responses.stream(
        model: 'local',
        input: List<LlamaResponseInputItem>.from(_turns),
        instructions:
            'You are a helpful assistant. Answer clearly and concisely.',
        temperature: 0.7, // greedy (no temp) loops/repeats on-device
        topP: 0.9,
        maxOutputTokens: 512,
      )) {
        if (event case LlamaResponseOutputTextDelta(:final delta)) {
          reply.write(delta);
          setState(() {
            final last = _messages.removeLast();
            _messages.add((text: last.text + delta, isUser: false));
          });
        }
      }
      _turns.add(LlamaResponseInputItem(
          role: 'assistant', content: reply.toString(), name: 'a'));
    } catch (e) {
      setState(() {
        _messages.removeLast();
        _messages.add((text: '⚠️ $e', isUser: false));
      });
    } finally {
      setState(() => _generating = false);
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('🧠 On-Device AI — private, local')),
        body: _client == null ? _setup() : _chatView(),
      );

  Widget _setup() {
    final ram = _ramGB;
    if (ram == null) {
      return const Center(child: Text('Scanning your device…'));
    }
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
              children: [
                for (final m in ranked) _modelTile(m, ram, best),
              ],
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
                  : 'Download & start  ·  ${_selected!.name}'),
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
      onChanged: _progress != null
          ? null
          : (_) => setState(() => _selected = m),
      secondary: Text(badge, style: const TextStyle(fontSize: 18)),
      title: Row(
        children: [
          Flexible(child: Text(m.name)),
          if (rec)
            const Padding(
              padding: EdgeInsets.only(left: 8),
              child: Text('★ best for you',
                  style: TextStyle(fontSize: 12, color: Colors.indigo)),
            ),
        ],
      ),
      subtitle: Text('${m.params}B · ${m.sizeGB.toStringAsFixed(1)} GB · '
          '${_tier(m.quality)} · ${m.family}'),
    );
  }

  Widget _chatView() {
    return Column(
      children: [
        Expanded(
          child: ListView.builder(
            padding: const EdgeInsets.all(12),
            itemCount: _messages.length,
            itemBuilder: (_, i) {
              final m = _messages[i];
              return Align(
                alignment:
                    m.isUser ? Alignment.centerRight : Alignment.centerLeft,
                child: Container(
                  margin: const EdgeInsets.symmetric(vertical: 4),
                  padding: const EdgeInsets.all(12),
                  constraints: const BoxConstraints(maxWidth: 520),
                  decoration: BoxDecoration(
                    color: m.isUser
                        ? Theme.of(context).colorScheme.primaryContainer
                        : Theme.of(context).colorScheme.surfaceContainerHighest,
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: Text(m.text.isEmpty ? '…' : m.text),
                ),
              );
            },
          ),
        ),
        SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
            child: Row(
              children: [
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
              ],
            ),
          ),
        ),
      ],
    );
  }
}

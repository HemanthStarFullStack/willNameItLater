import 'dart:io';

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:lib_llama_cpp/lib_llama_cpp.dart';
import 'package:path_provider/path_provider.dart';
import 'package:system_info_plus/system_info_plus.dart';

// Open GGUF model ladder (Qwen2.5) — direct HuggingFace downloads, no account
// and no token. `minRamGB` is the total device RAM below which we won't
// auto-pick a model: the weights + KV cache + the app all have to fit in memory
// or the OS kills us.
const _models = [
  (
    name: 'Qwen2.5 0.5B',
    size: '~400 MB',
    minRamGB: 0.0,
    file: 'qwen2.5-0.5b-instruct-q4_k_m.gguf',
    url: 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/'
        'qwen2.5-0.5b-instruct-q4_k_m.gguf',
  ),
  (
    name: 'Qwen2.5 1.5B',
    size: '~1.0 GB',
    minRamGB: 3.0,
    file: 'qwen2.5-1.5b-instruct-q4_k_m.gguf',
    url: 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/'
        'qwen2.5-1.5b-instruct-q4_k_m.gguf',
  ),
  (
    name: 'Qwen2.5 3B',
    size: '~2.0 GB',
    minRamGB: 6.0,
    file: 'qwen2.5-3b-instruct-q4_k_m.gguf',
    url: 'https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/'
        'qwen2.5-3b-instruct-q4_k_m.gguf',
  ),
];

// Let the device's specs choose: the biggest model whose RAM floor it clears.
int _pickModel(double ramGB) {
  var best = 0;
  for (var i = 0; i < _models.length; i++) {
    if (ramGB >= _models[i].minRamGB) best = i;
  }
  return best;
}

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
  final _input = TextEditingController();
  int _modelIx = 0;
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
    // Read total RAM and let the phone's specs choose the model automatically.
    final mb = await SystemInfoPlus.physicalMemory ?? 0;
    final ramGB = mb / 1024.0;
    final ix = _pickModel(ramGB);
    setState(() {
      _ramGB = ramGB > 0 ? ramGB : null;
      _modelIx = ix;
    });
    // If that model is already on disk from a previous run, load it now.
    final path = await _modelPath(_models[ix].file);
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
    final m = _models[_modelIx];
    final path = await _modelPath(m.file);
    if (File(path).existsSync()) {
      _load(path);
      return;
    }
    setState(() {
      _progress = 0;
      _status = 'Downloading ${m.name} (${m.size})… first time only.';
    });
    try {
      // Download to a .part file, then rename — so an interrupted download is
      // never mistaken for a complete model on the next launch.
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
    setState(() {
      _messages.add((text: text, isUser: true));
      _messages.add((text: '', isUser: false));
      _generating = true;
    });
    try {
      await for (final event in client.responses.stream(
        model: 'local',
        input: text,
      )) {
        if (event case LlamaResponseOutputTextDelta(:final delta)) {
          setState(() {
            final last = _messages.removeLast();
            _messages.add((text: last.text + delta, isUser: false));
          });
        }
      }
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
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            _ramGB == null
                ? 'Reading your device to choose a model…'
                : 'Your device has ${_ramGB!.toStringAsFixed(1)} GB RAM — '
                    'auto-selected ${_models[_modelIx].name}. Everything runs on '
                    'the device; nothing you type ever leaves it, and no account '
                    'or token is needed.',
          ),
          const SizedBox(height: 16),
          for (var i = 0; i < _models.length; i++)
            RadioListTile<int>(
              value: i,
              groupValue: _modelIx,
              title: Text('${_models[i].name}  (${_models[i].size})'),
              onChanged: _progress != null
                  ? null
                  : (v) => setState(() => _modelIx = v!),
            ),
          const SizedBox(height: 16),
          if (_progress != null) ...[
            LinearProgressIndicator(value: _progress),
            const SizedBox(height: 8),
            Text('${((_progress ?? 0) * 100).toStringAsFixed(0)}%'),
          ] else
            FilledButton(
              onPressed: _download,
              child: const Text('Download & start'),
            ),
          if (_status != null) ...[
            const SizedBox(height: 12),
            Text(_status!, style: const TextStyle(color: Colors.orange)),
          ],
        ],
      ),
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

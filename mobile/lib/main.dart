import 'package:flutter/material.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:flutter_gemma_mediapipe/flutter_gemma_mediapipe.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:system_info_plus/system_info_plus.dart';

// On-device model ladder, smallest first. `minRamGB` is the total device RAM
// below which we won't auto-pick a model (weights + KV cache + the app all have
// to fit in memory or the OS kills us). Gated models sit behind a free
// HuggingFace token with the Gemma license accepted.
const _models = [
  (
    name: 'Gemma 3 1B',
    size: '~550 MB',
    minRamGB: 0.0,
    gated: false,
    url: 'https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/'
        'Gemma3-1B-IT_multi-prefill-seq_q8_ekv2048.task',
  ),
  (
    name: 'Gemma 3n E2B',
    size: '~3.1 GB',
    minRamGB: 6.0,
    gated: true,
    url: 'https://huggingface.co/google/gemma-3n-E2B-it-litert-preview/'
        'resolve/main/gemma-3n-E2B-it-int4.task',
  ),
];

// Let the device's specs choose: the biggest model whose RAM floor it clears.
// Gated models are skipped when there's no token (we can't download them), so
// the auto-pick always lands on something actually installable.
int _pickModel(double ramGB, bool hasToken) {
  var best = 0;
  for (var i = 0; i < _models.length; i++) {
    if (ramGB >= _models[i].minRamGB && (!_models[i].gated || hasToken)) {
      best = i;
    }
  }
  return best;
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await FlutterGemma.initialize(inferenceEngines: [MediaPipeEngine()]);
  runApp(const App());
}

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
  InferenceChat? _chat;
  final _messages = <({String text, bool isUser})>[];
  final _input = TextEditingController();
  final _token = TextEditingController();
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
    final prefs = await SharedPreferences.getInstance();
    _token.text = prefs.getString('hf_token') ?? '';

    // Read total RAM and let the phone's specs choose the model automatically.
    final mb = await SystemInfoPlus.physicalMemory ?? 0;
    final ramGB = mb / 1024.0;
    setState(() {
      _ramGB = ramGB > 0 ? ramGB : null;
      _modelIx = _pickModel(ramGB, _token.text.trim().isNotEmpty);
    });

    if (FlutterGemma.hasActiveModel()) {
      try {
        await _loadModel();
      } catch (_) {
        // Model file gone or engine failed; show setup again.
      }
    }
  }

  Future<void> _loadModel() async {
    setState(() => _status = 'Loading model…');
    final model = await FlutterGemma.getActiveModel(
      maxTokens: 2048,
      preferredBackend: PreferredBackend.gpu, // Metal on the iPad
    );
    final chat = await model.createChat(temperature: 0.7);
    setState(() {
      _chat = chat;
      _status = null;
    });
  }

  Future<void> _download() async {
    final m = _models[_modelIx];
    final token = _token.text.trim();
    if (m.gated && token.isEmpty) {
      setState(() => _status = 'This model needs a HuggingFace token '
          '(free: huggingface.co/settings/tokens, accept the Gemma license).');
      return;
    }
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('hf_token', token);
    setState(() {
      _progress = 0;
      _status = 'Downloading ${m.name}…';
    });
    try {
      await FlutterGemma.installModel(modelType: ModelType.gemmaIt)
          .fromNetwork(m.url, token: token.isEmpty ? null : token)
          .withProgress((p) => setState(() => _progress = p / 100))
          .install();
      await _loadModel();
    } catch (e) {
      setState(() => _status = 'Download failed: $e');
    } finally {
      setState(() => _progress = null);
    }
  }

  Future<void> _send() async {
    final text = _input.text.trim();
    final chat = _chat;
    if (text.isEmpty || chat == null || _generating) return;
    _input.clear();
    setState(() {
      _messages.add((text: text, isUser: true));
      _messages.add((text: '', isUser: false));
      _generating = true;
    });
    try {
      await chat.addQuery(Message.text(text: text, isUser: true));
      await for (final r in chat.generateChatResponseAsync()) {
        if (r is TextResponse) {
          setState(() {
            final last = _messages.removeLast();
            _messages.add((text: last.text + r.token, isUser: false));
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
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('🧠 On-Device AI — private, local')),
      body: _chat == null ? _setup() : _chatView(),
    );
  }

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
                    'auto-selected ${_models[_modelIx].name}. Everything runs '
                    'on the device; nothing you type ever leaves it.',
          ),
          const SizedBox(height: 16),
          for (var i = 0; i < _models.length; i++)
            RadioListTile<int>(
              value: i,
              groupValue: _modelIx,
              title: Text('${_models[i].name}  (${_models[i].size})'
                  '${_models[i].gated ? ' — needs HF token' : ''}'),
              onChanged: _progress != null
                  ? null
                  : (v) => setState(() => _modelIx = v!),
            ),
          TextField(
            controller: _token,
            decoration: const InputDecoration(
              labelText: 'HuggingFace token (only for gated models)',
              border: OutlineInputBorder(),
            ),
            obscureText: true,
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
                      hintText: 'Ask anything — it stays on this iPad',
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

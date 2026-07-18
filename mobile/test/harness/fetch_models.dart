/// Downloads the GGUF models the harness tests use into test/harness/models/
/// (gitignored). All URLs are ungated public mirrors — no HuggingFace token.
/// Skips files that already exist; safe to re-run.
library;

import 'dart:io';

const _models = <String, String>{
  // --- catalog candidates (July 2026) ---
  'Qwen3.5-0.8B-Q4_K_M.gguf':
      'https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf',
  'Qwen3.5-2B-Q4_K_M.gguf':
      'https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf',
  'gemma-3-1b-it-Q4_K_M.gguf':
      'https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf',
  'Qwen3.5-4B-Q4_K_M.gguf':
      'https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf',
  'gemma-4-E2B-it-Q4_K_M.gguf':
      'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf',
  // --- embedder candidates ---
  'embeddinggemma-300m-qat-Q8_0.gguf':
      'https://huggingface.co/ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/resolve/main/embeddinggemma-300m-qat-Q8_0.gguf',
  'Qwen3-Embedding-0.6B-Q8_0.gguf':
      'https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf',
};

Future<void> main(List<String> args) async {
  final dir = Directory('test/harness/models');
  dir.createSync(recursive: true);
  final only = args.isEmpty ? null : args.toSet();
  final client = HttpClient();
  for (final MapEntry(key: name, value: url) in _models.entries) {
    if (only != null && !only.contains(name)) continue;
    final file = File('${dir.path}/$name');
    if (file.existsSync() && file.lengthSync() > 1024 * 1024) {
      stdout.writeln('✓ $name (cached)');
      continue;
    }
    stdout.writeln('↓ $name');
    var uri = Uri.parse(url);
    HttpClientResponse resp;
    // Follow redirects manually so cross-host CDN hops work.
    while (true) {
      final req = await client.getUrl(uri);
      req.followRedirects = false;
      resp = await req.close();
      if (resp.isRedirect) {
        uri = uri.resolve(resp.headers.value('location')!);
        await resp.drain<void>();
        continue;
      }
      break;
    }
    if (resp.statusCode != 200) {
      stderr.writeln('✗ $name: HTTP ${resp.statusCode} — check URL/gating');
      await resp.drain<void>();
      exitCode = 1;
      continue;
    }
    final tmp = File('${file.path}.part');
    final sink = tmp.openWrite();
    var got = 0;
    final total = resp.contentLength;
    await for (final chunk in resp) {
      sink.add(chunk);
      got += chunk.length;
      if (total > 0 && got % (64 << 20) < chunk.length) {
        stdout.writeln('  ${(got / total * 100).toStringAsFixed(0)}%');
      }
    }
    await sink.close();
    tmp.renameSync(file.path);
    stdout.writeln('✓ $name (${(got / (1 << 20)).toStringAsFixed(0)} MB)');
  }
  client.close();
}

/// Agentic tool router — FunctionGemma-270M (Google, Dec 2025) as a tiny
/// specialist that turns a user message into a tool call. Runs as a third
/// llama.cpp session (~300MB), same engine as chat + embeddings, using the
/// engine's NATIVE tool channel: the chat template declares tools in the
/// model's trained format and the engine's parser returns structured calls.
///
/// Adding a capability later (reminders was first) = one [AgentTool] entry in
/// the registry + a handler where the pipeline dispatches the call.
library;

import 'dart:convert';

import 'package:lib_llama_cpp/lib_llama_cpp.dart';

import 'llm.dart';

class AgentTool {
  final String name;
  final String description;

  /// Parameter name -> short description. All params are STRING — keep tools
  /// simple; the handler parses richer types ("9pm tomorrow") itself.
  final Map<String, String> params;
  const AgentTool(this.name, this.description, this.params);

  LlamaTool toLlama() => LlamaTool(
        name: name,
        description: description,
        parameters: {
          'type': 'object',
          'properties': {
            for (final e in params.entries)
              e.key: {'type': 'string', 'description': e.value},
          },
          'required': [...params.keys],
        },
      );
}

/// Tools the agent actually dispatches. Measured on the harness (July 2026):
/// zero-shot FunctionGemma-270M is reliable on intents with distinctive verbs
/// (remind/list) but confuses save-vs-forget and misses lookups — so memory
/// and web stay on the proven deterministic pipeline, and this registry holds
/// only the agentic capabilities. Upgrade path when more tools land here:
/// fine-tune FunctionGemma on the full set (google's documented workflow).
const kAgentTools = [
  AgentTool(
      'set_reminder',
      'Set a reminder, e.g. "remind me to take my pills at 9pm".',
      {'text': 'what to remind about', 'time': 'when, as the user said it'}),
  AgentTool('list_reminders',
      'Show the user\'s reminders, e.g. "what reminders do I have?"', {}),
];

class ToolCall {
  final String name;
  final Map<String, String> args;
  const ToolCall(this.name, this.args);
  @override
  String toString() =>
      '$name(${args.entries.map((e) => '${e.key}: ${e.value}').join(', ')})';
}

/// A dedicated FunctionGemma session. Load once, route many.
class ToolRouter {
  final Llm _llm;
  final List<LlamaTool> _tools;
  ToolRouter({String? libraryPath, List<AgentTool> tools = kAgentTools})
      : _llm = Llm(libraryPath: libraryPath),
        _tools = [for (final t in tools) t.toLlama()];

  Future<void> load(String modelPath) =>
      _llm.load(modelPath, contextSize: 2048);

  /// The tool FunctionGemma picks for this message, or null for plain chat.
  Future<ToolCall?> route(String message) async {
    try {
      final (raw, calls) = await _llm.chatWithTools(
        message,
        tools: _tools,
        // The model's own end-of-call marker; without it the 270M rambles
        // fabricated <start_function_response> turns until maxTokens.
        stop: ['<end_function_call>'],
      );
      if (calls.isNotEmpty) {
        // Engine parser knew the format (newer pins) — trust it.
        final c = calls.first;
        final args = <String, String>{};
        try {
          final parsed = jsonDecode(c.arguments);
          if (parsed is Map) parsed.forEach((k, v) => args['$k'] = '$v');
        } catch (_) {}
        return ToolCall(c.name, args);
      }
      return parseToolCall(raw);
    } catch (_) {
      return null; // router failure must never kill the turn
    }
  }
}

// --- FunctionGemma output parsing (this llama.cpp pin predates the format) --

final _callRe = RegExp(r'call:(\w+)\{([\s\S]*)$');
// Args use <escape> as the trained string delimiter:  time:<escape>9pm<escape>
final _escArg = RegExp(r'(\w+)\s*:\s*<escape>([\s\S]*?)<escape>');
final _bareArg = RegExp(r'(\w+)\s*:\s*([^,}<\n]+)');

ToolCall? parseToolCall(String raw) {
  // stop:['<end_function_call>'] trims the tail, so the body runs to EOS.
  final body = raw.split('<end_function_call>').first;
  final m = _callRe.firstMatch(body);
  if (m == null) return null;
  final argText = m.group(2)!;
  final args = <String, String>{};
  for (final a in _escArg.allMatches(argText)) {
    args[a.group(1)!] = a.group(2)!.trim();
  }
  if (args.isEmpty) {
    for (final a in _bareArg.allMatches(argText)) {
      args[a.group(1)!] = a.group(2)!.trim();
    }
  }
  return ToolCall(m.group(1)!, args);
}

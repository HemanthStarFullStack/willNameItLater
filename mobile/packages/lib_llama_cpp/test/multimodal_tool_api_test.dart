import 'dart:typed_data';

import 'package:lib_llama_cpp/lib_llama_cpp.dart';
import 'package:lib_llama_cpp_platform_interface/lib_llama_cpp_platform_interface.dart';
import 'package:test/test.dart';

final class ScriptedLlamaEngine implements LlamaEngine {
  ScriptedLlamaEngine(this.responses);

  final List<LlamaResponse> responses;
  final commands = <LlamaCommand>[];

  @override
  Stream<LlamaResponse> transform(
    Stream<LlamaCommand> input, {
    LlamaState initialState = const LlamaState.empty(),
    LlamaCppLibraryRequest libraryRequest = const LlamaCppLibraryRequest(),
  }) async* {
    commands.addAll(await input.toList());
    yield* Stream<LlamaResponse>.fromIterable(responses);
  }
}

void main() {
  group('multimodal message generation', () {
    test(
      'maps multipart input and tools to message generation commands',
      () async {
        final imageBytes = Uint8List.fromList([1, 2, 3, 4]);
        final engine = ScriptedLlamaEngine([
          const LlamaTokenResponse(text: 'ok', index: 0),
        ]);
        final client = LlamaOpenAIClient(
          models: {
            'local': const LlamaModelConfig(
              modelPath: '/models/local.gguf',
              contextSize: 4096,
              gpuLayerCount: 0,
              mmprojPath: '/models/mmproj.gguf',
              imageMinTokens: 64,
              imageMaxTokens: 512,
            ),
          },
          engine: engine,
        );

        final response = await client.responses.create(
          model: 'local',
          instructions: 'Be concise.',
          input: [
            LlamaResponseInputItem(
              role: 'user',
              content: [
                const LlamaTextPart('Describe this media.'),
                LlamaImageBytesPart(bytes: imageBytes, mimeType: 'image/png'),
                const LlamaAudioFilePart(path: '/tmp/question.wav'),
              ],
            ),
          ],
          maxOutputTokens: 32,
          tools: const [
            LlamaTool(
              name: 'lookup',
              description: 'Lookup facts.',
              parameters: {'type': 'object'},
            ),
          ],
          toolChoice: const LlamaToolChoice.tool('lookup'),
        );

        expect(response.outputText, 'ok');
        expect(
          engine.commands,
          contains(
            const LlamaLoadModelCommand(
              modelPath: '/models/local.gguf',
              contextSize: 4096,
              gpuLayerCount: 0,
              mmprojPath: '/models/mmproj.gguf',
              imageMinTokens: 64,
              imageMaxTokens: 512,
            ),
          ),
        );

        final generate = engine.commands
            .whereType<LlamaGenerateMessagesCommand>()
            .single;
        expect(generate.maxTokens, 32);
        expect(generate.toolChoice, const LlamaToolChoice.tool('lookup'));
        expect(generate.tools.single.name, 'lookup');
        expect(generate.messages.first.role, 'system');
        expect(generate.messages.first.content, 'Be concise.');
        expect(generate.messages.last.role, 'user');
        final parts = generate.messages.last.content as List<LlamaContentPart>;
        expect(parts.whereType<LlamaImageBytesPart>().single.bytes, imageBytes);
        expect(
          parts.whereType<LlamaAudioFilePart>().single.path,
          '/tmp/question.wav',
        );
      },
    );

    test(
      'fails clearly when media is used without a multimodal projector',
      () async {
        final engine = ScriptedLlamaEngine(const []);
        final client = LlamaOpenAIClient(
          models: {
            'local': const LlamaModelConfig(modelPath: '/models/local.gguf'),
          },
          engine: engine,
        );

        await expectLater(
          client.responses.create(
            model: 'local',
            input: const [
              LlamaResponseInputItem(
                role: 'user',
                content: [
                  LlamaTextPart('Describe this image.'),
                  LlamaImageFilePart(path: '/tmp/image.png'),
                ],
              ),
            ],
          ),
          throwsA(
            isA<LlamaOpenAIException>()
                .having(
                  (error) => error.code,
                  'code',
                  'unsupported_model_capability',
                )
                .having((error) => error.param, 'param', 'input'),
          ),
        );
        expect(engine.commands, isEmpty);
      },
    );

    test('preserves tool result follow-up messages', () async {
      final engine = ScriptedLlamaEngine([
        const LlamaTokenResponse(text: 'done', index: 0),
      ]);
      final client = LlamaOpenAIClient(
        models: {
          'local': const LlamaModelConfig(modelPath: '/models/local.gguf'),
        },
        engine: engine,
      );

      await client.responses.create(
        model: 'local',
        input: const [
          LlamaResponseInputItem(
            role: 'assistant',
            content: '',
            toolCalls: [
              LlamaToolCall(
                id: 'call_1',
                index: 0,
                name: 'lookup',
                arguments: '{"query":"weather"}',
              ),
            ],
          ),
          LlamaResponseInputItem(
            role: 'tool',
            content: '{"temperature":72}',
            toolCallId: 'call_1',
            name: 'lookup',
          ),
        ],
      );

      final generate = engine.commands
          .whereType<LlamaGenerateMessagesCommand>()
          .single;
      expect(generate.messages.first.toolCalls.single.id, 'call_1');
      expect(generate.messages.last.role, 'tool');
      expect(generate.messages.last.toolCallId, 'call_1');
      expect(generate.messages.last.name, 'lookup');
    });
  });

  group('tool call streaming', () {
    test('emits structured tool-call events and requires action', () async {
      final engine = ScriptedLlamaEngine([
        const LlamaToolCallResponse(
          toolCall: LlamaToolCall(
            id: 'call_1',
            index: 0,
            name: 'lookup',
            arguments: '{"query":"weather"}',
          ),
        ),
      ]);
      final client = LlamaOpenAIClient(
        models: {
          'local': const LlamaModelConfig(modelPath: '/models/local.gguf'),
        },
        engine: engine,
      );

      final events = await client.responses
          .stream(
            model: 'local',
            input: 'What is the weather?',
            tools: const [
              LlamaTool(
                name: 'lookup',
                description: 'Lookup facts.',
                parameters: {'type': 'object'},
              ),
            ],
          )
          .toList();

      final toolDone = events.whereType<LlamaResponseToolCallDone>().single;
      expect(toolDone.toolCall.name, 'lookup');
      expect(toolDone.toolCall.arguments, '{"query":"weather"}');
      final requiresAction = events
          .whereType<LlamaResponseRequiresAction>()
          .single;
      expect(requiresAction.response.status, 'requires_action');
      expect(requiresAction.toolCalls.single.id, 'call_1');
      expect(events.last, isA<LlamaResponseRequiresAction>());
    });
  });
}

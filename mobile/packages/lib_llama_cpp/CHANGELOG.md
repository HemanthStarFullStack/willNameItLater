## Unreleased

- Exported `lib_llama_cpp_server` from the app-facing facade and documented
  server mode as the recommended local-model integration path.
- Documented CPU-only pub.dev prebuilts with separate Metal, Vulkan, and CUDA
  GitHub release assets.
- Stream parsed plain-text deltas during tool-aware message generation while
  preserving structured tool-call parsing.
- Added Metal capability reporting for bundled Apple libraries and explicit
  unsupported-capability errors for bundled CPU-only libraries.
- Added CMake build options for Metal, CUDA, and Vulkan native backend builds.

## 0.4.0

- Added chat-template-aware `LlamaGenerateMessagesCommand` for multipart
  messages while keeping raw `LlamaGenerateCommand(prompt: ...)` unchanged.
- Added image/audio content parts, `mmprojPath` model configuration, runtime
  model capabilities, and unsupported-capability errors for media requests.
- Added OpenAI-style function tool definitions, tool choices, tool call
  responses, tool result message fields, and `requires_action` stream events.
- Document CPU-only native platform packages built by the GitHub release
  workflow.

## 0.1.0

- Initial release of the app-facing Flutter plugin facade.
- Added command and response types for model loading, generation, and disposal.
- Added inference isolate lifecycle orchestration and federated platform lookup.

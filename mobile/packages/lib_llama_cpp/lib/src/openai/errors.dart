final class LlamaOpenAIException implements Exception {
  const LlamaOpenAIException({
    required this.code,
    required this.message,
    this.param,
    this.type = 'invalid_request_error',
  });

  final String code;
  final String message;
  final String? param;
  final String type;

  @override
  String toString() {
    final parameter = param == null ? '' : ', param: $param';
    return 'LlamaOpenAIException(code: $code$parameter, message: $message)';
  }
}

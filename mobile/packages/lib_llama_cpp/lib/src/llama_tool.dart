import 'dart:convert';

final class LlamaTool {
  const LlamaTool({
    required this.name,
    this.description = '',
    this.parameters = const {},
  });

  factory LlamaTool.fromJson(Map<String, Object?> json) {
    final function = _stringKeyedMapOrNull(json['function']);
    final source = function ?? json;
    final parameters =
        source['parameters'] ?? source['input_schema'] ?? source['inputSchema'];

    return LlamaTool(
      name: _requiredString(source, 'name'),
      description: _optionalString(source['description']) ?? '',
      parameters: parameters == null
          ? const {}
          : _stringKeyedMap(parameters, 'parameters'),
    );
  }

  final String name;
  final String description;
  final Map<String, Object?> parameters;

  Map<String, Object?> toJson() {
    return {
      'type': 'function',
      'function': {
        'name': name,
        if (description.isNotEmpty) 'description': description,
        'parameters': parameters,
      },
    };
  }
}

final class LlamaToolChoice {
  const LlamaToolChoice.tool(String name) : this._('tool', name: name);

  const LlamaToolChoice._(this.mode, {this.name});

  factory LlamaToolChoice.fromJson(Object? json) {
    if (json == null) {
      return auto;
    }
    if (json is String) {
      return switch (json) {
        'auto' => auto,
        'none' => none,
        'required' => required,
        _ => throw ArgumentError.value(
          json,
          'json',
          'Unsupported tool choice mode',
        ),
      };
    }

    final map = _stringKeyedMap(json, 'json');
    final type = _optionalString(map['type']);
    if (type == 'auto') {
      return auto;
    }
    if (type == 'none') {
      return none;
    }
    if (type == 'required') {
      return required;
    }

    final function = _stringKeyedMapOrNull(map['function']);
    final name = function == null
        ? _optionalString(map['name'])
        : _optionalString(function['name']);
    if ((type == 'function' || type == 'tool') &&
        name != null &&
        name.isNotEmpty) {
      return LlamaToolChoice.tool(name);
    }

    throw ArgumentError.value(json, 'json', 'Unsupported tool choice shape');
  }

  static const auto = LlamaToolChoice._('auto');
  static const none = LlamaToolChoice._('none');
  static const required = LlamaToolChoice._('required');

  final String mode;
  final String? name;

  Object toJson() {
    if (mode == 'tool' && name != null) {
      return {
        'type': 'function',
        'function': {'name': name},
      };
    }
    return mode;
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        other is LlamaToolChoice && other.mode == mode && other.name == name;
  }

  @override
  int get hashCode => Object.hash(mode, name);

  @override
  String toString() {
    final toolName = name == null ? '' : ', name: $name';
    return 'LlamaToolChoice(mode: $mode$toolName)';
  }
}

final class LlamaToolCall {
  const LlamaToolCall({
    required this.id,
    required this.index,
    required this.name,
    required this.arguments,
  });

  factory LlamaToolCall.fromJson(Map<String, Object?> json, {int? index}) {
    final function = _stringKeyedMapOrNull(json['function']);
    final source = function ?? json;
    final resolvedIndex = index ?? _optionalInt(json['index']) ?? 0;
    final rawArguments = source['arguments'];

    return LlamaToolCall(
      id: _optionalString(json['id']) ?? 'call_$resolvedIndex',
      index: resolvedIndex,
      name: _requiredString(source, 'name'),
      arguments: rawArguments is String
          ? rawArguments
          : jsonEncode(rawArguments ?? const <String, Object?>{}),
    );
  }

  final String id;
  final int index;
  final String name;
  final String arguments;

  Map<String, Object?> toJson() {
    return {
      'id': id,
      'index': index,
      'type': 'function',
      'function': {'name': name, 'arguments': arguments},
    };
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        other is LlamaToolCall &&
            other.id == id &&
            other.index == index &&
            other.name == name &&
            other.arguments == arguments;
  }

  @override
  int get hashCode => Object.hash(id, index, name, arguments);

  @override
  String toString() {
    return 'LlamaToolCall('
        'id: $id, '
        'index: $index, '
        'name: $name, '
        'arguments: $arguments'
        ')';
  }
}

String _requiredString(Map<String, Object?> json, String key) {
  final value = json[key];
  if (value is String && value.isNotEmpty) {
    return value;
  }
  throw ArgumentError.value(json, 'json', 'Expected non-empty string "$key"');
}

String? _optionalString(Object? value) {
  return value is String ? value : null;
}

int? _optionalInt(Object? value) {
  return value is int ? value : null;
}

Map<String, Object?> _stringKeyedMap(Object? value, String name) {
  final map = _stringKeyedMapOrNull(value);
  if (map != null) {
    return map;
  }
  throw ArgumentError.value(value, name, 'Expected a JSON object');
}

Map<String, Object?>? _stringKeyedMapOrNull(Object? value) {
  if (value is! Map) {
    return null;
  }

  return {
    for (final entry in value.entries)
      if (entry.key is String) entry.key as String: entry.value,
  };
}

import 'dart:typed_data';

sealed class LlamaContentPart {
  const LlamaContentPart();

  bool get isMedia {
    final part = this;
    return part is LlamaImageFilePart ||
        part is LlamaImageBytesPart ||
        part is LlamaAudioFilePart ||
        part is LlamaAudioBytesPart;
  }
}

final class LlamaTextPart extends LlamaContentPart {
  const LlamaTextPart(this.text);

  final String text;
}

final class LlamaImageFilePart extends LlamaContentPart {
  const LlamaImageFilePart({required this.path, this.mimeType});

  final String path;
  final String? mimeType;
}

final class LlamaImageBytesPart extends LlamaContentPart {
  const LlamaImageBytesPart({required this.bytes, this.mimeType});

  final Uint8List bytes;
  final String? mimeType;
}

final class LlamaAudioFilePart extends LlamaContentPart {
  const LlamaAudioFilePart({required this.path, this.mimeType});

  final String path;
  final String? mimeType;
}

final class LlamaAudioBytesPart extends LlamaContentPart {
  const LlamaAudioBytesPart({required this.bytes, this.mimeType});

  final Uint8List bytes;
  final String? mimeType;
}

bool llamaContentHasMedia(Object content) {
  if (content is List<LlamaContentPart>) {
    return content.any((part) => part.isMedia);
  }
  return false;
}

Object llamaContentFromJson(Object? content) {
  if (content == null) {
    return '';
  }
  if (content is String) {
    return content;
  }
  if (content is List) {
    return [for (final item in content) _contentPartFromJson(item)];
  }
  throw ArgumentError.value(content, 'content', 'Unsupported message content');
}

Object llamaContentToJson(Object content) {
  if (content is String) {
    return content;
  }
  if (content is List<LlamaContentPart>) {
    return [for (final part in content) _contentPartToJson(part)];
  }
  throw ArgumentError.value(content, 'content', 'Unsupported message content');
}

String llamaContentToPlainText(Object content) {
  if (content is String) {
    return content;
  }
  if (content is List<LlamaContentPart>) {
    final buffer = StringBuffer();
    for (final part in content) {
      if (part is LlamaTextPart) {
        buffer.write(part.text);
      } else if (part.isMedia) {
        buffer.write('<__media__>');
      }
    }
    return buffer.toString();
  }
  throw ArgumentError.value(content, 'content', 'Unsupported message content');
}

LlamaContentPart _contentPartFromJson(Object? value) {
  if (value is String) {
    return LlamaTextPart(value);
  }

  final json = _stringKeyedMap(value, 'content part');
  final type = json['type'];
  if (type == 'text' || type == 'input_text') {
    return LlamaTextPart(_optionalString(json['text']) ?? '');
  }

  if (type == 'image_url' || type == 'input_image') {
    final path =
        _optionalString(json['path']) ??
        _mediaUrl(json['image_url']) ??
        _mediaUrl(json['image']);
    if (path != null) {
      return LlamaImageFilePart(
        path: path,
        mimeType: _optionalString(json['mime_type']),
      );
    }
  }

  if (type == 'input_audio' || type == 'audio_file') {
    final path =
        _optionalString(json['path']) ??
        _mediaUrl(json['input_audio']) ??
        _mediaUrl(json['audio_file']) ??
        _mediaUrl(json['audio']);
    if (path != null) {
      return LlamaAudioFilePart(
        path: path,
        mimeType: _optionalString(json['mime_type']),
      );
    }
  }

  throw ArgumentError.value(value, 'content part', 'Unsupported content part');
}

Map<String, Object?> _contentPartToJson(LlamaContentPart part) {
  if (part is LlamaTextPart) {
    return {'type': 'text', 'text': part.text};
  }
  if (part is LlamaImageFilePart) {
    return {
      'type': 'image_url',
      'image_url': {'url': part.path},
      if (part.mimeType != null) 'mime_type': part.mimeType,
    };
  }
  if (part is LlamaAudioFilePart) {
    return {
      'type': 'input_audio',
      'input_audio': {'url': part.path},
      if (part.mimeType != null) 'mime_type': part.mimeType,
    };
  }
  throw ArgumentError.value(part, 'part', 'Unsupported content part');
}

String? _mediaUrl(Object? value) {
  if (value is String) {
    return value;
  }
  final map = _stringKeyedMapOrNull(value);
  return map == null ? null : _optionalString(map['url']);
}

String? _optionalString(Object? value) {
  return value is String ? value : null;
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

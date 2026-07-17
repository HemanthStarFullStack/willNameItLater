/// Web search tool — the port of web.py. Free, no API key: DuckDuckGo's HTML
/// endpoint (what the ddgs package scrapes), parsed with regex. Returns short
/// snippets to keep the local model's context small. Never throws.
library;

import 'package:dio/dio.dart';

class WebResult {
  final String title, url, snippet;
  const WebResult(this.title, this.url, this.snippet);
}

final _dio = Dio(BaseOptions(
  connectTimeout: const Duration(seconds: 8),
  receiveTimeout: const Duration(seconds: 8),
  headers: {'User-Agent': 'Mozilla/5.0 (Linux; Android 14) OnDeviceAI/1.0'},
  responseType: ResponseType.plain,
));

String _unescape(String s) => s
    .replaceAll(RegExp(r'<[^>]+>'), '')
    .replaceAll('&amp;', '&')
    .replaceAll('&lt;', '<')
    .replaceAll('&gt;', '>')
    .replaceAll('&quot;', '"')
    .replaceAll('&#x27;', "'")
    .replaceAll('&#39;', "'")
    .trim();

/// DDG wraps result links as //duckduckgo.com/l/?uddg=<encoded-url>&...
String _realUrl(String href) {
  final m = RegExp(r'uddg=([^&]+)').firstMatch(href);
  if (m != null) return Uri.decodeComponent(m.group(1)!);
  return href.startsWith('//') ? 'https:$href' : href;
}

Future<List<WebResult>> searchWeb(String query, {int k = 4}) async {
  query = query.trim();
  if (query.isEmpty) return const [];
  try {
    final resp = await _dio.get<String>(
      'https://html.duckduckgo.com/html/',
      queryParameters: {'q': query},
    );
    final html = resp.data ?? '';
    final links = RegExp(
      r'class="result__a"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)</a>',
    ).allMatches(html).toList();
    final snippets = RegExp(
      r'class="result__snippet"[^>]*>([\s\S]*?)</a>',
    ).allMatches(html).toList();
    final out = <WebResult>[];
    for (var i = 0; i < links.length && out.length < k; i++) {
      final title = _unescape(links[i].group(2)!);
      final url = _realUrl(links[i].group(1)!);
      final snippet =
          i < snippets.length ? _unescape(snippets[i].group(1)!) : '';
      if (title.isNotEmpty) out.add(WebResult(title, url, snippet));
    }
    return out;
  } catch (_) {
    return const [];
  }
}

/// Compact text block to feed the model as tool output.
String formatResults(List<WebResult> results) {
  if (results.isEmpty) return 'No web results found.';
  return [
    for (var i = 0; i < results.length; i++)
      '[${i + 1}] ${results[i].title}\n${results[i].snippet}\n(${results[i].url})',
  ].join('\n\n');
}

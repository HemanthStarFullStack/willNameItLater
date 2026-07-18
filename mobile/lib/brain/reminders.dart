/// Reminders — the first agentic capability. FunctionGemma routes
/// "remind me to take my pills at 9pm" here; we store it (encrypted like all
/// user data) and schedule a local notification. Fully offline.
library;

import 'dart:convert';
import 'dart:io';

import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:timezone/data/latest.dart' as tzdata;
import 'package:timezone/timezone.dart' as tz;

import 'vault.dart';

class Reminder {
  final int id;
  final String text;
  final String whenText; // as the user said it ("9pm", "tomorrow morning")
  final DateTime? at; // parsed, null if we couldn't — still listed, not fired
  const Reminder(this.id, this.text, this.whenText, this.at);

  Map<String, dynamic> toJson() => {
        'id': id,
        'text': text,
        'when': whenText,
        'at': at?.millisecondsSinceEpoch,
      };
  static Reminder fromJson(Map<String, dynamic> j) => Reminder(
      j['id'] as int,
      j['text'] as String,
      j['when'] as String? ?? '',
      j['at'] == null
          ? null
          : DateTime.fromMillisecondsSinceEpoch(j['at'] as int));
}

/// "9pm" / "21:30" / "at 7" / "tomorrow 8am" / "in 20 minutes" → DateTime.
/// ponytail: regex heuristics cover the common phrasings; unparsed times are
/// stored + listed (never lost), just not notified. Upgrade path: let the chat
/// model normalize the time to ISO if these misses ever matter.
DateTime? parseWhen(String whenText, {DateTime? now}) {
  final t = whenText.toLowerCase().trim();
  now ??= DateTime.now();

  final inM = RegExp(r'in\s+(\d+)\s*(minute|min|hour|hr)s?').firstMatch(t);
  if (inM != null) {
    final n = int.parse(inM.group(1)!);
    return now.add(inM.group(2)!.startsWith('m')
        ? Duration(minutes: n)
        : Duration(hours: n));
  }

  final hm =
      RegExp(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b').firstMatch(t);
  if (hm == null) return null;
  var h = int.parse(hm.group(1)!);
  final m = int.parse(hm.group(2) ?? '0');
  final ap = hm.group(3);
  if (h > 23 || m > 59) return null;
  if (ap == 'pm' && h < 12) h += 12;
  if (ap == 'am' && h == 12) h = 0;
  var at = DateTime(now.year, now.month, now.day, h, m);
  if (t.contains('tomorrow') || !at.isAfter(now)) {
    at = at.add(const Duration(days: 1));
  }
  return at;
}

class ReminderStore {
  final String path;
  final Vault? vault;
  final bool notify; // false on the desktop harness (no plugin registrar)
  final _items = <Reminder>[];
  var _nextId = 1;
  FlutterLocalNotificationsPlugin? _plugin;

  ReminderStore(this.path, {this.vault, this.notify = true});

  List<Reminder> get items => List.unmodifiable(_items);

  Future<void> load() async {
    try {
      final f = File(path);
      if (!f.existsSync()) return;
      final raw = vault != null
          ? await vault!.readString(f)
          : await f.readAsString();
      if (raw == null) return;
      final list = (jsonDecode(raw) as List).cast<Map<String, dynamic>>();
      _items
        ..clear()
        ..addAll(list.map(Reminder.fromJson));
      if (_items.isNotEmpty) {
        _nextId = _items.map((r) => r.id).reduce((a, b) => a > b ? a : b) + 1;
      }
    } catch (_) {/* corrupt file: start empty, never crash boot */}
  }

  Future<void> _save() async {
    final raw = jsonEncode([for (final r in _items) r.toJson()]);
    final f = File(path);
    if (vault != null) {
      await vault!.writeString(f, raw);
    } else {
      await f.writeAsString(raw);
    }
  }

  Future<FlutterLocalNotificationsPlugin?> _notifier() async {
    if (!notify) return null;
    if (_plugin != null) return _plugin;
    try {
      tzdata.initializeTimeZones();
      final p = FlutterLocalNotificationsPlugin();
      await p.initialize(
          settings: const InitializationSettings(
              android: AndroidInitializationSettings('@mipmap/ic_launcher')));
      await p
          .resolvePlatformSpecificImplementation<
              AndroidFlutterLocalNotificationsPlugin>()
          ?.requestNotificationsPermission();
      _plugin = p;
    } catch (_) {/* notifications unavailable — reminders still stored */}
    return _plugin;
  }

  /// Returns a user-facing confirmation line. [fallbackWhen] is the raw user
  /// message — tried when the router's extracted time doesn't parse.
  Future<String> add(String text, String whenText,
      {String? fallbackWhen}) async {
    final at = parseWhen(whenText) ??
        (fallbackWhen == null ? null : parseWhen(fallbackWhen));
    final r = Reminder(_nextId++, text, whenText, at);
    _items.add(r);
    await _save();

    if (at != null) {
      final p = await _notifier();
      if (p != null) {
        try {
          await p.zonedSchedule(
            id: r.id,
            title: 'Reminder',
            body: text,
            scheduledDate: tz.TZDateTime.from(at, tz.local),
            notificationDetails: const NotificationDetails(
                android: AndroidNotificationDetails(
                    'reminders', 'Reminders',
                    importance: Importance.high, priority: Priority.high)),
            // Inexact avoids the exact-alarm permission dance; ±a few minutes
            // is fine for personal reminders.
            androidScheduleMode: AndroidScheduleMode.inexactAllowWhileIdle,
          );
        } catch (_) {/* scheduling failed — reminder still listed */}
      }
      final local = at.toString().substring(0, 16);
      return '⏰ Reminder set: "$text" at $local';
    }
    return '⏰ Saved reminder: "$text" ($whenText) — I couldn\'t parse the '
        'time, so it\'s listed but won\'t ring.';
  }

  String listText() {
    if (_items.isEmpty) return 'No reminders set.';
    return [
      'Your reminders:',
      for (final r in _items)
        '• #${r.id} ${r.text} — ${r.at != null ? r.at.toString().substring(0, 16) : r.whenText}',
    ].join('\n');
  }

  Future<String?> remove(int id) async {
    final i = _items.indexWhere((r) => r.id == id);
    if (i < 0) return null;
    final r = _items.removeAt(i);
    await _save();
    try {
      (await _notifier())?.cancel(id: id);
    } catch (_) {}
    return r.text;
  }
}

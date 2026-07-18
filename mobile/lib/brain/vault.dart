/// Encrypted-at-rest file storage for the private data files (memories,
/// chats). AES-GCM-256; the key is generated once and lives in the platform
/// keystore (Android Keystore via flutter_secure_storage), never in the file.
///
/// File format: 'ODV1' magic + 12-byte nonce + ciphertext + 16-byte MAC.
/// Legacy plaintext JSON files are migrated transparently on first read.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:cryptography/cryptography.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

class Vault {
  static const _magic = [0x4F, 0x44, 0x56, 0x31]; // 'ODV1'
  static const _keyName = 'ondevice_ai_data_key';

  final SecretKey _key;
  final _cipher = AesGcm.with256bits();

  Vault._(this._key);

  /// Loads (or creates on first run) the data key from the platform keystore.
  static Future<Vault> open() async {
    const storage = FlutterSecureStorage(
      aOptions: AndroidOptions(encryptedSharedPreferences: true),
    );
    final stored = await storage.read(key: _keyName);
    if (stored != null) {
      return Vault._(SecretKey(base64Decode(stored)));
    }
    final key = await AesGcm.with256bits().newSecretKey();
    final bytes = await key.extractBytes();
    await storage.write(key: _keyName, value: base64Encode(bytes));
    return Vault._(SecretKey(bytes));
  }

  /// In-memory vault for tests/harness — no platform keystore involved.
  static Vault ephemeral() =>
      Vault._(SecretKey(List<int>.generate(32, (i) => i * 7 % 256)));

  Future<String?> readString(File f) async {
    if (!f.existsSync()) return null;
    final raw = await f.readAsBytes();
    if (raw.length < 4 + 12 + 16 ||
        !(raw[0] == _magic[0] &&
            raw[1] == _magic[1] &&
            raw[2] == _magic[2] &&
            raw[3] == _magic[3])) {
      // Legacy plaintext file — return as-is; the next write encrypts it.
      return utf8.decode(raw, allowMalformed: true);
    }
    final box = SecretBox(
      raw.sublist(16, raw.length - 16),
      nonce: raw.sublist(4, 16),
      mac: Mac(raw.sublist(raw.length - 16)),
    );
    final clear = await _cipher.decrypt(box, secretKey: _key);
    return utf8.decode(clear);
  }

  Future<void> writeString(File f, String content) async {
    final box = await _cipher.encrypt(utf8.encode(content), secretKey: _key);
    final out = BytesBuilder()
      ..add(_magic)
      ..add(box.nonce)
      ..add(box.cipherText)
      ..add(box.mac.bytes);
    await f.parent.create(recursive: true);
    // Write-then-rename so a mid-write crash never corrupts the only copy.
    final tmp = File('${f.path}.tmp');
    await tmp.writeAsBytes(out.toBytes(), flush: true);
    if (f.existsSync()) await f.delete();
    await tmp.rename(f.path);
  }
}

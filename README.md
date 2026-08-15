# tinydrone — Otonom Drone Hedef Tespit ve Takip Sistemi

ESP32-CAM üzerinde, **tinycml** (kendi C11 ML kütüphanemiz) ile çalışan, düşük maliyetli otonom hedef tespit ve takip sistemi. Eğitim dahil her şey C ile yazıldı — Python yalnızca veri hazırlama, dokümantasyon ve simülasyon görselleştirme/bağlantı katmanında.

> Tiny Ecosystem'ün parçası. Felsefe: zero-dependency, pure C, embedded-first.

## Özellikler

- **5 sınıf CNN** (tank, armored_vehicle, drone_uav, background, military_building) — 136K parametre
- **int8 niceleme** — float ile %99.5+ uyum, ESP32'de 0.43MB firmware
- **Kilitli takip** (sliding window + sınıf tutarlılığı) — hedef görüşten sapınca servo/UART komutu
- **UART çıkışı** — `T%+03dY%+03d` protokolü (uçuş platformuna hedef yönlendirme)
- **ESP-IDF v5.3** firmware — OV2640 (160x120), ESP-DSP SIMD, double buffer, watchdog
- **Canlı web simülasyonu** — ArduPilot SITL + MAVLink + gerçek 3D kamera render'ı (720p)
  - 1. ve 3. şahıs kamera, hareketli hedefler, otonom takip
  - Başı-sonu olan **görev senaryosu**: kalkış → 360° arama → tespit → takip → iniş
  - Süreç akışı paneli: KAMERA → CNN → TESPİT → TAKİP → KOMUT

## Mimari

```
simulation/   Canlı web paneli (SITL + 3D render + CNN köprüsü — ctypes)
training/     Veri hazırlama + C eğitimi + niceleme (tinycml ile)
firmware/     ESP32-CAM (ESP-IDF v5.3): kamera → CNN(int8) → takip → UART
docs/         DRONE_REHBERI, ENTEGRASYON_REHBERI, GELISTIRME_AKISI, SIMULASYON_PLANI
```

CNN köprüsü `simulation/bridge.c` — firmware'deki **aynı** `inference_int8.c` + `sliding_window.c` kodunu paylaşır: "aynı model hem simülasyonda hem gömülüde".

## Veri

Eğitim verisi gerçek görsellerden toplandı (Kaggle + xView) — sentetik veri kullanılmadı. Sınıf dağılımı ve kaynaklar: `training/sources.md`.

## Sonuçlar

| Model | Doğruluk | int8 uyumu |
|-------|----------|------------|
| Float | ~%90 | — |
| Int8  | ~%85-90 | %99.5+ |

## Hızlı Başlangıç

```bash
# Simülasyon (SITL + web paneli)
cd simulation && make && python3 sim_telemetry.py --port 8090

# Host testleri
make test-host

# Firmware
cd firmware && idf.py build
```

## Lisans

MIT — bkz. [LICENSE](LICENSE)

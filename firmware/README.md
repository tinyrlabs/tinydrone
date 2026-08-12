# tinydrone — ESP32-CAM Firmware

AI-Thinker ESP32-CAM üzerinde otonom askeri hedef tespiti ve takibi.

## Mimari

```
Kamera (OV2640, QQVGA 160x120)
    │ RGB565
    ▼
Downsample → 32x32x3 NCHW [-1,1]
    ▼
tinycml CNN: Conv(16)→ReLU→Pool→Conv(32)→ReLU→Pool→FC(64)→FC(4)
    ▼
Sınıf + güven skoru  (0=tank, 1=armored, 2=drone, 3=background)
    ▼
Eşik ≥ %60 → Servo takip (pan GPIO14, tilt GPIO15)
```

## Donanım

| Bileşen | Pin |
|---------|-----|
| Kamera | Standart ESP32-CAM pinout |
| Servo pan | GPIO14 |
| Servo tilt | GPIO15 |
| Flash LED | GPIO4 (kullanılmıyor) |

## Build

```bash
# ESP-IDF v5.x kurulu olmalı
export IDF_PATH=~/esp-idf
. $IDF_PATH/export.sh

cd firmware
idf.py set-target esp32
cp sdkconfig.example sdkconfig
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

## Bileşenler

- `components/tinycml/` — tinycml kütüphanesinin ESP-IDF component kopyası
  (matrix, conv2d, pool2d). Kaynak: `tinyrlabs/tinycml`
- `main/model.h` — eğitilmiş model ağırlıkları (training/output/tinydrone_model.h kopyası)

## Bilinen sınırlamalar

- **double (float64) inference**: ~300-800ms/kare, ~1-3 FPS. Gerçek drone için
  int8 quantization şart (Faz 4.5).
- **im2col ara buffer'ları**: col_cache SRAM'de ~200-300KB tutar. 520KB SRAM
  sınırında — PSRAM'e taşınabilir.
- **Basit P kontrolcü**: hedef merkezde varsayılır, bounding box yok.
  Sliding-window tespiti Faz 4.5'te eklenecek.

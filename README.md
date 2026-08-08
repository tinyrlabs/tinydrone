# tinydrone — Otonom Drone Hedef Tespit Sistemi

ESP32-CAM üzerinde, tinycml + klasik CV ile çalışan, düşük maliyetli otonom hedef tespit ve takip sistemi.

> Tiny Ecosystem'ün 7. kütüphanesi. Felsefe: zero-dependency, pure C, embedded-first.

## Proje Vizyonu

Bir drone üzerinde, tamamen gömülü sistemde (ESP32-CAM), herhangi bir bulut/ağ bağlantısı olmadan:
- Kameradan gerçek zamanlı görüntü al
- Hedef nesneyi tespit et (renk + şekil tabanlı)
- tinycml ile nesne sınıflandırması yap
- Servo motorlarla hedefi takip et

## Faz Planı

### Faz 1: Desktop Simülasyon (Mevcut Faz)
- [ ] Webcam'den görüntü yakalama (Linux/macOS)
- [ ] Renk eşikleme ile hedef tespiti (klasik CV, OpenCV'siz)
- [ ] Kontür bulma, bounding box çizme
- [ ] Basit takip algoritması (merkezden sapma → yön komutu)
- [ ] Görsel debug penceresi (SDL2 veya ASCII art)

**Hedef:** Gerçek donanıma geçmeden algoritmaları masaüstünde doğrula.

### Faz 2: ESP32-CAM Firmware
- [ ] OV2640 kamera sürücüsü (ESP-IDF)
- [ ] 320x240 → 32x32 downsample
- [ ] RGB→HSV dönüşümü, renk maskesi
- [ ] Kontür merkezi hesaplama
- [ ] PWM servo kontrolü
- [ ] WiFi üzerinden debug stream (opsiyonel)

**Hedef:** ESP32-CAM üzerinde bağımsız çalışan hedef tespiti.

### Faz 3: tinycml Entegrasyonu
- [ ] Eğitim veri seti toplama (hedef nesne fotoğrafları)
- [ ] Python ile MLP eğitimi (feature extraction + sınıflandırma)
- [ ] Ağırlıkları C array olarak export etme
- [ ] tinycml inference'in ESP32'ye gömülmesi
- [ ] Klasik CV + ML hibrit karar mekanizması

**Hedef:** Sadece rengi değil, nesne tipini de tanıyan sistem.

### Faz 4: CNN Modülü (tinycml'e katkı)
- [ ] Conv2D layer implementasyonu (C11, pool allocator)
- [ ] MaxPool2D layer
- [ ] Im2col + GEMM tabanlı konvolüsyon
- [ ] Python → C weight exporter
- [ ] ESP32'de uçtan uca CNN inference

**Hedef:** tinycml kütüphanesine görüntü işleme yeteneği kazandırmak.

## Teknik Detaylar

### Donanım
| Bileşen | Model | Not |
|---------|-------|-----|
| Mikrodenetleyici | ESP32-CAM (AI-Thinker) | 240MHz dual-core, 520KB SRAM, 4MB flash |
| Kamera | OV2640 | 2MP, JPEG çıkışı, I2C konfigürasyon |
| Güç | 5V ≥2A | Drone BEC veya ayrı regülatör |
| Servo | SG90 (pan/tilt) | 2 adet, PWM kontrollü |

### Klasik CV Pipeline (Faz 1-2)
```
Kamera → 320x240 RGB → 32x32 downsample → RGB→HSV
→ Renk maskesi (hedef renk aralığı) → Kontür bulma
→ En büyük kontürün merkezi → Sapma vektörü → Servo PWM
```

### tinycml Pipeline (Faz 3)
```
Kamera → 32x32 gray → Feature vektörü (histogram, edge, moment)
→ MLP ([64,32] hidden) → Sınıf etiketi → Güven skoru
```

### Hedef Nesne Tipleri (Faz 3)
- Turuncu koni (iniş pisti işareti)
- Kırmızı top (hedef)
- Mavi kare (yön işareti)

## Dizin Yapısı

```
tinydrone/
├── README.md                   # Bu dosya
├── ROADMAP.md                  # Detaylı görev takibi
├── Makefile                    # Host-side simülasyon build
├── firmware/                   # ESP32-CAM firmware
│   ├── CMakeLists.txt          # ESP-IDF build
│   └── main/
│       ├── main.c              # FreeRTOS entry point
│       ├── camera.c/h          # OV2640 wrapper
│       ├── detector.c/h        # Renk/şekil tespiti
│       ├── tracker.c/h         # Servo kontrol
│       └── model.h             # tinycml export ağırlıkları
├── training/                   # Offline eğitim pipeline'ı
│   ├── capture.py              # Webcam'den veri toplama
│   ├── train.py                # sklearn/tinycml ile eğitim
│   └── export_weights.py       # Model → C array
├── simulation/                 # Desktop simülasyon (Faz 1)
│   ├── main.c                  # Ana simülasyon döngüsü
│   ├── capture.c/h             # Webcam capture (v4l2/AVFoundation)
│   ├── detector.c/h            # Klasik CV pipeline'ı
│   └── display.c/h             # SDL2/ASCII görselleştirme
└── docs/
    └── architecture.md         # Sistem mimarisi
```

## Build & Çalıştırma

### Desktop Simülasyon
```bash
# Linux (v4l2)
make sim
./build/sim_tinydrone

# macOS (AVFoundation)
make sim PLATFORM=macos
./build/sim_tinydrone
```

### ESP32 Firmware
```bash
cd firmware
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

### Eğitim
```bash
cd training
python capture.py --class red_ball --count 200
python train.py --dataset ../data/
python export_weights.py --model model.pkl --output ../firmware/main/model.h
```

## Tasarım Kararları

1. **Klasik CV + ML hibrit**: Salt ML çözümleri gömülü sistemde ağır ve kırılgan. Klasik CV (renk/şekil) hızlı filtre, ML sınıflandırma için kullanılır.
2. **32x32 çözünürlük**: 1024 byte/piksel. ESP32 SRAM'inde rahat. MLP için yeterli ayırt edicilik.
3. **Zero-dependency**: tinycml felsefesiyle uyumlu. Host tarafında bile OpenCV bağımlılığı yok — saf C.
4. **Önce simüle et, sonra göm**: Algoritmaları masaüstünde doğrula, sonra ESP32'ye portla.

## Bağımlılıklar (Host Simülasyon)

- C11 derleyici (gcc/clang)
- SDL2 (görselleştirme, opsiyonel — ASCII fallback var)
- v4l2 (Linux) veya AVFoundation (macOS)

## Lisans

MIT — Tiny R Labs

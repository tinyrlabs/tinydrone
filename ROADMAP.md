# tinydrone ROADMAP

> Son güncelleme: 8 Ağustos 2026

## Faz 1: Desktop Simülasyon

### 1.1 Temel İskelet
- [ ] Makefile (host build, `make sim`)
- [ ] main.c (argparse, ana döngü)
- [ ] capture.c/h (webcam yakalama — v4l2 + AVFoundation)
- [ ] detector.c/h (renk eşikleme + kontür)
- [ ] display.c/h (ASCII görselleştirme)

### 1.2 Renk Tabanlı Tespit
- [ ] RGB → HSV dönüşümü
- [ ] Renk maskesi (alt/üst eşik)
- [ ] Morfolojik temizleme (erozyon/genişleme, 3×3 kernel)
- [ ] Bağlı bileşen etiketleme (flood fill)
- [ ] En büyük bileşenin merkezi + bounding box

### 1.3 Takip Algoritması
- [ ] Merkezden sapma vektörü (dx, dy)
- [ ] P-controller (oransal kontrol)
- [ ] Servo açı hesaplama (simülasyonda: ekrana yazdır)
- [ ] Kayıp hedef durumu (timeout + arama pattern'i)

### 1.4 Görselleştirme
- [ ] Terminal ASCII çıktı (renkli, hedef konumu)
- [ ] Opsiyonel SDL2 penceresi
- [ ] FPS sayacı
- [ ] Debug overlay (maske, kontür, merkez işareti)

## Faz 2: ESP32-CAM Firmware

### 2.1 ESP-IDF Kurulumu
- [ ] Toolchain kurulumu
- [ ] ESP32-CAM board config
- [ ] OV2640 kamera driver konfigürasyonu
- [ ] WiFi + HTTP debug endpoint (opsiyonel)

### 2.2 Görüntü İşleme
- [ ] Kamera frame yakalama (JPEG → RGB decode veya direkt YUV)
- [ ] 320×240 → 32×32 downsample (integer arithmetic, no float)
- [ ] RGB → HSV (fixed-point math, `cml_math_lut.h`)
- [ ] Renk maskesi (binary threshold)
- [ ] Pixel-counting ile merkez bulma (tam kontür yerine hafif yaklaşım)

### 2.3 Servo Kontrol
- [ ] PWM driver (MCPWM veya LEDC)
- [ ] Pan servo (0-180°, yatay)
- [ ] Tilt servo (0-180°, dikey)
- [ ] Soft limitler (mekanik sınır)
- [ ] Smooth hareket (ramping)

### 2.4 Ana Döngü (FreeRTOS)
- [ ] Core 0: Kamera yakalama + ön işleme (30fps hedef)
- [ ] Core 1: Tespit + takip (10fps hedef)
- [ ] Queue-based IPC (frame buffer)
- [ ] Watchdog timer

## Faz 3: tinycml Entegrasyonu

### 3.1 Veri Seti
- [ ] Webcam'den etiketli görüntü toplama (capture.py)
- [ ] Sınıflar: red_ball, orange_cone, blue_square, background
- [ ] Sınıf başına 200+ örnek
- [ ] Augmentasyon (rotation, brightness, blur)

### 3.2 Feature Extraction
- [ ] Renk histogramı (HSV, 8×8×8 = 512 boyut)
- [ ] HOG benzeri basit gradyan (sobel 3×3)
- [ ] Hu moments (şekil tanımlayıcıları)
- [ ] Feature vektörü: 512 + 64 + 7 = ~580 boyut

### 3.3 Model Eğitimi
- [ ] Python/scikit-learn baseline (MLP, 580→64→32→4)
- [ ] tinycml C model eğitimi (opsiyonel — host tarafında tinycml kullanarak)
- [ ] Accuracy hedefi: >%85 (4 sınıf)
- [ ] Confusion matrix, precision/recall analizi

### 3.4 Ağırlık Export
- [ ] export_weights.py: Python model → C header (float array)
- [ ] Quantization opsiyonu: float32 → int8 (ESP32 SRAM için)
- [ ] Model metadata (layer sizes, activations, normalization params)

### 3.5 ESP32 Inference
- [ ] tinycml ESP-IDF component olarak ekleme
- [ ] CML_POOL_SIZE ayarı (~128KB)
- [ ] Feature extraction C port
- [ ] Inference pipeline: frame → features → MLP forward → class
- [ ] Benchmark: inference süresi (ms), RAM kullanımı

## Faz 4: CNN Modülü (tinycml Katkısı)

### 4.1 Conv2D Layer
- [ ] `src/conv2d.c` + `include/conv2d.h`
- [ ] im2col dönüşümü (görüntü → matris sütunları)
- [ ] GEMM ile konvolüsyon (mevcut matrix_matmul kullanarak)
- [ ] Stride, padding, dilation desteği
- [ ] Bias ekleme
- [ ] Pool allocator uyumlu

### 4.2 MaxPool2D Layer
- [ ] `src/pool2d.c` + `include/pool2d.h`
- [ ] 2×2 max pooling (stride 2, padding 0)
- [ ] Output boyut hesaplama

### 4.3 CNN Model API
- [ ] `include/cnn.h`: CNN model struct (Estimator uyumlu)
- [ ] Layer dizisi: Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool → Flatten → Dense → Softmax
- [ ] fit/predict/clone/free (Estimator vtable)
- [ ] Model save/load (binary format)

### 4.4 Python Export
- [ ] PyTorch/Keras CNN → tinycml CNN weight export
- [ ] Katman katman ağırlık + bias + batch norm parametreleri
- [ ] C header generator

### 4.5 ESP32 CNN Inference
- [ ] TinyCNN: 32×32×3 giriş → Conv(8, 3×3) → MaxPool → Conv(16, 3×3) → MaxPool → FC(32) → FC(N)
- [ ] Bellek hesabı: <200KB RAM (ESP32 için uygun)
- [ ] Inference süresi: <500ms hedef

## Riskler ve Pitfallar

| Risk | Olasılık | Etki | Çözüm |
|------|----------|------|-------|
| ESP32 SRAM yetersiz | Orta | Yüksek | Quantization, daha küçük model, feature reduction |
| Kamera gecikmesi | Yüksek | Orta | Düşük çözünürlük, JPEG bypass, double buffering |
| Işık değişimi | Yüksek | Orta | HSV renk uzayı, adaptif eşik, ML robustluk |
| Drone titreşimi | Yüksek | Düşük | Dijital stabilizasyon (ROI crop), moving average filtre |
| tinycml CNN performans | Orta | Düşük | Baseline MLP yeterli olabilir, CNN Faz 4 bonus |

## Oturum Günlüğü

| Tarih | Faz | Ne Yapıldı |
|-------|-----|------------|
| 08.08.2026 | 0 | Proje başlatıldı, README + ROADMAP yazıldı |
| 08.08.2026 | 0 | GitHub repo: https://github.com/tinyrlabs/tinydrone (private) |
| 08.08.2026 | 0 | Simülasyon iskeleti (detector + display + main, C11) |
| 08.08.2026 | 0 | Dataset pipeline: Kaggle + HuggingFace download scripts |
| 08.08.2026 | 1 | CNN modülü: Conv2D + MaxPool2D + CNN API (tinycml) |
| 08.08.2026 | 1 | 14 yeni test, 38 test paketi geçti, 0 warning |
| 08.08.2026 | 1 | Push: tinyrlabs/tinycml (317977b) — ~1400 LOC |

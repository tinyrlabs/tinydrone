# TINYDRONE — Bizim Taraf Geliştirme Akışı (Faz 4.6)

**Kapsam:** Samet'in sorumluluğundaki geliştirmeler (ESP32-CAM hedef tespit modülü)
**Hedef:** Canlı görüntüde 20-40 FPS hedef takibi, sunucudan tamamen bağımsız
**Repo:** tinyrlabs/tinydrone · firmware/ altında

---

## Aşamalar

| # | Aşama | İçerik | Durum | Test Kriteri |
|---|-------|--------|-------|--------------|
| A | **Takip modu** | Sliding window → hedef kilitlenince sadece pencere takibi (1-9 inference/kare) | ✅ | Host: 6/6 PASS — kilitliyken 1 inference (30-60 FPS eşdeğeri) |
| B | **UART offset çıkışı** | GPIO4/5 UART2, `T+035Y-012\n` formatı, 115200 baud | ✅ | Build OK — format rehberde tanımlı, donanımda doğrulanacak |
| C | **Kayıp/kilitleme mantığı** | Güven eşiği, ardışık kayıp sayacı, yeniden kilitleme + sınıf tutarlılığı | ✅ | Sınıf değişimi 3 karede kilit çözme (test 5) |
| D | **ESP-DSP hızlandırma** | Xtensa SIMD ile conv2d dot product (dsps_dp_s8) | ✅ | Build OK, host testler değişmedi — hız ölçümü donanımda |
| E | **Kamera/gerçek donanım testi** | ESP32-CAM'de FPS ölçümü, gerçek saha doğrulaması | ⏳ | Donanımda 20+ FPS takip |
| F | **ESP32-S3 geçişi (ops.)** | Pin uyumlu, SIMD, 2-4x hız | ⏳ | Gerekirse |

## Aşama Detayları

### A. Takip Modu (sliding_window.c + main.c)

```
Durum makinesi:
  TARAMA (tam sliding window, 72 inference)
     │ hedef bulundu (güven ≥ 0.60)
     ▼
  KİLİTLİ (takip)
     │ her karede: önceki konumda 1 inference
     │   güven ≥ 0.60 → hedef orada, konum güncel
     │   güven < 0.60 → çevre arama (3×3, stride 4 → 9 inference)
     │     bulunursa → konum güncelle, kilitli devam
     │     bulunmazsa → lost_count++
     │       lost_count ≥ 3 → TARAMA'ya dön
     ▼
  TARAMA
```

**FPS hesabı:**
- Kilitli + hedef sabit: 1 inference/kare → 25-60 FPS
- Kilitli + hedef hareketli: 1-9 inference/kare → 6-40 FPS
- Kayıp → tam tarama: 0.5-1.5 FPS (nadiren)

### B. UART Çıkışı

```
Format:  T<offset_x>Y<offset_y>\n  (offset -100..+100, 3 hane)
Örnek:   T+035Y-012\n  → hedef sağda 0.35, yukarıda 0.12
Gönderim: her karede (tespit varsa) — FC tarafı okur
Baud:     115200 · GPIO4 (TXD) · GPIO5 (RXD) · UART2
```

### C. Kayıp/Kilitleme

- `SWTracker`: locked, last_x, last_y, lost_count
- Eşik: güven ≥ 0.60 kilit, < 0.40 tam kayıp (histerezis)
- 3 ardışık kayıp kare → tam tarama

### D. ESP-DSP

- `esp_dsp` component (IDF component manager: esp-dsp)
- `dsps_mul_s8` / dot product hızlandırma
- Conv2d inner loop'u ESP-DSP ile değiştir

## Bağımlılıklar

- A → B (offset, tespit konumundan türetilir)
- A → C (kilitleme mantığı A'nın parçası)
- D bağımsız, A'dan sonra
- E donanım gerektirir (ESP32-CAM fiziksel)

## Kabul Kriterleri (Faz 4.6 sonu)

- [ ] Host test: takip modu kilitliyken 20+ FPS eşdeğeri
- [ ] UART çıktısı formatı doğru
- [ ] Kayıp → yeniden kilitleme çalışıyor
- [ ] Firmware build OK (int8 + takip + UART)
- [ ] Tüm commit'ler push'lu, doküman güncel

---
**Başlangıç:** 13 Ağustos 2026 · **Tahmini süre:** 3-4 gün (A-D)

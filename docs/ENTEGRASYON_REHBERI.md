# TINYDRONE — Hedef Tespit Modülü Entegrasyon Rehberi
## Drone Geliştirici Arkadaş İçin Teknik Döküman

**Versiyon:** 1.0 · **Tarih:** 13 Ağustos 2026
**Modül:** tinyrlabs/tinydrone — ESP32-CAM gömülü hedef tespit sistemi
**Hazırlayan:** Tiny R Labs (Samet Yılmaz Temel)

---

## Bu Doküman Kimin İçin?

Drone uçuş platformunu geliştirecek teknik arkadaş için hazırlanmıştır.
Uçuş platformu (çerçeve, motor, ESC, uçuş kontrolcü, batarya, RC) **tamamen
onun kontrolündedir**. Bu doküman, bizim tarafımız olan **hedef tespit
modülünün** (ESP32-CAM) ne olduğunu, ne gerektirdiğini ve drone ile nasıl
entegre edileceğini tanımlar.

**Sorumluluk ayrımı:**

| Taraf | Sorumluluk |
|-------|-----------|
| Biz (Tiny R Labs) | Kamera + yapay zeka (CNN) + hedef tespiti + gimbal servo kontrolü |
| Arkadaş (Drone geliştirici) | Uçuş platformu: çerçeve, motor, ESC, FC, batarya, RC, uçuş testi |

---

## İçindekiler

1. Modül Tanıtımı — Ne Yapıyor?
2. Modül Teknik Özellikleri
3. Arkadaşın Sağlaması Gereken Uçuş Platformu
4. Elektriksel Entegrasyon
5. Mekanik Entegrasyon
6. Veri Arayüzü ve Protokol
7. Entegrasyon Test Planı
8. Kabul Kriterleri
9. Yol Haritası ve İletişim

---

## 1. Modül Tanıtımı — Ne Yapıyor?

ESP32-CAM üzerinde çalışan gömülü yapay zeka sistemi:

1. **OV2640 kamera** ile görüntü alır (160×120)
2. **tinycml CNN** (kendi C11 kütüphanemiz, int8 quantize) ile 4 sınıfı tespit eder:
   - `tank` (0)
   - `armored_vehicle` (1)
   - `drone_uav` (2)
   - `background` (3)
3. **Sliding window** ile görüntüde hedefin **konumunu** (bounding box) bulur
4. Hedefi **2 servo ile takip eder** (pan/tilt gimbal)
5. Opsiyonel: hedef offset'ini UART üzerinden dışarı verir (uçuş kontrolcüye)

**Kritik: Modül bağımsız çalışır.** Gimbal takibi için uçuş platformuna
ihtiyaç duymaz. Uçuş platformu ile entegrasyon, otonom takip için bir
**genişletme**dir.

---

## 2. Modül Teknik Özellikleri

### 2.1 Donanım

| Parça | Model | Adet |
|-------|-------|------|
| Mikrodenetleyici | ESP32 (dual-core 240MHz, 520KB SRAM + 4MB PSRAM) | 1 |
| Kamera | OV2640 (2MP, modülde entegre) | 1 |
| Servo (pan) | SG90 mikro servo | 1 |
| Servo (tilt) | SG90 mikro servo | 1 |
| Gimbal | 3D baskı braket (STL bizden temin edilir) | 1 |

### 2.2 Ağırlık ve Güç

| Öğe | Değer |
|-----|-------|
| ESP32-CAM + kamera | ~10 g |
| SG90 servo (×2) | ~9 g × 2 = 18 g |
| Gimbal braket | ~10-20 g |
| **Modül toplam ağırlık** | **~40-50 g** |
| ESP32-CAM akım | ~180-300 mA @ 5V |
| Servo akım (peak) | ~200-700 mA @ 5V (hareket anında) |
| **Toplam güç ihtiyacı** | **5V, ~1A peak** |

### 2.3 Yazılım / Yapay Zeka

| Özellik | Değer |
|---------|-------|
| Model | tinycml CNN, ~136K parametre |
| Quantization | int8 (0.41 MB flash) |
| Doğruluk | int8: %85.93 (float ile uyum %99.62) |
| Sınıflar | tank, armored_vehicle, drone_uav, background |
| Tespit yöntemi | Sliding window (coarse stride 16 → fine stride 4) |
| FPS | Tek kare: 4-8 FPS · Sliding window: 0.5-1.5 FPS |
| Gecikme | ~165 ms / kare (host), ESP32'de tahmini 0.7-2 s |
| Güven eşiği | 0.60 (altı hedef sayılmaz) |

### 2.4 Elektriksel Giriş/Çıkış

| Sinyal | Pin | Açıklama |
|--------|-----|----------|
| Güç 5V | 5V pini | BEC'ten (batarya → regülatör) |
| Toprak | GND | Ortak toprak |
| Servo pan | GPIO14 | 50Hz PWM, 500-2500µs |
| Servo tilt | GPIO15 | 50Hz PWM, 500-2500µs |
| Debug serial | GPIO1 (TXD), GPIO3 (RXD) | 115200 baud, loglar |
| **UART2 (opsiyonel)** | GPIO4 (TXD), GPIO5 (RXD) | Hedef offset çıkışı → FC |

---

## 3. Arkadaşın Sağlaması Gereken Uçuş Platformu

Uçuş platformu tamamen arkadaşın seçimine bağlıdır. Önerilen minimum
gereksinimler (modülü taşıyacak kapasite):

### 3.1 Önerilen Konfigürasyon

| Bileşen | Öneri | Neden |
|---------|-------|-------|
| Çerçeve | 250-450 mm (F450 sınıfı) | Stabilite + modül taşıma |
| Uçuş kontrolcü | F4/F7 (Betaflight) veya ArduPilot | Standart, iyi dokümante |
| Motor | 2212 920KV × 4 | 10" pervane, 500-800g kaldırma |
| ESC | 30A × 4 (BLHeli_S/32) | Motor akımına yeter |
| Pervane | 10" × 2 CW + 2 CCW | F450 standardı |
| Batarya | 3S 2200mAh LiPo (30C+) | 10-15 dk uçuş |
| RC | FS-i6 + X6B (veya ELRS) | 6+ kanal, SBUS |
| BEC | 5V 3A | Modül + servo gücü (ayrı BEC önerilir) |

### 3.2 Uçuş Platformu Gereksinimleri

1. **Kaldırma kuvveti:** Modül (~50g) + gimbal dahil toplam yük 500g'ı
   geçmemeli (F450'nin ~800g-1kg taşıma kapasitesi vardır)
2. **Güç dağıtımı:** Modüle **5V/1A** ayrı BEC çıkışı (FC'nin BEC'i
   yetersiz kalabilir — servo peak akımı)
3. **Montaj noktası:** Gövde ön üst kısmında titreşim sönümleyicili
   platform (modül için ~4 cm × 4 cm alan)
4. **Ağırlık merkezi:** Modül öne monte edilirse bataryayı arkaya
   kaydırarak dengeleyin (CG'yi koruyun)
5. **RC kanalları:** En az 4 (throttle, roll, pitch, yaw) + mod (arm/disarm)

### 3.3 Titreşim Yönetimi (Kritik!)

- Kamera görüntüsü titreşimden etkilenir → tespit kalitesi düşer
- Modül montajında **kauçuk grommet / sünger bant** kullanın
- Pervane dengesizliği (unbalanced) titreşimin ana kaynağıdır
- FPV'de jello (titreşim çizgisi) görürseniz motor mili/pervane dengesini kontrol edin

---

## 4. Elektriksel Entegrasyon

### 4.1 Bağlantı Şeması

```
Batarya (3S LiPo 11.1V)
   │
   ├──► PDB ──► ESC×4 ──► Motor×4
   │
   ├──► FC (VBAT, motor sinyalleri)
   │
   └──► BEC 5V 3A (ayrı regülatör) ──► ESP32-CAM 5V
                                      ├──► Servo pan (VCC)
                                      └──► Servo tilt (VCC)
   (GND: tüm sistem ortak)
```

### 4.2 Pin Bağlantı Tablosu (Arkadaş Tarafı)

| ESP32-CAM Pini | Bağlanacak Yer | Kablo Rengi (öneri) |
|----------------|----------------|---------------------|
| 5V | BEC 5V çıkışı | Kırmızı |
| GND | Ortak GND | Siyah |
| GPIO14 (servo pan) | Pan servo sinyal | Turuncu/Sarı |
| GPIO15 (servo tilt) | Tilt servo sinyal | Turuncu/Sarı |
| GPIO4 (UART2 TX, ops.) | FC UART RX | Beyaz |
| GPIO5 (UART2 RX, ops.) | FC UART TX | Yeşil |

> **DİKKAT:** ESP32-CAM'e **doğrudan batarya voltajı (11.1V) BAĞLAMAYIN**.
> 5V BEC şarttır. ESP32-CAM'in 5V pini 5V'tan fazlasını kaldırmaz (bazı
> modüllerde dahili regülatör vardır ama riske atmayın).

### 4.3 Güç Bütçesi (Uçuş Platformu Tarafı)

| Tüketici | Akım | Kaynak |
|----------|------|--------|
| ESP32-CAM | ~250 mA @ 5V | BEC |
| 2× SG90 servo (peak) | ~700 mA @ 5V | BEC |
| FC + alıcı | ~300 mA @ 5V | FC BEC veya ayrı |
| 4× motor (hover) | ~12-20A @ 11.1V | PDB (batarya direkt) |
| **Toplam hover akımı** | **~14-22A** | 2200mAh 30C → 66A peak kapasite ✓ |

---

## 5. Mekanik Entegrasyon

### 5.1 Gimbal Montajı

1. **Pan servo** → drone gövdesine, şaftı yukarı bakacak şekilde
2. **Tilt servo** → pan servo şaftına 90° açıyla
3. **ESP32-CAM** → tilt servo koluna, kamera lensi ileri bakacak şekilde
4. Kablo payı bırakın (servo dönüşü kabloyu sıkıştırmasın)

### 5.2 Konum ve Yön

- Kamera **ileri-doğru** bakmalı (uçuş yönü ile aynı)
- ESP32-CAM'in üzerindeki kamera, board kenarına paralel takılır
- Lens koruması: drone inişlerinde lens çizilmesin diye koruyucu halka
- Anten yönü: WiFi/anten drone gövdesinden uzak, metal kısımlara değmesin

### 5.3 Ağırlık Merkezi (CG)

- Modül öne takılırsa → batarya arka tarafta (CG'yi çerçeve merkezine çekin)
- Parmak testi: drone'u 2 parmakla motor şaftlarından tutun, yatay dursun
- CG bozuksa: hover testinde drone geriye/öne eğilir

---

## 6. Veri Arayüzü ve Protokol

### 6.1 Şu Anki Durum (Bağımsız Mod)

Modül **kendi başına** çalışır: hedefi bulur, gimbal ile takip eder.
Uçuş platformuna veri göndermez. Bu mod için UART bağlantısı GEREKMEZ.

### 6.2 Otonom Takip Modu (Entegrasyon — ileri adım)

Modül, her tespitte UART üzerinden **hedef offset** gönderir:

```
Format:  T<offset_x_int>Y<offset_y_int>\n
Örnek:   T+035Y-012\n   → hedef sağda 0.35, yukarıda 0.12 (normalize -1..1)

Alan      | Aralık     | Anlam
----------|------------|----------------------
offset_x  | -100..+100 | + = hedef sağda, - = solda
offset_y  | -100..+100 | + = hedef yukarıda, - = aşağıda
T+000Y+000\n          → hedef merkezde (takip tamam)
Hedef yoksa: 5 saniye sessizlik
```

**FC tarafında önerilen kullanım:**
- `offset_x > +20` → sağa yaw (dönüş hızı offset ile orantılı)
- `offset_x < -20` → sola yaw
- `|offset_x| < 20` → yaw sabit (hedef merkezde)
- İlk sürümde **sadece yaw** takibi önerilir (pitch = irtifa değişimi riskli)

### 6.3 Seri Ayarları (UART2)

```
Baud:     115200
Data:     8N1 (8 bit, no parity, 1 stop)
Akış:     Modül → FC (tek yön, TX only)
```

### 6.4 Alternatif Protokoller (ileri)

| Protokol | Durum | Kullanım |
|----------|-------|----------|
| Basit UART (yukarıdaki) | Önerilen MVP | Hızlı entegrasyon |
| MAVLink | ArduPilot uyumlu | `SET_ATTITUDE_TARGET` ile yaw |
| CRSF | Betaflight + ELRS | RC kanalı üzerinden |

---

## 7. Entegrasyon Test Planı

### Aşama 1: Modül Yalnız Test (bizim taraf, masaüstü)
- [ ] ESP32-CAM flash edildi, serial log çıkıyor
- [ ] Kamera görüntüsü alınıyor (QQVGA)
- [ ] Hedef görseli önünde → `HEDEF: sınıf=X güven=Y` logu
- [ ] Servolar GPIO14/15'ten hareket ediyor

### Aşama 2: Modül + Gimbal (masaüstü, arkadaşla ortak)
- [ ] Gimbal mekanik olarak monte edildi
- [ ] Servo pan/tilt doğru yönde
- [ ] Hedef görseli çevirince gimbal takip ediyor
- [ ] Titreşim yok (elle sallayınca görüntü stabil)

### Aşama 3: Uçuş Platformu (arkadaş, pervanesiz)
- [ ] Motorlar doğru yönde dönüyor
- [ ] FC stabilizasyon çalışıyor (ellerle eğilince düzeliyor)
- [ ] RC komutları doğru

### Aşama 4: Entegre Yer Testi (pervanesiz)
- [ ] Modül + drone birlikte güç alıyor (5V BEC)
- [ ] Servo ve motor aynı anda çalışıyor, güç kesintisi yok
- [ ] UART çıktısı (opsiyonel) FC'de okunuyor
- [ ] 5 dk birlikte çalışma — ısınma yok

### Aşama 5: İlk Uçuş (arkadaş kontrolünde)
- [ ] Kablolu hover (0.5-1 m)
- [ ] Serbest uçuş (açık alan)
- [ ] Yerdeki hedefe gimbal takibi (havada)
- [ ] **Otonom yaw takibi** (UART entegre ise)

---

## 8. Kabul Kriterleri

Modül entegrasyonu **tamamlandı** sayılır:

1. ESP32-CAM drone'a monte edildi, uçuş sırasında güç stabil
2. Gimbal pan/tilt serbest çalışıyor, kablolar sıkışmıyor
3. Yerden 10-20 m yükseklikte hedef görseli tespit ediliyor (güven ≥ 0.60)
4. Gimbal hedefi merkezde tutuyor (±20° içinde)
5. Uçuş sırasında titreşim kaynaklı tespit kaybı yok (veya minimal)
6. (İleri) UART offset çıktısı FC'de doğru okunuyor

**Tespit sınırlamaları (bilinmeli):**
- 32×32 çözünürlük → küçük/uzak hedefler kaçabilir (hedef ~1m poster,
  10-20m mesafeden tespit hedefi)
- Işık azlığı (akşam) tespiti düşürür
- Hızlı manevra sırasında 0.5-1.5 FPS yetersiz kalabilir → hedef kaybı
  sonrası yeni tarama (gimbal merkeze döner)

---

## 9. Yol Haritası ve İletişim

### Önerilen Zaman Çizelgesi

| Hafta | Görev | Sorumlu |
|-------|-------|---------|
| 1-2 | Modül hazır (biz) + uçuş platformu parçaları (arkadaş) | Herkes kendi |
| 3 | Montaj (gimbal + platform) | Ortak |
| 4 | Yer testleri + entegrasyon | Ortak |
| 5 | İlk uçuş + gimbal takip testi | Arkadaş + biz (gözlem) |
| 6+ | Otonom yaw takibi (UART) | Ortak |

### Kaynaklar

| Kaynak | Konum |
|--------|-------|
| Repo | `github.com/tinyrlabs/tinydrone` (private — erişim için Samet) |
| Firmware kaynağı | `firmware/main/` (main.c, camera.c, inference_int8.c, tracker.c, sliding_window.c) |
| Model | `firmware/main/tinydrone_model_int8.h` (int8, 456KB) |
| Binary | `firmware/build/tinydrone.bin` (0.41 MB, flash'a hazır) |
| Bu rehber | `docs/ENTEGRASYON_REHBERI.md` |
| Genel rehber | `docs/DRONE_REHBERI.md` (montaj + uçuş detayları) |
| ESP-IDF | v5.3.2 (kurulum: `docs/DRONE_REHBERI.md` bölüm 7) |

### Flash Komutu (arkadaş için)

```bash
# ESP-IDF kuruluysa:
cd firmware
idf.py -p /dev/ttyUSB0 flash monitor
# (ESP32-CAM: IO0 pinini GND'ye kısalt, güç ver, flash et)
```

### İletişim

- Sorular: Samet Yılmaz Temel
- Değişiklik talepleri (pin, protokol, FPS): repo issue/PR üzerinden
- **Uçuş testleri yalnızca arkadaşın onayı ve kontrolünde yapılır**

---

## Ek A: Modül Çıktı Formatı (Serial Log Örneği)

```
tinydrone başlatılıyor...
Kamera OK (QQVGA RGB565)
tinycml CNN modeli yüklendi (int8)
Servolar OK
tarama: hedef yok (frame=20)
HEDEF: sınıf=2 güven=0.98 bbox=(64,32) off=(0.00,-0.20)
HEDEF: sınıf=2 güven=0.95 bbox=(80,32) off=(0.12,-0.20)
tarama: hedef yok (frame=80)
FPS: 3
```

## Ek B: Sık Sorulan Sorular (Arkadaştan Gelebilecek)

**S: Modülün ağırlığı ne?** ~40-50g (ESP32-CAM 10g + 2 servo 18g + gimbal).

**S: Neden ayrı BEC?** SG90 servo peak'te ~700mA çeker; FC BEC'i genelde
1A'dır ve FC + alıcı + servo birlikte yetersiz kalabilir. 3A BEC güvenli.

**S: FPS neden düşük?** Her karede 72 CNN inference yapılıyor (sliding
window). İsterseniz stride artırıp 3-4 FPS'e çıkabiliriz — tespit menzili
biraz düşer.

**S: Yağmur/soğukta çalışır mı?** ESP32-CAM IP korumasız — yağmurda uçmayın.
Çalışma sıcaklığı -10°C ila +50°C.

**S: Modülü başka FC'ye takabilir miyim?** Evet — UART çıkışı protokol
bazlı, FC markasından bağımsız. ArduPilot/Betaflight/INAV hepsi uyumlu
(basit UART formatında).

**S: Hedef tespiti gece çalışır mı?** OV2640 IR filtreli — gece çalışmaz.
IR-cut-off versiyon + IR spot ile deneysel gece modu mümkün (ileri adım).

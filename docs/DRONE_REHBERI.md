# TINYDRONE — Gömülü Otonom Hedef Tespit Drone'u
## Teknik Geliştirme, Montaj ve Uçuş Rehberi

**Versiyon:** 1.0 · **Tarih:** 13 Ağustos 2026
**Proje:** tinyrlabs/tinydrone · **Repo:** github.com/tinyrlabs/tinydrone (private)

---

## İçindekiler

1. Mevcut Durum Analizi
2. Sistem Mimarisi
3. Drone Gereksinimleri ve Özellikleri
4. Parça Listesi (BOM)
5. Montaj Aşamaları
6. Elektronik Bağlantı Şeması
7. Firmware Flash ve İlk Test
8. Kalibrasyon Prosedürü
9. Uçuş Testi Aşamaları
10. Otonom Takip Entegrasyonu
11. Güvenlik ve Yasal Uyarılar
12. Bütçe Tablosu
13. Yol Haritası ve Zaman Çizelgesi
14. Sık Karşılaşılan Sorunlar ve Çözümleri

---

## 1. Mevcut Durum Analizi

### 1.1 Tamamlanan Bileşenler

| Bileşen | Durum | Detay |
|---------|-------|-------|
| Veri seti | ✅ Tamam | 24.602 gerçek görsel, 4 sınıf (tank, armored, drone, background) |
| tinycml CNN | ✅ Tamam | Conv2D + MaxPool2D + backprop, 38 test paketi |
| Eğitim (C) | ✅ Tamam | Loss 0.64 → 0.20, tamamen kendi kütüphanesiyle |
| Model (float) | ✅ Tamam | %89.54 test doğruluğu, 136K parametre |
| Model (int8) | ✅ Tamam | %85.93, uyum %99.62, 0.41 MB |
| Sliding window | ✅ Tamam | Coarse-to-fine, bbox + servo offset |
| Firmware | ✅ Tamam | Build OK, flash'a hazır |

### 1.2 Mevcut Firmware Yetenekleri

- **Kamera:** OV2640, QQVGA 160×120 RGB565
- **Tespit:** 32×32 pencere, coarse (stride 16) → fine (stride 4), 72 inference/kare
- **Çıktı:** Sınıf + güven + bounding box + normalize offset (-1..1)
- **Takip:** 2 servo (pan/tilt) — GPIO14/15, LEDC PWM
- **Hız:** int8 ile tahmini 4-8 FPS (tek kare), 0.5-1.5 FPS (sliding window)
- **Boyut:** 0.41 MB binary (4MB flash'ta %10)

### 1.3 Eksikler / Yapılacaklar

| Eksik | Önem | Not |
|-------|------|-----|
| Gerçek donanım testi | Yüksek | Firmware derlendi, donanımda doğrulanmadı |
| Uçuş kontrolcü entegrasyonu | Yüksek | ESP32-CAM tek başına uçuramaz |
| FPS optimizasyonu | Orta | Stride 32 + sıcak başlangıç → 3-4 FPS |
| Motor kontrolü | Yüksek | ESC + motor sürücü modülü gerekli |
| Telemetri | Düşük | Opsiyonel, MAVLink/CRSF |

---

## 2. Sistem Mimarisi

```
┌─────────────────────────────────────────────────────┐
│                   DRONE (uçan platform)              │
│                                                     │
│  ┌──────────────┐    ┌──────────────────────────┐   │
│  │  UÇUŞ KONTROL │    │   ESP32-CAM (GÖZ)        │   │
│  │  (FC)         │◄───│   - OV2640 kamera        │   │
│  │  - IMU/GPS    │UART│   - tinycml CNN (int8)   │   │
│  │  - Motor PID  │    │   - Sliding window       │   │
│  │  - Stabilizasyon│   │   - Hedef offset çıkışı │   │
│  └──────┬───────┘    └──────────────────────────┘   │
│         │ ESC×4                                     │
│  ┌──────▼───────┐    ┌──────────────────────────┐   │
│  │  Motorlar    │    │   SERVO GIMBAL           │   │
│  │  (BLDC)      │    │   - Pan (yatay) GPIO14   │   │
│  └──────────────┘    │   - Tilt (dikey) GPIO15  │   │
│                      └──────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### 2.1 Veri Akışı

1. **Kamera** → RGB565 160×120 → RGB888
2. **Sliding window** → 32×32 patch'ler → int8 CNN
3. **CNN** → 4 sınıf olasılığı → argmax + güven
4. **Tespit kararı** → güven ≥ 0.60 → hedef bbox
5. **Servo takip** → bbox merkezi → offset → pan/tilt PWM
6. **(Otonom uçuş)** → offset → UART → FC → yaw/pitch komutu

---

## 3. Drone Gereksinimleri ve Özellikleri

### 3.1 Uçuş Performansı

| Özellik | Değer | Gerekçe |
|---------|-------|---------|
| Çerçeve | 250-450 mm (F450 sınıfı) | Stabilite + kamera taşıma |
| Uçuş süresi | 10-15 dk | 3S 2200mAh LiPo |
| Yük kapasitesi | 500-800 g | ESP32-CAM (~30g) + gimbal (~50g) |
| Menzil | 500-1000 m | 2.4GHz RC + video |
| Maks hız | 15-25 km/s | Tespit hızına uygun (1-2 FPS) |

### 3.2 Gerekli Özellikler (öncelik sırasıyla)

1. **Stabilizasyon (zorunlu)** — GPS'siz hover için barometre + IMU
2. **FPV video (zorunlu)** — gerçek zamanlı görüntü (5.8GHz TX veya WiFi)
3. **GPS (yüksek)** — "hedefe dön" ve rota takibi
4. **Hedef takibi (mevcut)** — ESP32-CAM + gimbal
5. **Telemetri (opsiyonel)** — uzun menzil için MAVLink
6. **Otonom görev (ileri)** — waypoint + hedef tespit birleşimi

### 3.3 Elektronik Gereksinimler

| Bileşen | Gerekli | Not |
|---------|---------|-----|
| Uçuş kontrolcü | Evet | Betaflight F4/F7 veya ArduPilot |
| ESC | 4× 30A | BLHeli_S veya BLHeli_32 |
| Motor | 4× 2212 920KV | 10-11" pervane |
| Pervane | 10" × 2 (CW+CCW) | Kendinden sıkmalı önerilir |
| Batarya | 3S 2200mAh LiPo | 30C+ |
| Güç dağıtım | PDB veya FC'ye entegre | 4S destekli |
| RC alıcı | ELRS/Crossfire veya 2.4GHz | FS-i6 + X6B gibi |
| Buzzer | Evet | Kaybolma durumunda |
| LED | Evet | Yön ve durum göstergesi |

---

## 4. Parça Listesi (BOM)

### 4.1 Hedef Tespit Bileşenleri (mevcut sistem)

| Parça | Adet | Tahmini Fiyat (₺) |
|-------|------|-------------------|
| ESP32-CAM (AI-Thinker) | 1 | 300-400 |
| OV2640 kamera modülü | 1 | (ESP32-CAM'de dahil) |
| SG90 mikro servo | 2 | 60-100 |
| 3D baskılı gimbal braketi | 1 | 50-100 |
| Jumper kablo + breadboard | 1 set | 50 |
| 5V BEC regülatör | 1 | 40-60 |
| **Alt toplam** | | **500-700** |

### 4.2 Uçuş Platformu (drone gövdesi)

| Parça | Adet | Tahmini Fiyat (₺) |
|-------|------|-------------------|
| F450 çerçeve kiti | 1 | 400-600 |
| F4/F7 uçuş kontrolcü (Betaflight) | 1 | 400-700 |
| 30A ESC (4'ü 1 arada) | 1 | 300-500 |
| 2212 920KV motor | 4 | 400-700 |
| 10" pervane seti | 2 | 100-150 |
| 3S 2200mAh LiPo | 2 | 400-600 |
| LiPo şarj cihazı | 1 | 300-500 |
| FS-i6 RC seti (alıcı dahil) | 1 | 800-1000 |
| PDB / güç kablosu seti | 1 | 100 |
| **Alt toplam** | | **3200-4800** |

### 4.3 Opsiyonel

| Parça | Fiyat (₺) |
|-------|-----------|
| GPS modülü (BN-880) | 200-300 |
| 5.8GHz FPV TX + kamera | 400-600 |
| FPV gözlük/monitör | 1000-3000 |
| Telemetri (MAVLink 433MHz) | 300-500 |

### 4.4 Toplam Bütçe

- **Temel (hedef tespit + uçuş):** ~3.700-5.500 ₺
- **FPV dahil:** ~5.100-9.100 ₺
- **Tam otonom (GPS + telemetri):** ~5.600-9.900 ₺

---

## 5. Montaj Aşamaları

### Aşama 1: Çerçeve Montajı (1-2 saat)

1. Çerçeve kollarını gövde plakasına vidalayın (çapraz düzen)
2. PDB veya güç dağıtımını alt plakaya sabitleyin
3. ESC'leri kolların içine yerleştirin (ısı shrink ile koruyun)
4. Motorları kol uçlarına monte edin:
   - **CW motorlar:** ön-sağ ve arka-sol
   - **CCW motorlar:** ön-sol ve arka-sağ
5. Motor kablolarını ESC'lere bağlayın (3 faz, herhangi bir sıra — yön yazılımdan düzeltilir)

### Aşama 2: Uçuş Kontrolcü (1-2 saat)

1. FC'yi çerçevenin merkezine, titreşim sönümleyici (grommet) ile monte edin
2. ESC sinyal kablolarını FC'ye bağlayın:
   - Motor 1 → ön-sağ, Motor 2 → arka-sol, Motor 3 → arka-sağ, Motor 4 → ön-sol
   - (Betaflight'ın standart M1-M4 ataması)
3. RC alıcıyı FC'ye bağlayın (SBUS: alıcı RX → FC TX, ortak GND)
4. Buzzer + LED bağlayın
5. Batarya kablosunu PDB'ye lehimleyin (XT60 konnektör)
6. FC'yi USB ile PC'ye bağlayıp Betaflight Configurator ile tanıtın

### Aşama 3: ESP32-CAM + Gimbal (1-2 saat)

1. Pan (yatay) servoyu drone'un ön üstüne sabitleyin
2. Tilt (dikey) servoyu pan servo üzerine 90° monte edin
3. ESP32-CAM'i gimbalın en üstüne, kamera ileri bakacak şekilde takın
4. Kablolama:
   - ESP32-CAM 5V → BEC çıkışı (bataryadan)
   - ESP32-CAM GND → ortak GND
   - Pan servo sinyal → GPIO14
   - Tilt servo sinyal → GPIO15
   - Servo güç → BEC 5V
5. ESP32-CAM'i FTDI/USB-UART ile programlama moduna geçirin (IO0 → GND, sonra güç)

### Aşama 4: Kablo Yönetimi (30 dk)

1. Tüm kabloları spiral/temiz çekme ile sabitleyin
2. Vidaların gevşek olmadığını kontrol edin
3. ESC ve FC üzerinde ısı shrink veya koruyucu kaplama
4. Batarya kayışı ile bataryayı alt plakaya sabitleyin

### Aşama 5: Kontrol Listesi (Montaj Sonrası)

- [ ] Tüm vidalar sıkı
- [ ] Motorlar serbest dönüyor (pervane takılı değilken)
- [ ] FC USB ile bağlanıyor
- [ ] Betaflight'te IMU yönü doğru
- [ ] ESC kalibrasyonu yapıldı
- [ ] RC alıcı bağlantısı (SBUS) çalışıyor
- [ ] ESP32-CAM flash edildi ve serial çıktı veriyor
- [ ] Servolar GPIO14/15'ten hareket ediyor

---

## 6. Elektronik Bağlantı Şeması

### 6.1 Güç Dağıtımı

```
Batarya (3S LiPo, XT60)
   │
   ├──► PDB ──► ESC×4 ──► Motor×4
   │
   ├──► BEC 5V ──► ESP32-CAM (5V, GND)
   │              └──► Servo×2 (5V, GND)
   │
   └──► FC (batarya voltajı, VBAT sensör)
```

### 6.2 ESP32-CAM Pin Kullanımı

| GPIO | Fonksiyon | Bağlantı |
|------|-----------|----------|
| GPIO14 | Servo pan (LEDC PWM, 50Hz) | Pan servo sinyal |
| GPIO15 | Servo tilt (LEDC PWM, 50Hz) | Tilt servo sinyal |
| GPIO0 | Programlama modu (boot) | FTDI DTR/RTS veya buton |
| GPIO1 (U0TXD) | Debug serial | FTDI RX |
| GPIO3 (U0RXD) | Debug serial | FTDI TX |
| 5V | Güç | BEC çıkışı |
| GND | Ortak toprak | Tüm sistem |

> **NOT:** ESP32-CAM'de kamera pinleri (GPIO32-39 vb.) OV2640'a ayrılmıştır.
> GPIO14/15 servo için serbesttir (camera pin listesinde değil).

### 6.3 UART (Otonom uçuş için — ileri adım)

```
ESP32-CAM (U2TXD GPIO4, U2RXD GPIO5)
   │
   └──► FC (UART RX/TX) — MAVLink veya CRSF protokolü
```

---

## 7. Firmware Flash ve İlk Test

### 7.1 ESP-IDF Kurulumu (bir kez)

```bash
# Linux/macOS
git clone --recursive -b v5.3.2 https://github.com/espressif/esp-idf.git ~/esp-idf
cd ~/esp-idf && ./install.sh esp32
. ./export.sh
```

### 7.2 Firmware Build ve Flash

```bash
cd ~/projects/tinydrone/firmware
idf.py set-target esp32
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

> **ESP32-CAM flash modu:** IO0 pinini GND'ye kısa devre yap, sonra güç ver.
> `idf.py flash` otomatik reset dener; olmazsa elle IO0-GND yap.

### 7.3 İlk Test (masaüstü)

1. ESP32-CAM'i USB'ye tak
2. `idf.py monitor` ile serial log izle
3. Beklenen çıktı:
```
tinydrone başlatılıyor...
Kamera OK (QQVGA RGB565)
tinycml CNN modeli yüklendi (int8)
Servolar OK
tarama: hedef yok (frame=20)
FPS: 4
```
4. Kameranın önüne tank/drone görseli (kağıt çıktı) tut
5. Beklenen çıktı:
```
HEDEF: sınıf=2 güven=0.98 bbox=(64,32) off=(0.00,-0.20)
```
6. Servo pan/tilt hedefe doğru hareket etmeli

---

## 8. Kalibrasyon Prosedürü

### 8.1 Kamera

1. OV2640 lensini bir duvara (1-2 m) odaklayın
2. `camera_init()` içinde `xclk_freq_hz = 20000000` (20MHz) — stabilite için
3. Renk doğruluğu: beyaz kağıt altında parlaklık kontrolü
4. Görüş açısı: QQVGA 160×120 → yatay ~65°, dikey ~50°

### 8.2 Servo

1. `tracker_init()` PWM ayarları: 50Hz, 500-2500µs
2. Merkez konumu: `ledc_set_duty` 1500µs → servo 90° olmalı
3. Pan aralığı: 0-180° (2.5ms max)
4. Tilt aralığı: 0-90° (gimbal kısıtlıysa 45-135°)

### 8.3 Uçuş Kontrolcü (Betaflight)

1. **Accelerometer kalibrasyonu:** FC'yi düz zemine koy, Betaflight'ta "Calibrate Accel"
2. **ESC kalibrasyonu:** motorlar maksimuma al, güç ver, beep'i bekle, minimuma indir
3. **Motor yönü:** motor testi ekranında her motorun doğru yönde döndüğünü doğrula (ters ise 2 faz değiştir)
4. **RC komutları:** RC verici ekranında tüm kanallar doğru yön ve aralıkta
5. **IMU yönü:** FC'yi eğdiğinde sanal drone doğru eğilsin
6. **PID:** varsayılanla başla (Betaflight 4.x default)

---

## 9. Uçuş Testi Aşamaları

### Aşama A: Yer Testi (pervanesiz)

1. Batarya tak, verici aç
2. Motorlar kol ile dönsün (pervane yok)
3. Gimbal servo testi: hedef görseli çevir → servo takip etsin
4. Tüm sistem 5 dk çalışsın, ısınma/koku kontrolü

### Aşama B: Kablolu Hover (ilk uçuş)

1. Pervaneleri tak (yön doğru!)
2. Açık alan, rüzgarsız gün
3. Gazı %30'a yavaşça çıkar
4. Hover yüksekliği: 0.5-1 m
5. Trim: sağa-sola-öne-arkaya kayma varsa trim düzelt

### Aşama C: Serbest Uçuş

1. 5-10 m yarıçaplı daireler
2. Yaw (dönüş) komutları
3. Alçak irtifa geçişleri
4. Batarya bitiş alarmına dikkat (3.7V/hücre)

### Aşama D: Hedef Takip Testi

1. Yere hedef görseli (1×1 m kumaş/poster) koy
2. Drone 10-20 m yükseklikte hover
3. Kamera hedefi bulsun (serial log: HEDEF)
4. Gimbal hedefi merkezde tutsun
5. Drone'u elle yana kaydır → gimbal hedefi takip etsin

### Aşama E: Otonom Takip (ileri seviye)

1. FC'ye UART üzerinden offset gönder (MAVLink)
2. FC yaw komutu: offset_x × maks_yaw_rate
3. Drone hedefe doğru dönsün
4. Yükseklik sabit, sadece yaw takibi

---

## 10. Otonom Takip Entegrasyonu

### 10.1 Protokol Seçimi

| Protokol | Kullanım | Avantaj |
|----------|----------|---------|
| MAVLink | ArduPilot | Endüstri standardı, geniş destek |
| CRSF | Betaflight (ELRS) | Hafif, hızlı |
| Basit UART (özel) | Herhangi | En basit MVP |

### 10.2 Basit UART MVP (önerilen ilk adım)

```c
// ESP32-CAM tarafı (firmware'e eklenecek)
// Her tespitte: "T<offset_x_int>Y<offset_y_int>\n" gönder
// Örnek: T+035Y-012  → sağa 0.35, yukarı 0.12

// FC tarafı: her satırı oku, yaw = offset_x * MAX_YAW_RATE
```

### 10.3 MAVLink (ileri)

- `MAV_CMD_SET_MESSAGE_INTERVAL` ile hedef konumu yayınla
- FC'de `GUIDED` modunda yaw komutu
- ArduPilot: `MAV_CMD_CONDITION_YAW`

---

## 11. Güvenlik ve Yasal Uyarılar

### 11.1 Fiziksel Güvenlik

- **Pervane koruyucu** ilk uçuşlarda zorunlu
- **Göz koruması** — pervaneler 10" ve keskin
- **Yangın riski:** LiPo şişmişse kullanma, kum kovasında izole et
- **İlk uçuşlar:** açık alan, insansız, çitli saha

### 11.2 Yasal (Türkiye — SHT-İHA)

- **500 g üstü drone:** Sivil Havacılık Genel Müdürlüğü'ne kayıt zorunlu
- **Uçuş izni:** yerleşim yerleri, havalimanı çevresi, sınır bölgeleri YASAK
- **Görüş hattı:** uçuş her zaman gözle görülebilir olmalı
- **Yükseklik:** 120 m üzeri yasak (kontrolsüz hava sahası)
- **Kamera/kayıt:** kişisel verilerin korunması kanunu (KVKK) kapsamında dikkat
- **Askeri hedef tespiti projesi** akademik/deneysel amaçlıdır; saha uygulamaları için ilgili izinler gerekir

### 11.3 Elektronik Güvenlik

- Tüm bağlantıları pervaneler çıkarılmışken test et
- Batarya her zaman fireproof çantada şarj et
- ESP32-CAM 5V'a doğrudan batarya (12.6V) bağlama — BEC şart

---

## 12. Bütçe Tablosu

| Kalem | Temel (₺) | FPV'li (₺) | Tam Otonom (₺) |
|-------|-----------|------------|----------------|
| Hedef tespit (ESP32-CAM + servo + gimbal) | 500-700 | 500-700 | 500-700 |
| F450 çerçeve + motor + ESC + pervane | 1.200-1.950 | 1.200-1.950 | 1.200-1.950 |
| Uçuş kontrolcü (F4/F7) | 400-700 | 400-700 | 400-700 |
| RC seti (FS-i6) | 800-1.000 | 800-1.000 | 800-1.000 |
| Batarya + şarj cihazı | 700-1.100 | 700-1.100 | 700-1.100 |
| GPS | — | — | 200-300 |
| FPV TX + kamera + gözlük | — | 1.500-3.600 | 1.500-3.600 |
| Telemetri | — | — | 300-500 |
| **TOPLAM** | **3.600-5.450** | **5.100-9.050** | **5.600-9.850** |

> Fiyatlar 2026 Türkiye pazarı tahminidir; güncel fiyatlar değişkenlik gösterebilir.

---

## 13. Yol Haritası ve Zaman Çizelgesi

### Hafta 1-2: Donanım Toplama
- Parçaları sipariş et (ESP32-CAM, F450 kiti, FC, RC seti)
- ESP32-CAM'i flash et ve masaüstünde test et

### Hafta 3: Montaj
- Çerçeve + motor + ESC montajı
- FC kurulumu (Betaflight)
- Gimbal + ESP32-CAM montajı

### Hafta 4: Yer Testleri
- ESC kalibrasyonu, motor yönleri
- Servo takip testi (masaüstü)
- ESP32-CAM + kamera entegrasyonu

### Hafta 5: İlk Uçuş
- Kablolu hover
- Serbest uçuş alıştırmaları
- Hedef takip testi (yerdeki hedefe)

### Hafta 6: Otonom Takip
- UART protokolü (ESP32-CAM → FC)
- Yaw takip modu
- Saha testleri

### Hafta 7-8: İyileştirme
- FPS optimizasyonu (stride 32, sıcak başlangıç)
- Telemetri + GPS
- Uzun menzil testleri

---

## 14. Sık Karşılaşılan Sorunlar ve Çözümleri

| Sorun | Belirti | Çözüm |
|-------|---------|-------|
| ESP32-CAM flash olmuyor | `Connecting...` döngüsü | IO0→GND yap, güç ver, sonra flash |
| Kamera görüntüsü bozuk | Yeşil/karışık kare | `xclk_freq_hz` 20MHz'e düşür, PSRAM kontrol |
| Servo titriyor | Sürekli jitter | PWM frekansı 50Hz, güç ayrı BEC'ten |
| Tespit yok | `tarama: hedef yok` | Görseli 1-2 m'den göster, ışık artır |
| Yanlış sınıf | Tank→armored | 32x32'de normal, ikisi de hedef sınıf |
| FPS düşük | <1 FPS | Stride 32'ye çıkar, fine'ı kaldır |
| Drone sağa kayıyor | Trim gerekli | RC trim + FC trim |
| Motor ters dönüyor | Kalkışta takla | 2 faz değiştir veya Betaflight motor yönü |
| Batarya çabuk bitiyor | <5 dk | 4S'e geç veya daha büyük kapasite |
| GPS uydu yok | 0 uydu | Açık alana çık, 1-2 dk bekle |

---

## Ek: Mevcut Proje Durumu (13 Ağustos 2026)

```
Repo:        tinyrlabs/tinydrone (private, 30+ commit, v0.1.0)
Kütüphane:   tinyrlabs/tinycml (CNN modülü: Conv2D/MaxPool + backprop)
Veri seti:   24.602 gerçek görsel, 4 sınıf
Model:       float %89.54 | int8 %85.93 (uyum %99.62)
Firmware:    firmware/build/tinydrone.bin (0.41 MB) — flash'a hazır
Doküman:     Bu rehber + ROADMAP.md + README.md
```

**Sonraki adım:** Donanımı topla → flash et → masaüstü test → montaj → uçuş.

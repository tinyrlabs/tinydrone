# TINYDRONE — Gazebo Tabanlı Simülasyon Planı
## Sanal Sahada Otonom Hedef Tespit/Takip Sistemi

**Versiyon:** 1.0 · **Tarih:** 14 Ağustos 2026
**Hedef:** Drone uçurmadan önce — sanal dünyada otonom hedef tespit + takip simülasyonu
**Ek hedef:** Yarışma standında gösterilebilir, etkileyici, canlı demo

---

## 1. Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────────────┐
│                    SİMÜLASYON ORTAMI                             │
│                                                                 │
│  ┌──────────────┐    ┌────────────────────┐    ┌────────────┐   │
│  │    GAZEBO    │    │  ArduPilot SITL    │    │  CNN KÖPRÜ │   │
│  │  (sanal dünya)│◄───│  (uçuş dinamiği)   │◄───│  (bizim    │   │
│  │  - arazi     │MAV │  - IMU/GPS/PID     │MAV │   int8     │   │
│  │  - drone     │Link│  - otonom modlar   │Link│   model)   │   │
│  │  - hedefler  │    │                    │    │            │   │
│  │  - kamera    │    └────────────────────┘    └─────┬──────┘   │
│  └──────┬───────┘                                   │          │
│         │ görüntü (kamera plugin)                    │ tespit   │
│         ▼                                            ▼          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  STAND GÖSTERİM PANELİ                    │    │
│  │  3D dünya + drone | kamera görüntüsü | CNN overlay       │    │
│  │  bbox + sınıf + güven + telemetri (altitude, yaw, FPS)   │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Veri Akışı
1. **Gazebo** sanal dünyayı render eder (arazi + hedef modeller + drone)
2. **Kamera plugin** drone'dan görüntü üretir (160x120'ye ölçeklenir)
3. **CNN köprüsü** görüntüyü int8 modelle işler → sınıf + konum + güven
4. **Tespit** → offset → **MAVLink** → ArduPilot SITL → drone yaw/ileri komutu
5. **Stand paneli** 3 görünümü birleştirir (3D + kamera + overlay)

---

## 2. Platform Kararı (kritik — ARM gerçeği)

| Bileşen | ARM (Oracle) | Not |
|---------|-------------|-----|
| Gazebo Classic (11) | ❌ apt'te yok (Noble kaldırdı) | Alternatif: gz-sim (Harmonic) |
| Gazebo Sim (gz-sim) | ⚠️ denenmeli | arm64 paketleri sınırlı, kurulum riskli |
| Webots | ⚠️ arm64 .deb var ama GUI gerekli | Headless zor, Xvfb ile denenebilir |
| **ArduPilot SITL** | ✅ C++ derlenir, headless | **Güvenli seçim** |
| ROS 2 Jazzy | ⚠️ arm64 var ama ağır (RAM!) | 11GB RAM'de zorlanır |
| Python görsel katman | ✅ | Her ortamda çalışır |

**RAM kısıtı:** 11GB toplam, ~4.9GB kullanımda → simülasyon + ROS 2 birlikte zorlanır.

### Karar: Katmanlı Yaklaşım

**Katman 1 (garantili, bu sunucuda):** ArduPilot SITL (gerçek uçuş fiziği) + Python CNN köprüsü + özel görselleştirici (sahne render'ı bizde) → **her koşulda çalışır**

**Katman 2 (hedef, denenmeli):** Gazebo (gz-sim) dünyası — kurulum POC'u yapılır; başarılıysa Katman 1'in görseli yerine Gazebo geçer

**Katman 3 (stand):** Laptopta (Mac/Win) Gazebo/Webots çalıştırma opsiyonu — sunucuda geliştirilen CNN aynen kullanılır

---

## 3. Faz Planı

### Faz S0: Altyapı Doğrulama (1 gün)
- [ ] ArduPilot SITL arm64 derleme (`git clone ArduPilot/ardupilot`, `./waf configure --board sitl`)
- [ ] gz-sim (Gazebo Harmonic) arm64 kurulum denemesi (`apt install gz-harmonic` veya osrf repos)
- [ ] MAVLink köprüsü: Python `pymavlink` kurulumu, SITL'den telemetri okuma
- **Çıkış:** Hangisi çalışıyor — SITL kesin, Gazebo POC sonucu

### Faz S1: SITL + Kontrol (1-2 gün)
- [ ] ArduPilot SITL headless başlat (ör. `sim_vehicle.py -v ArduCopter --console`)
- [ ] MAVLink ile drone'u arma + takeoff + yaw komutu (Python script)
- [ ] Basit görev: kalk → 10m'ye çık → hedefe yaw dön → geri gel
- **Çıkış:** Python'dan drone kontrolü çalışıyor

### Faz S2: Görselleştirici (2-3 gün)
- [ ] Python 3D sahne (pygame + OpenGL veya basit 2D izometrik): arazi, drone ikonu, hedef modeller
- [ ] Drone kamera görüntüsü simülasyonu: sahnedeki hedefin konumundan perspektif projeksiyon
- [ ] Hedef görselleri gerçek dataset'ten (tank/armored/building) sahneye yerleştir
- [ ] Kamera görüntüsü → 160x120 → int8 CNN → tespit overlay (bbox + sınıf + güven)
- **Çıkış:** Ekranda canlı tespit çalışıyor (SITL'e bağlı değil, bağımsız demo olarak da çalışır)

### Faz S3: Kapalı Döngü (3-5 gün)
- [ ] CNN tespit offset → MAVLink yaw komutu (ArduPilot GUIDED mode)
- [ ] Drone hedefe yönelir; hedef kaybolursa yeniden tarama (tam döngü: tespit→kilitle→takip→kayıp→tara)
- [ ] Farklı senaryolar: sabit hedef, hareketli hedef (görselleştiricide), çoklu hedef
- [ ] FPS ölçümü: tespit döngüsü hızı (hedef: 20+ FPS tespit, SITL 50Hz)
- **Çıkış:** Otonom takip simülasyonu uçtan uca

### Faz S4: Gazebo Entegrasyonu (opsiyonel, 3-5 gün)
- [ ] gz-sim kurulumu başarılıysa: sanal dünya (arazi + tank/bina modelleri)
- [ ] Drone modeli + gimbal + kamera plugin (gz-sim sensor)
- [ ] ArduPilot SITL ↔ Gazebo bağlantısı (gazebo-ardupilot-sitl bridge veya MAVLink)
- [ ] Kamera görüntüsü → CNN köprüsü (Katman 1'den aynen)
- **Çıkış:** Gerçek Gazebo dünyasında otonom takip (başarısızsa Katman 1 görseli yeterli)

### Faz S5: Stand Paketi (2 gün)
- [ ] Tek komut başlatma: `./start_demo.sh` — SITL + görselleştirici + CNN + panel
- [ ] Panel düzeni: 3D sahne | kamera + overlay | telemetri (yükseklik, yaw, FPS, kilit durumu)
- [ ] Ziyaretçi senaryoları:
  - "Hedefi sahneye koy" (klavye: hedef ekle/hareket ettir) → drone bulur + takip eder
  - "Hedefi kaçır" → kayıp → yeniden tarama → bulur
- [ ] ESP32-CAM (gerçek donanım) yan panelde: "aynı model gömülüde de çalışıyor"
- **Çıkış:** Standda tek laptopla çalışan demo

---

## 4. Teknoloji Detayları

### 4.1 ArduPilot SITL
```
git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git ~/ardupilot
cd ardupilot && ./waf configure --board sitl && ./waf copter
# Başlatma:
cd Tools/autotest && ./sim_vehicle.py -v ArduCopter --console --map  # GUI
# Headless:
sim_vehicle.py -v ArduCopter -L Kocaeli --no-mavproxy  # MAVLink TCP 5760
```
- Python: `pip install pymavlink dronekit` (dronekit eski ama basit — veya pymavlink raw)
- Komutlar: `ARM_THROTTLE`, `MAV_CMD_NAV_TAKEOFF`, `SET_ATTITUDE_TARGET` (yaw), `SET_POSITION_TARGET_GLOBAL_INT`

### 4.2 Gazebo (POC'a tabi)
```
# Ubuntu Noble arm64 — denenir:
sudo apt install gz-harmonic   # osrf repo gerekebilir
# Veya: https://packages.osrfoundation.org
```
- Dünya: arazi modeli (heightmap), hedef modeller (STL/DAE — Gazebo model DB'den "military" modelleri veya basit mesh'ler)
- Kamera: `gz-sim sensor camera` plugin → görüntü topic'i → Python bridge

### 4.3 CNN Köprüsü (bizim kod)
- **Mevcut int8 model** (5 sınıf, %90.6) — aynen kullanılır
- C inference'ı Python'dan çağırma: `ctypes` ile `libtinydrone.so` (inference_int8.c'yi paylaşımlı kütüphane yap) VEYA saf Python repro
- Sliding window + takip modu mantığı (C'deki sw_detect_track) → Python'a taşınır veya ctypes ile C'de kalır
- **Karar:** ctypes + C kütüphanesi (performans + aynı kod) — simülasyonda da ESP32'deki kod çalışır (hikaye güçlenir)

### 4.4 Görselleştirici
- **Seçenek A (basit, hızlı):** pygame 2D — üstten görünüm harita + drone + hedef marker + kamera kesit
- **Seçenek B (etkileyici):** Python + moderngl/OpenGL 3D — arazi, drone model, hedef 3D
- **Kamera görüntüsü üretimi:** hedefin gerçek dataset görselini sahne konumuna göre ölçekleyip yerleştir (gerçek görüntü, sentetik değil — dataset'ten)
- Overlay: OpenCV ile bbox + sınıf adı + güven + FPS

---

## 5. Stand Demo Senaryoları

| Senaryo | Ziyaretçi etkisi | Uygulama |
|---------|-----------------|----------|
| 1. Canlı tespit | Fotoğraf göster → anında sınıf + bbox | Webcam veya sahne hedefi |
| 2. Otonom takip | Hedef ekle → drone bulur + kilitler + takip eder | SITL + görselleştirici |
| 3. Hedef kaçırma | Hedefi taşı → kayıp → yeniden tarama → bulur | Aynı, hareketli hedef |
| 4. Gömülü eşleştirme | "Aynı model ESP32'de 0.43MB'da" | Yan panelde donanım + serial log |
| 5. Canlı FPS göstergesi | 30+ FPS tespit sayacı | Panelde sayaç |

---

## 6. Riskler ve Önlemler

| Risk | Olasılık | Önlem |
|------|----------|-------|
| gz-sim arm64 kurulamaz | Yüksek | Katman 1 (Python görsel) garantili — Gazebo opsiyonel kalır |
| RAM yetmez (ROS 2 + sim) | Orta | ROS 2 kullanılmaz — ArduPilot SITL doğrudan MAVLink |
| SITL + görsel FPS düşük | Düşük | Görselleştirici ayrı thread, tespit ayrı — panel senkron değil |
| Model sentetik sahnede zayıf | Orta | Kamera görüntüsü GERÇEK dataset görsellerinden üretilir (sentetik render değil) |
| Standda internet yok | — | Tamamen lokal çalışır (sunucuda veya laptoptan) |

---

## 7. Zaman Çizelgesi

| Faz | İş | Süre | Toplam |
|-----|----|------|--------|
| S0 | Altyapı doğrulama (SITL derle + gz-sim dene) | 1 gün | 1 |
| S1 | SITL + MAVLink kontrol | 1-2 gün | 2-3 |
| S2 | Görselleştirici + CNN overlay | 2-3 gün | 4-6 |
| S3 | Kapalı döngü (otonom takip) | 3-5 gün | 7-11 |
| S4 | Gazebo entegrasyonu (opsiyonel) | 3-5 gün | 10-16 |
| S5 | Stand paketi | 2 gün | 12-18 |

**MVP hedefi (stand için):** Faz S0-S2 + S5 → **6-8 günde** çalışan stand demosu (SITL + görselleştirici + canlı tespit). Gazebo (S4) bonus.

---

## 8. Laptop Deployment (sunum/gösterim ortamı)

**Kullanıcı kararı:** Simülasyon, sunum sırasında GPU'lu ve ekranlı bir laptopta çalıştırılacak.
Amaç: gerçek uçuş simülasyonundan çok **sürecin nasıl işlediğini anlatmak** (veri akışı gösterimi).

### 8.1 Neden Laptop-Öncelikli Tasarım

| Faktör | Etki |
|--------|------|
| CNN küçük (int8, 1.4M MAC) | **GPU gerekmez** — CPU'da 30+ FPS tespit; GPU opsiyonel (render) |
| Ekran var | Canlı görsel gösterim — stand'ın kalbi |
| Sunum ortamı | Tek laptopta çalışan, kolay kurulan sistem |
| Cross-platform | Windows/Mac/Linux'ta çalışmalı (laptop OS bilinmiyor) |

### 8.2 Mimari (laptop-öncelikli)

```
┌─────────────── LAPTOP (sunum) ───────────────┐
│                                              │
│  GÖRSEL KATMAN (seçenekler)                  │
│  A. Python görselleştirici (garantili,       │
│     pygame/OpenGL — her OS'ta çalışır)       │
│  B. Gazebo/Webots (laptop OS'una bağlı,      │
│     GPU ile render)                          │
│              │ kamera görüntüsü (160x120)    │
│              ▼                               │
│  CNN KÖPRÜSÜ (ctypes + C kütüphanesi)        │
│  → int8 model → tespit → bbox + sınıf + güven│
│              │ offset                        │
│              ▼                               │
│  KONTROL (MAVLink → SITL veya sanal takip)   │
│  → drone yaw/ileri → görsel katmana geri     │
│                                              │
│  SÜREÇ PANELİ: kamera→CNN→tespit→takip→komut │
│  her adım canlı değerlerle (veri akışı anlatım)│
└──────────────────────────────────────────────┘
```

### 8.3 Süreç Anlatımı Odaklı Panel (sunumun kalbi)

Sunumda "nasıl çalışıyor" anlatmak için ekran 4 panele bölünür:

| Panel | İçerik | Anlattığı |
|-------|--------|-----------|
| 1 | Sanal dünya (drone + hedef, 3D/2D) | "Drone sahadayken ne görür" |
| 2 | Drone kamerası + CNN overlay | "Model her kareyi nasıl işler" (bbox, sınıf, güven) |
| 3 | **Süreç akışı** (canlı pipeline) | "Veri nasıl akar": Kamera → CNN → Tespit → Takip → Komut, her adımda canlı değerler |
| 4 | Telemetri | Yükseklik, yaw, GPS, FPS, kilit durumu |

**Süreç paneli örneği:**
```
KAMERA ──▶ CNN ──▶ TESPİT ──▶ TAKİP ──▶ KOMUT
160x120   int8    drone_uav   KİLİTLİ     YAW +12°
30 FPS    2.5ms   %97.3       güven≥.60   ileri 2m/s
```

Bu panel, jüriye/izleyiciye **tüm sistemi tek bakışta** anlatır — drone'u uçurmadan.

### 8.4 Kurulum (laptop'a taşıma)

```bash
# Tek script: gereksinimler + model + çalıştırma
git clone tinyrlabs/tinydrone && cd tinydrone/simulation
./setup.sh        # Python venv + pygame/OpenGL + ctypes lib derle
./run_demo.sh     # Görselleştirici + CNN + panel başlat
```

- **CNN kütüphanesi:** `inference_int8.c` → paylaşımlı kütüphane (`libtinydrone.so` / `.dll` / `.dylib`) — model header'ları gömülü (0.43MB), ek dosya yok
- **Bağımlılıklar:** Python 3 + pygame (2D) veya moderngl (3D) + OpenCV (overlay) — GPU şart değil
- **Gazebo opsiyonu:** Laptop OS'una göre (Linux: Gazebo; Windows: WSL2; Mac: Webots) — çalışmazsa Panel 1 Python görselleştiriciyle doldurulur
- **Geliştirme sunucuda, gösterim laptopta:** Kod tek repo, iki ortam — aynı CNN, aynı senaryolar

### 8.5 Laptop Bilinmeyenleri (başlamadan netleşecek)

1. İşletim sistemi? (Windows / macOS / Linux)
2. GPU? (NVIDIA / AMD / Intel — opsiyonel render için)
3. Sunumda internet var mı? (yoksa tamamen lokal çalışmalı — tasarım zaten lokal)

---

## 9. Başlangıç Kararları (onayın gereken)

1. **Görselleştirici:** 2D (pygame, hızlı) mu 3D (OpenGL, etkileyici) mi? → Öneri: 2D ile başla, 3D'ye geç
2. **CNN bağlama:** ctypes + C kütüphanesi mi (performans, aynı kod) Python repro mu? → Öneri: ctypes
3. **Gazebo:** POC yapalım mı yoksa direkt Katman 1 ile mi gidelim? → Öneri: POC'u 1 günde dene, başarısızsa Katman 1
4. **SITL mi PX4 mü?** → Öneri: ArduPilot SITL (ARM'de derlenir, Gazebo desteği, MAVLink)
5. **Laptop OS/GPU?** → Sunum ortamını bilirsek deployment buna göre kesinleşir

---

*Plan onaylanırsa Faz S0 ile başlarız: ArduPilot SITL derleme + gz-sim kurulum denemesi.*

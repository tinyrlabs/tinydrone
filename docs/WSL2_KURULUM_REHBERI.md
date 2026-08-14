# TINYDRONE — WSL2 Kurulum Rehberi (MSI Thin 15, Windows)

**Amaç:** Laptopta Gazebo + ArduPilot SITL simülasyon ortamı kurmak.
**Süre:** ~1 saat (indirme hızına bağlı)
**Ön koşul:** Windows 10/11, internet, yönetici yetkisi

---

## Adım 1 — WSL2 Kurulumu (PowerShell, yönetici)

1. Başlat menüsü → "PowerShell" → sağ tık → **Yönetici olarak çalıştır**
2. Şu komutu çalıştır:
```powershell
wsl --install -d Ubuntu-22.04
```
3. Bilgisayar yeniden başlar (gerekirse). Tekrar açılınca Ubuntu terminali açılır.
4. **Kullanıcı adı ve şifre** belirle (ör: `samet`). Bu WSL Linux kullanıcısı — Windows hesabından farklı.

> Not: `Ubuntu-22.04` bulunamazsa: `wsl --list --online` ile kontrol et,
> ya da Microsoft Store'dan "Ubuntu 22.04.3 LTS" kur.

5. WSL sürümünü doğrula (PowerShell):
```powershell
wsl --set-default-version 2
wsl -l -v
```
Çıktıda `VERSION 2` olmalı. Değilse: `wsl --set-version Ubuntu-22.04 2`

---

## Adım 2 — Ubuntu İçinde İlk Kurulum

Ubuntu terminaline gir ve sırayla:

```bash
# Güncelle
sudo apt update && sudo apt upgrade -y

# Temel araçlar
sudo apt install -y git curl wget build-essential cmake python3-pip python3-dev \
    python3-numpy python3-lxml libxml2-dev libxslt1-dev

# Gazebo Classic 11 (Ubuntu 22.04'te resmi paket — en olgun ArduPilot desteği)
sudo apt install -y gazebo libgazebo11-dev

# Doğrula
gazebo --version   # "Gazebo multi-robot simulator, version 11.x" görmelisin
```

> **GUI testi (ilk kez):** `gazebo` yaz → pencere açılmalı (WSLg otomatik).
> Pencere açılmazsa: Windows'ta WSLg güncellemesi gerekebilir → `wsl --update`

---

## Adım 3 — ArduPilot SITL

```bash
# Bağımlılıklar
sudo apt install -y libtool automake autoconf libexpat1-dev libsdl2-dev

# MAVProxy (uçuş aracı — telemetri konsolu)
pip install --user mavproxy pymavlink

# ArduPilot kaynak
cd ~
git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git
cd ardupilot

# SITL derle (Copter)
./waf configure --board sitl
./waf copter

# Doğrula: sim_vehicle script'i
ls Tools/autotest/sim_vehicle.py
```

---

## Adım 4 — İlk Uçuş Testi (SITL headless)

```bash
cd ~/ardupilot
# Basit SITL başlat (uçuş aracı, MAVLink TCP 5760)
sim_vehicle.py -v ArduCopter -L Kocaeli --no-mavproxy --console
```

Başka terminalde:
```bash
# MAVProxy ile bağlan
mavproxy.py --master tcp:127.0.0.1:5760
# Konsolda: STABILIZE modunda, arm edip uçur
```

**Test:** `mode guided`, `arm throttle`, `takeoff 10` → drone 10m'ye çıkar.

---

## Adım 5 — Gazebo + ArduPilot Entegrasyonu

```bash
# gazebo-ardupilot-sitl plugin
cd ~
git clone https://github.com/ArduPilot/ardupilot_gazebo
cd ardupilot_gazebo
mkdir build && cd build
cmake .. && make -j$(nproc)
sudo make install

# Ortam değişkenleri (her terminalde):
echo 'export GAZEBO_PLUGIN_PATH=~/ardupilot_gazebo/build:${GAZEBO_PLUGIN_PATH}' >> ~/.bashrc
echo 'export GAZEBO_MODEL_PATH=~/ardupilot_gazebo/models:${GAZEBO_MODEL_PATH}' >> ~/.bashrc
source ~/.bashrc
```

**Test — drone Gazebo'da uçuyor:**
```bash
# Terminal 1: Gazebo dünyası
cd ~/ardupilot/Tools/autotest/ardupilot_gazebo
gazebo --verbose worlds/iris_ardupilot.world

# Terminal 2: SITL (Gazebo modeline bağlı)
cd ~/ardupilot
sim_vehicle.py -v ArduCopter -f gazebo-iris
```
Gazebo'da quadcopter görünür, SITL konsolundan uçurabilirsin.

---

## Adım 6 — Proje Repo + CNN Bridge

```bash
cd ~
git clone https://github.com/tinyrlabs/tinydrone.git
cd tinydrone

# Python bağımlılıkları (CNN bridge + süreç paneli)
python3 -m pip install --user numpy opencv-python pygame

# CNN kütüphanesi (C → paylaşımlı lib)
cd simulation
make          # libtinydrone.so üretir
```

---

## Sorun Giderme

| Sorun | Çözüm |
|-------|-------|
| `wsl` komutu yok | Windows 10 eski: WSL2 kernel güncellemesi indir |
| Gazebo pencere açılmıyor | `wsl --update` + `wsl --shutdown`, tekrar aç |
| SITL derleme hatası | `pip install --user pymavlink` tekrar + `./waf distclean && ./waf configure --board sitl` |
| GPU yavaş render | WSLg varsayılan; NVIDIA driver güncelse otomatik |
| 4050 CUDA (opsiyonel) | Windows NVIDIA driver kuruluysa WSL2'de `nvidia-smi` çalışır |

---

## Sonraki Adım (bu rehberden sonra)

1. Adım 4 testini geçir (SITL headless uçuş)
2. Adım 5 testini geçir (Gazebo'da drone uçuyor)
3. Repo'ya dön: `simulation/` klasöründe CNN bridge + dünya dosyaları hazırlanacak
4. Sonraki faz: sanal dünyaya hedef modelleri (tank/bina) ekleme

**Başarı kriteri (Faz S0):** Gazebo penceresinde quadcopter + SITL'den takeoff komutu ile yükseliyor.

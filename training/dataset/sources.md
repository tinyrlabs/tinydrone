# tinydrone Veri Seti Kaynakları

> Son güncelleme: 8 Ağustos 2026

## Kullanılan Veri Setleri

### 1. OpenImages V7 (Google)
- **URL:** https://storage.googleapis.com/openimages/
- **Lisans:** CC BY 4.0
- **Sınıflar:** Tank, Military vehicle, Armored car, Helicopter, Aircraft, Military aircraft
- **Durum:** Erişim kısıtlı (403 Forbidden) — 8 Ağustos 2026 itibariyle doğrudan CSV erişimi çalışmıyor. Google Cloud Storage bucket politikası değişmiş olabilir.
- **Fallback:** `aws s3 cp s3://open-images-dataset/` veya Kaggle mirror kullanılabilir.

### 2. xView (DIUx / National Geospatial-Intelligence Agency)
- **URL:** http://xviewdataset.org/
- **Lisans:** CC BY-NC-SA 4.0 (araştırma amaçlı)
- **İçerik:** 1M+ overhead (uydudan/İHA'dan) obje etiketi, 60 sınıf
- **Boyut:** ~15GB (training set)
- **Askeri sınıflar:** Heavy equipment, truck variants, cargo containers
- **Not:** Doğrudan askeri araç sınıfı YOK — genel araç/altyapı sınıfları üzerinden filtreleme yapılıyor.

### 3. VisDrone (Tianjin University)
- **URL:** https://github.com/VisDrone/VisDrone-Dataset
- **Lisans:** CC BY-NC-SA 4.0
- **İçerik:** Drone ile çekilmiş 10K+ görüntü, 2.6M bounding box
- **Kullanım:** Background (negatif) sınıfı için — insansız hava aracı perspektifi
- **Boyut:** ~2GB

### 4. DOTA (Wuhan University)
- **URL:** https://captain-whu.github.io/DOTA/
- **Lisans:** Akademik kullanım
- **İçerik:** Havadan çekilmiş obje tespiti, 15 sınıf
- **Askeri sınıflar:** Vehicle, plane, ship, storage tank
- **Not:** Faz 0'da indirilmedi — ihtiyaç halinde eklenebilir.

### 5. MSTAR (AFRL)
- **URL:** https://www.sdms.afrl.af.mil/
- **Lisans:** ABD Hükümeti, onay gerektirir
- **İçerik:** SAR (Synthetic Aperture Radar) tank görüntüleri
- **Sınıflar:** BMP-2, BTR-70, T-72, BTR-60, 2S1, BRDM-2, D7, T-62, ZIL-131, ZSU-23/4
- **Durum:** Erişim başvurusu gerekiyor — şimdilik dahil edilmedi.

## Sentetik Veri Üretimi

Gerçek veri setlerine erişim sınırlı olduğu için, Faz 0'da ağırlıklı olarak **sentetik veri** kullanıldı:

- **Yöntem:** PIL + numpy ile geometrik şekil çizimi
- **Özellikler:** Her sınıf için karakteristik şekiller (tank: dikdörtgen+taret, drone: çapraz, bina: dikdörtgen+pencereler), rastgele arazi arka planı
- **Augmentasyon:** Renk varyasyonu, gürültü, döndürme
- **Limitasyon:** Gerçek fotografik doku yok — sadece şekil tanıma için uygun

⚠️ **Uyarı:** Sentetik veri, gerçek dünya performansını garanti etmez. Production deployment öncesi gerçek veri ile eğitim şart.

## Veri Seti İstatistikleri

| Sınıf | Ham (raw) | Train | Val | Test | Kaynak |
|-------|-----------|-------|-----|------|--------|
| tank | 500 | 350 | 75 | 75 | Sentetik |
| armored_vehicle | 500 | 350 | 75 | 75 | Sentetik |
| military_building | 500 | 350 | 75 | 75 | Sentetik |
| drone_uav | 500 | 350 | 75 | 75 | Sentetik |
| background | 500 | 350 | 75 | 75 | Sentetik |
| **Toplam** | **2,500** | **1,750** | **375** | **375** | |

## Geliştirme Notları

- **Faz 3+ için:** Gerçek askeri araç görselleri (Kaggle, Roboflow) eklenecek
- **Kaggle API:** `kaggle datasets download` ile askeri veri setleri çekilebilir (API key gerekli)
- **Roboflow:** Universe API'si rate-limited, key ile 1000 görsel/ay bedava

## Referanslar

- OpenImages V7: https://storage.googleapis.com/openimages/web/index.html
- xView: http://xviewdataset.org/
- VisDrone: http://aiskyeye.com/
- DOTA: https://captain-whu.github.io/DOTA/
- MSTAR: https://www.sdms.afrl.af.mil/index.php?collection=mstar

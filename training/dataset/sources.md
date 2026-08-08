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

## Veri Seti İstatistikleri (Final)

| Sınıf | Train | Val | Test | Toplam | Kaynak |
|-------|-------|-----|------|--------|--------|
| tank | 114 | 24 | 26 | 164 | Kaggle DS4: UAV Battle Tank Detection |
| armored_vehicle | 5.605 | 1.201 | 1.202 | 8.008 | Kaggle DS2: Normal vs Military Vehicles |
| drone_uav | 3.666 | 1.112 | 1.128 | 8.028 | Kaggle DS5+DS6: Drone YOLO + Amateur UAV |
| background | 6.611 | 1.416 | 1.418 | 9.445 | Kaggle DS2: Other (civilian vehicles) |
| military_building | 0 | 0 | 0 | 0 | ⏳ Pending (COCO/Places365) |
| **Toplam** | **15.996** | **3.753** | **3.774** | **23.523** | **4 Kaggle datasets** |

**Format:** 32×32 RGB PNG  
**Disk boyutu:** ~94MB (processed)  
**Split:** 70% train / 15% val / 15% test

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

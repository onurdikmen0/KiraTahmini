"""
Coğrafi Zenginleştirme — İlçe Geocoding
=======================================

Veri setindeki benzersiz (il, ilçe) çiftlerini OpenStreetMap/Nominatim ile
geocode ederek lat/lon üretir ve `data/geo/ilce_latlon.csv` dosyasına yazar.
Ardından her il için merkez (ilçelerin ortalaması) ve her ilçenin il merkezine
uzaklığını (km) hesaplar.

Özellikler:
- **Resumable**: çıktı CSV'de zaten olan ilçeler atlanır; tekrar çalıştırınca
  kaldığı yerden devam eder.
- **Kademeli kayıt**: her başarılı istekte diske yazar (kesinti güvenli).
- Nominatim kullanım politikasına saygı için istekler arası 1 sn bekler
  (osmnx cache açık -> tekrarlar anında).

Çalıştırma:
    python src/cografi_zenginlestirme.py

Mahalle düzeyi (~4600 istek) pratik olmadığı için ilçe düzeyi seçildi.
"""

from pathlib import Path
import sys
import time

import numpy as np
import osmnx as ox
import pandas as pd

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK / "src"))
from model_egitimi import temizle  # noqa: E402

HAM_VERI = KOK / "data" / "raw" / "turkey_rent.csv"
CIKTI = KOK / "data" / "geo" / "ilce_latlon.csv"

ox.settings.use_cache = True
ox.settings.log_console = False


def benzersiz_ilceler() -> pd.DataFrame:
    df = temizle(pd.read_csv(HAM_VERI, low_memory=False))
    return (
        df[["il", "ilce"]].dropna().drop_duplicates()
        .sort_values(["il", "ilce"]).reset_index(drop=True)
    )


def geocode_dene(il: str, ilce: str):
    """İlçeyi birkaç sorgu varyantıyla geocode etmeyi dener; (lat, lon) veya None."""
    for sorgu in (f"{ilce}, {il}, Türkiye", f"{ilce}, {il}, Turkey", f"{ilce} ilçesi, {il}"):
        try:
            lat, lon = ox.geocode(sorgu)
            # Türkiye sınırları kabaca: lat 35-43, lon 25-45
            if 35 <= lat <= 43 and 25 <= lon <= 45:
                return lat, lon
        except Exception:
            pass
    return None


def main():
    hedef = benzersiz_ilceler()
    print(f"Toplam benzersiz ilçe: {len(hedef)}")

    # resume: mevcut çıktıyı oku
    if CIKTI.exists():
        mevcut = pd.read_csv(CIKTI)
        bilinen = set(zip(mevcut["il"], mevcut["ilce"]))
        print(f"Zaten geocode edilmiş: {len(mevcut)} (atlanacak)")
    else:
        mevcut = pd.DataFrame(columns=["il", "ilce", "lat", "lon"])
        bilinen = set()

    kayitlar = mevcut.to_dict("records")
    basarisiz = []
    for i, satir in hedef.iterrows():
        il, ilce = satir["il"], satir["ilce"]
        if (il, ilce) in bilinen:
            continue
        sonuc = geocode_dene(il, ilce)
        if sonuc:
            kayitlar.append({"il": il, "ilce": ilce, "lat": sonuc[0], "lon": sonuc[1]})
            pd.DataFrame(kayitlar).to_csv(CIKTI, index=False, encoding="utf-8")
            print(f"[{i+1}/{len(hedef)}] {il}/{ilce} -> {sonuc[0]:.4f}, {sonuc[1]:.4f}")
        else:
            basarisiz.append((il, ilce))
            print(f"[{i+1}/{len(hedef)}] {il}/{ilce} -> BAŞARISIZ")
        time.sleep(1)  # Nominatim'e saygı

    print(f"\nGeocode tamam. Başarısız: {len(basarisiz)} -> {basarisiz[:20]}")

    # il merkezi (ilçelerin ortalaması) + il merkezine uzaklık (km, haversine)
    df = pd.read_csv(CIKTI)
    il_merkez = df.groupby("il")[["lat", "lon"]].mean().rename(
        columns={"lat": "il_lat", "lon": "il_lon"})
    df = df.merge(il_merkez, on="il")
    df["il_merkeze_uzaklik_km"] = _haversine(df["lat"], df["lon"], df["il_lat"], df["il_lon"])
    df.to_csv(CIKTI, index=False, encoding="utf-8")
    print(f"İl merkezi + uzaklık eklendi. Kaydedildi: {CIKTI}")


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


if __name__ == "__main__":
    main()

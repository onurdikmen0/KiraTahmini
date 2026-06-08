"""
Kira Tahmin Modeli — Eğitim Scripti
====================================

Bu script HAM veriden (`data/raw/turkey_rent.csv`) başlayarak veriyi temizler,
sızıntısız öznitelik üretir, XGBoost modelini eğitir ve tüm artefaktları
`models/` altına kaydeder. Streamlit arayüzü `temizle()` ve `hazirla_X()`
fonksiyonlarını + kaydedilen artefaktları yeniden kullanır.

VERİ SIZINTISI DÜZELTMELERİ (eski notebook'a göre):
  1. `metrekare_basi_kira = fiyat / net_metrekare` ve `fiyat_oda_orani = fiyat / oda`
     hedefin kendisini içeriyordu -> HİÇ ÜRETİLMİYOR.
  2. İl/ilçe/mahalle ortalama kira (target encoding) eskiden TÜM veriyle (test
     dahil) hesaplanıyordu -> artık SADECE train üzerinden, mahalle->ilçe->il->global
     hiyerarşik fallback ve smoothing ile hesaplanır.
  3. Konum eskiden one-hot'tı (3000+ kolon, ~150 MB, ezberleme) -> kompakt target
     encoding ile değiştirildi.

Çalıştırma:
    python src/model_egitimi.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

# --- Yollar (script src/ içinde olduğu için proje köküne çıkıyoruz) ---
KOK = Path(__file__).resolve().parent.parent
HAM_VERI = KOK / "data" / "raw" / "turkey_rent.csv"
GEO_DOSYA = KOK / "data" / "geo" / "ilce_latlon.csv"
MODEL_DIZIN = KOK / "models"
MODEL_DOSYA = MODEL_DIZIN / "kira_model.joblib"

# Coğrafi öznitelik tablosu tembel yüklenir (Streamlit'te her tahminde tekrar okumamak için)
_GEO_CACHE = None

# Bina yaşı aralıklarını tek sayıya indir
YAS_MAP = {
    "0 (Yeni)": 0, "1": 1, "2": 2, "3": 3, "4": 4,
    "5-10": 7, "11-15": 13, "16-20": 18, "21 Ve Üzeri": 25,
}

# Bulunduğu kat metinlerini sayıya indir (özel durumlar sentinel değerlerle)
KAT_MAP = {
    "Düz Giriş (Zemin)": 0, "Bahçe Katı": 0, "Bahçe Dublex": 0,
    "Yüksek Giriş": 0, "Bodrum Kat": -1,
    "Kot 1 (-1)": -1, "Kot 2 (-2)": -2, "Kot 3 (-3)": -3, "Kot 4 (-4)": -4,
    "Çatı Katı": 999, "Çatı Dubleks": 999, "Müstakil": 1000, "Villa Tipi": 1000,
}

# Tapu durumu sıralı (ordinal) kodlama
TAPU_MAP = {
    "Tapu Kaydı Yok": 0, "Bilinmiyor": 1, "Yabancıdan": 2, "Kıbrıs Tapulu": 3,
    "Kooperatiften Tapu": 4, "Hisseli Tapu": 5, "Arsa Tapulu": 6,
    "Müstakil Tapulu": 7, "Kat İrtifakı": 8, "Kat Mülkiyeti": 9,
}

# Ham (Türkçe) kolon adlarından çalışma adlarına
KOLON_AD = {
    "Net Metrekare": "net_metrekare", "Brüt Metrekare": "brut_metrekare",
    "Oda Sayısı": "oda_sayisi", "Binanın Yaşı": "binanin_yasi",
    "Bulunduğu Kat": "bulundugu_kat", "Binanın Kat Sayısı": "binanin_kat_sayisi",
    "Isıtma Tipi": "isitma_tipi", "Kullanım Durumu": "kullanim_durumu",
    "Tapu Durumu": "tapu_durumu", "Site İçerisinde": "site_icerisinde",
    "Banyo Sayısı": "banyo_sayisi", "Eşya Durumu": "esya_durumu",
}

# Modele giren sayısal öznitelikler (target-encode + coğrafi kolonlar sona eklenir)
SAYISAL_KOLONLAR = [
    "net_metrekare", "brut_metrekare", "oda", "salon", "binanin_yasi",
    "bulundugu_kat", "binanin_kat_sayisi", "banyo_sayisi", "site_icerisinde",
    "tapu_durumu", "esya_durumu", "yapi_durumu", "oda_basi_alan", "burut_net_orani",
    "il_te", "ilce_te", "mahalle_te",
    "ilce_lat", "ilce_lon", "il_merkeze_uzaklik_km",
]
# One-hot uygulanacak kategorik kolonlar
KATEGORIK_KOLONLAR = ["isitma_tipi", "kullanim_durumu"]
# Target encoding için konum kolonları
KONUM_KOLONLAR = ["il", "ilce", "mahalle"]


def _geo_yukle():
    """İlçe lat/lon tablosunu (varsa) tembel yükler. Yoksa None döner."""
    global _GEO_CACHE
    if _GEO_CACHE is None and GEO_DOSYA.exists():
        g = pd.read_csv(GEO_DOSYA)
        il_mrkz = g.groupby("il")[["lat", "lon"]].mean().rename(
            columns={"lat": "_il_lat", "lon": "_il_lon"})
        _GEO_CACHE = {
            "ilce": g.set_index(["il", "ilce"]).to_dict("index"),
            "il": il_mrkz.to_dict("index"),
            "global": {"lat": g["lat"].mean(), "lon": g["lon"].mean()},
        }
    return _GEO_CACHE


def _cografi_ekle(df: pd.DataFrame) -> pd.DataFrame:
    """il/ilce'ye göre ilce_lat, ilce_lon, il_merkeze_uzaklik_km kolonlarını ekler.

    Fallback: ilçe yoksa il merkezi, o da yoksa global ortalama koordinat kullanılır.
    Geo tablosu hiç yoksa kolonlar NaN bırakılır (XGBoost NaN'i tolere eder).
    """
    geo = _geo_yukle()
    if geo is None:
        df["ilce_lat"] = np.nan
        df["ilce_lon"] = np.nan
        df["il_merkeze_uzaklik_km"] = np.nan
        return df

    glob = geo["global"]

    def cek(il, ilce):
        rec = geo["ilce"].get((il, ilce))
        if rec:
            return rec["lat"], rec["lon"], rec.get("il_merkeze_uzaklik_km", 0.0)
        ilm = geo["il"].get(il)
        if ilm:
            return ilm["_il_lat"], ilm["_il_lon"], 0.0
        return glob["lat"], glob["lon"], 0.0

    sonuc = [cek(il, ilce) for il, ilce in zip(df["il"], df["ilce"])]
    df["ilce_lat"] = [s[0] for s in sonuc]
    df["ilce_lon"] = [s[1] for s in sonuc]
    df["il_merkeze_uzaklik_km"] = [s[2] for s in sonuc]
    return df


def temizle(df: pd.DataFrame) -> pd.DataFrame:
    """Ham ilan verisini modele uygun hale getirir (sızıntısız).

    Tek bir ilanlık DataFrame ile de çağrılabilir (Streamlit için).
    Çıktı: sayısal öznitelikler + isitma_tipi/kullanim_durumu + il/ilce/mahalle
    (+ varsa `fiyat`).
    """
    df = df.copy()
    df = df.rename(columns=KOLON_AD)

    # net / brüt metrekare: " m²" ve binlik "." temizle
    for kol in ["net_metrekare", "brut_metrekare"]:
        df[kol] = (
            df[kol].astype(str)
            .str.replace(" m²", "", regex=False)
            .str.replace(".", "", regex=False)
        )
        df[kol] = pd.to_numeric(df[kol], errors="coerce")

    # oda sayısı: "2+1" -> oda=2, salon=1 ; "Stüdyo" -> 0 ; "1 Oda" -> oda=1
    oda_salon = df["oda_sayisi"].astype(str).str.split("+", n=1, expand=True)
    df["oda"] = (
        oda_salon[0].str.replace("Stüdyo", "0", regex=False)
        .str.replace(" Oda", "", regex=False).str.strip()
    )
    df["oda"] = pd.to_numeric(df["oda"], errors="coerce")
    if oda_salon.shape[1] > 1:
        df["salon"] = pd.to_numeric(oda_salon[1], errors="coerce").fillna(0)
    else:
        df["salon"] = 0

    # bina yaşı aralık -> sayı
    df["binanin_yasi"] = df["binanin_yasi"].map(YAS_MAP)

    # yapı durumu: yaş<=1 ise sıfır(0) değilse ikinci el(1)  (notebook ile aynı mantık)
    df["yapi_durumu"] = (df["binanin_yasi"] <= 1).astype(int)

    # bulunduğu kat
    df["bulundugu_kat"] = (
        df["bulundugu_kat"].astype(str).str.replace(".Kat", "", regex=False).replace(KAT_MAP)
    )
    df["bulundugu_kat"] = pd.to_numeric(df["bulundugu_kat"], errors="coerce")

    # banyo sayısı: "Yok" -> 0, "6+" -> 6
    df["banyo_sayisi"] = (
        df["banyo_sayisi"].astype(str)
        .str.replace("Yok", "0", regex=False).str.replace("6+", "6", regex=False)
    )
    df["banyo_sayisi"] = pd.to_numeric(df["banyo_sayisi"], errors="coerce")

    # site içerisinde / eşya durumu -> ikili
    df["site_icerisinde"] = (
        df["site_icerisinde"].astype(str)
        .str.replace("Evet", "1", regex=False).str.replace("Hayır", "0", regex=False)
    )
    df["site_icerisinde"] = pd.to_numeric(df["site_icerisinde"], errors="coerce").fillna(0)
    df["esya_durumu"] = df["esya_durumu"].map({"Boş": 0, "Eşyalı": 1}).fillna(0)

    df["tapu_durumu"] = df["tapu_durumu"].map(TAPU_MAP).fillna(1)
    df["binanin_kat_sayisi"] = pd.to_numeric(df["binanin_kat_sayisi"], errors="coerce")

    # konum -> il / ilçe / mahalle
    konum = df["konum"].astype(str).str.split(" - ", n=2, expand=True)
    df["il"] = konum[0].str.strip()
    df["ilce"] = konum[1].str.strip() if konum.shape[1] > 1 else None
    df["mahalle"] = (
        konum[2].str.replace(" Mahallesi", "", regex=False).str.strip()
        if konum.shape[1] > 2 else None
    )

    # fiyat (varsa) : "11.000TL" -> 11000
    if "fiyat" in df.columns:
        df["fiyat"] = (
            df["fiyat"].astype(str)
            .str.replace("TL", "", regex=False).str.replace(".", "", regex=False).str.strip()
        )
        df["fiyat"] = pd.to_numeric(df["fiyat"], errors="coerce")

    # sızıntısız türetilmiş öznitelikler
    df["oda_basi_alan"] = np.where(df["oda"] > 0, df["net_metrekare"] / df["oda"], df["net_metrekare"])
    df["burut_net_orani"] = df["brut_metrekare"] / df["net_metrekare"]

    # coğrafi öznitelikler (ilçe lat/lon + il merkezine uzaklık)
    df = _cografi_ekle(df)

    return df


def fit_encoders(df: pd.DataFrame, smoothing: int = 10) -> dict:
    """Konum target encoding haritalarını SADECE verilen (train) veriden öğrenir.

    Smoothing: az örnekli mahalle/ilçe değerlerini global ortalamaya çeker, böylece
    nadir konumlar ezberlenmez.
    """
    global_mean = df["fiyat"].mean()

    def harita(kol):
        g = df.groupby(kol)["fiyat"].agg(["mean", "count"])
        te = (g["count"] * g["mean"] + smoothing * global_mean) / (g["count"] + smoothing)
        return te.to_dict()

    return {
        "global_mean": global_mean,
        "il": harita("il"),
        "ilce": harita("ilce"),
        "mahalle": harita("mahalle"),
    }


def _konum_te(df: pd.DataFrame, enc: dict) -> pd.DataFrame:
    """Hiyerarşik fallback ile il/ilçe/mahalle target encoding kolonlarını üretir."""
    g = enc["global_mean"]
    il_te = df["il"].map(enc["il"]).fillna(g)
    ilce_te = df["ilce"].map(enc["ilce"]).fillna(il_te)
    mahalle_te = df["mahalle"].map(enc["mahalle"]).fillna(ilce_te)
    return pd.DataFrame({"il_te": il_te, "ilce_te": ilce_te, "mahalle_te": mahalle_te},
                        index=df.index)


def hazirla_X(df: pd.DataFrame, enc: dict, dummy_kolonlar: list) -> pd.DataFrame:
    """Temizlenmiş df'ten modele girecek nihai öznitelik matrisini kurar.

    enc          : fit_encoders çıktısı
    dummy_kolonlar: eğitimdeki one-hot kolon sırası (hizalama için)
    """
    te = _konum_te(df, enc)
    sayisal = pd.concat([df[[c for c in SAYISAL_KOLONLAR if c not in te.columns]], te], axis=1)
    sayisal = sayisal[SAYISAL_KOLONLAR]  # sırayı sabitle

    kategorik = pd.get_dummies(df[KATEGORIK_KOLONLAR], columns=KATEGORIK_KOLONLAR).astype(int)
    X = pd.concat([sayisal, kategorik], axis=1)
    # eğitimdeki kolon kümesine hizala (eksikleri 0, fazlaları at)
    X = X.reindex(columns=SAYISAL_KOLONLAR + dummy_kolonlar, fill_value=0)
    return X


def _yeni_model() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )


def _iqr_filtre(df: pd.DataFrame, kol: str) -> pd.DataFrame:
    q1, q3 = df[kol].quantile([0.25, 0.75])
    iqr = q3 - q1
    return df[(df[kol] >= q1 - 1.5 * iqr) & (df[kol] <= q3 + 1.5 * iqr)]


def main():
    print("Ham veri okunuyor:", HAM_VERI)
    ham = pd.read_csv(HAM_VERI, low_memory=False).drop_duplicates()
    df = temizle(ham)

    # zorunlu alanlar eksikse at + aykırı fiyat/alan temizliği
    df = df.dropna(subset=["fiyat", "net_metrekare", "brut_metrekare", "oda",
                           "binanin_yasi", "bulundugu_kat", "il", "ilce", "mahalle"])
    df = df[(df["net_metrekare"] > 0) & (df["fiyat"] > 0)]
    df = _iqr_filtre(df, "fiyat")
    df = _iqr_filtre(df, "net_metrekare")
    df = df.reset_index(drop=True)
    print(f"Temiz satır sayısı: {len(df)}")

    y = df["fiyat"].values

    # --- 1) Dürüst değerlendirme: train/test ayır, encoder SADECE train'den ---
    tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
    df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]

    enc = fit_encoders(df_tr)
    # dummy kolon sırasını train'den belirle
    dummy_kolonlar = sorted(
        c for c in pd.get_dummies(df_tr[KATEGORIK_KOLONLAR], columns=KATEGORIK_KOLONLAR).columns
    )

    X_tr = hazirla_X(df_tr, enc, dummy_kolonlar)
    X_te = hazirla_X(df_te, enc, dummy_kolonlar)

    model = _yeni_model()
    model.fit(X_tr, np.log1p(df_tr["fiyat"]))

    def degerlendir(X, gercek):
        tah = np.expm1(model.predict(X))
        return r2_score(gercek, tah), np.sqrt(mean_squared_error(gercek, tah))

    r2_te, rmse_te = degerlendir(X_te, df_te["fiyat"].values)
    r2_tr, rmse_tr = degerlendir(X_tr, df_tr["fiyat"].values)
    # log-ölçekli RMSE: tahmin aralığı için çarpımsal bant (fiyatla orantılı)
    rmse_log = float(np.sqrt(mean_squared_error(
        np.log1p(df_te["fiyat"].values), model.predict(X_te))))
    print("\n=== Hold-out (sızıntısız) ===")
    print(f"Test  R²: {r2_te:.4f}   RMSE: {rmse_te:,.0f} TL")
    print(f"Train R²: {r2_tr:.4f}   RMSE: {rmse_tr:,.0f} TL")
    print(f"(Train-Test R² farkı: {r2_tr - r2_te:.3f} — büyükse ezberleme işareti)")

    # --- 2) Dürüst CV: her fold'da encoder yeniden fit edilir ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_skor = []
    for f_tr, f_va in kf.split(df):
        d_tr, d_va = df.iloc[f_tr], df.iloc[f_va]
        e = fit_encoders(d_tr)
        dk = sorted(pd.get_dummies(d_tr[KATEGORIK_KOLONLAR], columns=KATEGORIK_KOLONLAR).columns)
        m = _yeni_model()
        m.fit(hazirla_X(d_tr, e, dk), np.log1p(d_tr["fiyat"]))
        tah = np.expm1(m.predict(hazirla_X(d_va, e, dk)))
        cv_skor.append(r2_score(d_va["fiyat"].values, tah))
    print("\n=== 5-Fold CV (sızıntısız) ===")
    print("Fold R²:", [f"{s:.3f}" for s in cv_skor])
    print(f"CV R² ortalama: {np.mean(cv_skor):.4f} (+/- {np.std(cv_skor):.4f})")

    # --- 3) Nihai model: TÜM veriyle yeniden eğit ve kaydet ---
    enc_full = fit_encoders(df)
    dummy_full = sorted(pd.get_dummies(df[KATEGORIK_KOLONLAR], columns=KATEGORIK_KOLONLAR).columns)
    X_full = hazirla_X(df, enc_full, dummy_full)
    final_model = _yeni_model()
    final_model.fit(X_full, np.log1p(df["fiyat"]))

    # Kademeli dropdown'lar için konum ağacı: {il: {ilce: [mahalle, ...]}}
    konum_agaci = {}
    for il, ilce, mah in df[["il", "ilce", "mahalle"]].drop_duplicates().itertuples(index=False):
        konum_agaci.setdefault(il, {}).setdefault(ilce, set()).add(mah)
    konum_agaci = {il: {ilce: sorted(m) for ilce, m in sorted(ilceler.items())}
                   for il, ilceler in sorted(konum_agaci.items())}

    # Sayısal alanların kullanıcıya sunulacak makul varsayılan/aralıkları
    sayisal_ozet = {
        kol: {"min": float(df[kol].min()), "max": float(df[kol].max()),
              "medyan": float(df[kol].median())}
        for kol in ["net_metrekare", "brut_metrekare", "binanin_kat_sayisi", "banyo_sayisi"]
    }

    MODEL_DIZIN.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "encoders": enc_full,
            "dummy_kolonlar": dummy_full,
            "sayisal_kolonlar": SAYISAL_KOLONLAR,
            "konum_agaci": konum_agaci,
            "isitma_tipleri": sorted(df["isitma_tipi"].dropna().unique()),
            "kullanim_durumlari": sorted(df["kullanim_durumu"].dropna().unique()),
            "yas_secenekleri": list(YAS_MAP.keys()),
            "sayisal_ozet": sayisal_ozet,
            "metrikler": {
                "test_r2": r2_te, "cv_r2": float(np.mean(cv_skor)),
                "test_rmse": rmse_te, "rmse_log": rmse_log,
            },
        },
        MODEL_DOSYA,
    )
    print(f"\nModel kaydedildi: {MODEL_DOSYA}")


if __name__ == "__main__":
    main()

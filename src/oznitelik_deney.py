"""
Öznitelik Deneyi (geçici)
=========================
Aday öznitelikleri dürüst 5-fold CV ile ölçer; baseline'a katkısını gösterir.
Faydalı çıkanlar model_egitimi.py'ye taşınır, bu dosya sonra silinebilir.
"""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from model_egitimi import KATEGORIK_KOLONLAR, fit_encoders, hazirla_X, _yeni_model
from model_karsilastirma import veri_yukle
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


def brut_temizle(df):
    """Brüt metrekare veri hatalarını ele (brüt/net oranı 0.8-2.5 dışı atılır)."""
    return df[df["burut_net_orani"].between(0.8, 2.5)].reset_index(drop=True)


def ek_oznitelikler(d, ref, hangi):
    """hangi: hangi öznitelik gruplarının ekleneceği (set)."""
    out = pd.DataFrame(index=d.index)
    if "kat" in hangi:
        kat = d["bulundugu_kat"]
        out["kat_tipi"] = np.select(
            [kat >= 1000, kat >= 999, kat < 0, kat == 0],
            ["mustakil", "cati", "bodrum", "zemin"], default="normal")
        kat_temiz = np.where(kat >= 1000, np.nan,
                             np.where(kat >= 999, d["binanin_kat_sayisi"], kat))
        out["kat_temiz"] = kat_temiz
        out["kat_orani"] = kat_temiz / d["binanin_kat_sayisi"]
    if "oda" in hangi:
        out["toplam_oda"] = d["oda"] + d["salon"]
    if "mahalle_ist" in hangi:
        yog = ref["mahalle"].value_counts()
        out["mahalle_yogunluk"] = d["mahalle"].map(yog).fillna(1)
        med = ref.groupby("mahalle")["net_metrekare"].median()
        glob = ref["net_metrekare"].median()
        out["alan_vs_mahalle"] = d["net_metrekare"] / d["mahalle"].map(med).fillna(glob)
    return out


def cv(df, hangi=frozenset()):
    kf = KFold(5, shuffle=True, random_state=42)
    r2s, mdapes = [], []
    for tr, va in kf.split(df):
        dtr, dva = df.iloc[tr], df.iloc[va]
        enc = fit_encoders(dtr)
        dk = sorted(pd.get_dummies(dtr[KATEGORIK_KOLONLAR], columns=KATEGORIK_KOLONLAR).columns)
        Xtr, Xva = hazirla_X(dtr, enc, dk), hazirla_X(dva, enc, dk)
        if hangi:
            etr = ek_oznitelikler(dtr, dtr, hangi)
            eva = ek_oznitelikler(dva, dtr, hangi)
            if "kat" in hangi:
                etr = pd.get_dummies(etr, columns=["kat_tipi"])
                eva = pd.get_dummies(eva, columns=["kat_tipi"]).reindex(columns=etr.columns, fill_value=0)
            Xtr = pd.concat([Xtr.reset_index(drop=True), etr.reset_index(drop=True)], axis=1)
            Xva = pd.concat([Xva.reset_index(drop=True), eva.reset_index(drop=True)], axis=1)
        m = _yeni_model(); m.fit(Xtr, np.log1p(dtr["fiyat"]))
        tah = np.expm1(m.predict(Xva)); ger = dva["fiyat"].values
        r2s.append(r2_score(ger, tah)); mdapes.append(np.median(np.abs(tah - ger) / ger * 100))
    return np.mean(r2s), np.std(r2s), np.mean(mdapes)


if __name__ == "__main__":
    ham = veri_yukle()
    print(f"Ham veri: {len(ham)} ilan")
    df = brut_temizle(ham)
    print(f"Brüt temizlenmiş: {len(df)} ilan ({len(ham)-len(df)} satır atıldı)\n")

    def yaz(ad, hangi):
        r2, s, md = cv(df, hangi)
        print(f"{ad:34s} R²={r2:.4f} (±{s:.3f})  MdAPE=%{md:.1f}")

    yaz("Baseline (brüt temiz)", frozenset())
    yaz("+ kat tipi/oranı", {"kat"})
    yaz("+ toplam_oda", {"oda"})
    yaz("+ mahalle istatistikleri", {"mahalle_ist"})
    yaz("+ HEPSİ", {"kat", "oda", "mahalle_ist"})

"""
Model Karşılaştırma
===================

Aynı sızıntısız kurulumla (her fold'da target encoder yeniden fit, log-hedef,
coğrafi öznitelikler dahil) farklı regresyon modellerini 5-fold CV ile kıyaslar.
Her model için R² ve medyan mutlak yüzde hata (MdAPE) raporlanır.

Çalıştırma:
    python src/model_karsilastirma.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK / "src"))
from model_egitimi import (  # noqa: E402
    HAM_VERI, KATEGORIK_KOLONLAR, _iqr_filtre, fit_encoders, hazirla_X, temizle,
)

warnings.filterwarnings("ignore")


def model_fabrikalari():
    """Her çağrıda taze bir model üreten fabrikalar (CV'de fold başına yeni model)."""
    import lightgbm as lgb
    import xgboost as xgb

    return {
        "Ridge (lineer)": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=42),
        "HistGBM": lambda: HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.05, max_depth=8, random_state=42),
        "XGBoost": lambda: xgb.XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=42, n_jobs=-1),
        "LightGBM": lambda: lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1),
    }


def veri_yukle():
    df = temizle(pd.read_csv(HAM_VERI, low_memory=False)).drop_duplicates()
    df = df.dropna(subset=["fiyat", "net_metrekare", "brut_metrekare", "oda",
                           "binanin_yasi", "bulundugu_kat", "il", "ilce", "mahalle"])
    df = df[(df["net_metrekare"] > 0) & (df["fiyat"] > 0)]
    df = _iqr_filtre(df, "fiyat")
    df = _iqr_filtre(df, "net_metrekare")
    return df.reset_index(drop=True)


def cv_degerlendir(fabrika, df, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2_list, mdape_list = [], []
    for f_tr, f_va in kf.split(df):
        d_tr, d_va = df.iloc[f_tr], df.iloc[f_va]
        enc = fit_encoders(d_tr)
        dk = sorted(pd.get_dummies(d_tr[KATEGORIK_KOLONLAR], columns=KATEGORIK_KOLONLAR).columns)
        model = fabrika()
        model.fit(hazirla_X(d_tr, enc, dk), np.log1p(d_tr["fiyat"]))
        # log tahmini makul üst sınıra clip'le (lineer modellerde taşmayı önler)
        log_tah = np.clip(model.predict(hazirla_X(d_va, enc, dk)), 0, 16)
        tah = np.expm1(log_tah)
        ger = d_va["fiyat"].values
        r2_list.append(r2_score(ger, tah))
        mdape_list.append(np.median(np.abs(tah - ger) / ger * 100))
    return np.mean(r2_list), np.std(r2_list), np.mean(mdape_list)


def main():
    df = veri_yukle()
    print(f"Veri: {len(df)} ilan, 5-fold CV ile karşılaştırma\n")
    sonuc = []
    for ad, fabrika in model_fabrikalari().items():
        t0 = time.time()
        r2, r2_std, mdape = cv_degerlendir(fabrika, df)
        sure = time.time() - t0
        sonuc.append((ad, r2, r2_std, mdape, sure))
        print(f"{ad:18s} R²={r2:.4f} (±{r2_std:.3f})  MdAPE=%{mdape:.1f}  ({sure:.0f}s)")

    print("\n=== Sıralama (R²'ye göre) ===")
    for ad, r2, r2_std, mdape, _ in sorted(sonuc, key=lambda x: -x[1]):
        print(f"{ad:18s} R²={r2:.4f}  MdAPE=%{mdape:.1f}")


if __name__ == "__main__":
    main()

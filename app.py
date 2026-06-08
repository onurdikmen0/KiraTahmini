"""
Kira Tahmin Uygulaması — Streamlit Arayüzü
==========================================

Kullanıcıdan konut özelliklerini alır, `models/kira_model.joblib` modelini
kullanarak tahmini aylık kirayı bir aralıkla birlikte gösterir.

Çalıştırma:
    streamlit run app.py

Not: Önce modelin eğitilmiş olması gerekir -> `python src/model_egitimi.py`
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

# src/ içindeki temizle() ve hazirla_X() fonksiyonlarını yeniden kullan
KOK = Path(__file__).resolve().parent
sys.path.insert(0, str(KOK / "src"))
from model_egitimi import hazirla_X, temizle, YAS_MAP  # noqa: E402

MODEL_DOSYA = KOK / "models" / "kira_model.joblib"


def tahmin_et(paket: dict, girdi: dict) -> tuple[float, float, float]:
    """Tek bir ilan girdisinden (alt, tahmin, üst) kira aralığını döndürür.

    Aralık log-ölçekli RMSE ile çarpımsal banttır (fiyatla orantılı).
    Bu fonksiyon Streamlit'ten bağımsız test edilebilir.
    """
    df = pd.DataFrame([girdi])
    temiz = temizle(df)
    X = hazirla_X(temiz, paket["encoders"], paket["dummy_kolonlar"])
    log_tahmin = float(paket["model"].predict(X)[0])
    tahmin = float(np.expm1(log_tahmin))
    rmse_log = paket["metrikler"]["rmse_log"]
    alt = float(np.expm1(log_tahmin - rmse_log))
    ust = float(np.expm1(log_tahmin + rmse_log))
    return alt, tahmin, ust


@st.cache_resource
def model_yukle() -> dict:
    import joblib
    return joblib.load(MODEL_DOSYA)


def main():
    st.set_page_config(page_title="Kira Tahmin", page_icon="🏠", layout="centered")
    st.title("🏠 Türkiye Kira Tahmin Uygulaması")

    if not MODEL_DOSYA.exists():
        st.error("Model bulunamadı. Önce şunu çalıştırın:  `python src/model_egitimi.py`")
        st.stop()

    paket = model_yukle()
    agac = paket["konum_agaci"]
    ozet = paket["sayisal_ozet"]
    m = paket["metrikler"]

    st.caption(
        f"Model doğruluğu (sızıntısız): Test R² ≈ {m['test_r2']:.2f} · "
        f"CV R² ≈ {m['cv_r2']:.2f}"
    )

    # --- Konum: kademeli dropdown (gerçek veriden) ---
    st.subheader("Konum")
    c1, c2, c3 = st.columns(3)
    il = c1.selectbox("İl", list(agac.keys()))
    ilce = c2.selectbox("İlçe", list(agac[il].keys()))
    mahalle = c3.selectbox("Mahalle", agac[il][ilce])

    # --- Konut özellikleri ---
    st.subheader("Konut Özellikleri")
    c1, c2 = st.columns(2)
    net = c1.number_input("Net metrekare", min_value=10, max_value=1000,
                          value=int(ozet["net_metrekare"]["medyan"]))
    brut = c2.number_input("Brüt metrekare", min_value=10, max_value=1000,
                           value=int(ozet["brut_metrekare"]["medyan"]))
    c1, c2, c3 = st.columns(3)
    oda = c1.number_input("Oda sayısı", min_value=0, max_value=10, value=2)
    salon = c2.number_input("Salon sayısı", min_value=0, max_value=3, value=1)
    banyo = c3.number_input("Banyo sayısı", min_value=0, max_value=6, value=1)

    c1, c2 = st.columns(2)
    yas = c1.selectbox("Binanın yaşı", paket["yas_secenekleri"], index=2)
    kat = c2.number_input("Bulunduğu kat", min_value=-4, max_value=50, value=2)
    bina_kat = st.number_input("Binanın kat sayısı", min_value=1, max_value=60,
                               value=int(ozet["binanin_kat_sayisi"]["medyan"]))

    c1, c2 = st.columns(2)
    isitma = c1.selectbox("Isıtma tipi", paket["isitma_tipleri"])
    kullanim = c2.selectbox("Kullanım durumu", paket["kullanim_durumlari"])

    c1, c2 = st.columns(2)
    site = c1.radio("Site içerisinde mi?", ["Hayır", "Evet"], horizontal=True)
    esya = c2.radio("Eşya durumu", ["Boş", "Eşyalı"], horizontal=True)

    if st.button("Kirayı Tahmin Et", type="primary", use_container_width=True):
        # temizle() ham (Türkçe kolonlu) formatı bekler -> aynı şemada satır kur
        girdi = {
            "Net Metrekare": f"{net} m²",
            "Brüt Metrekare": f"{brut} m²",
            "Oda Sayısı": f"{oda}+{salon}",
            "Binanın Yaşı": yas,
            "Bulunduğu Kat": f"{kat}.Kat",
            "Binanın Kat Sayısı": bina_kat,
            "Isıtma Tipi": isitma,
            "Kullanım Durumu": kullanim,
            "Tapu Durumu": "Kat Mülkiyeti",
            "Site İçerisinde": site,
            "Banyo Sayısı": str(banyo),
            "Eşya Durumu": esya,
            "konum": f"{il} - {ilce} - {mahalle} Mahallesi",
        }
        alt, tahmin, ust = tahmin_et(paket, girdi)

        st.markdown("### Tahmini Aylık Kira")
        st.metric(label=f"{il} / {ilce} / {mahalle}", value=f"{tahmin:,.0f} TL")
        st.info(f"Tahmini aralık: **{alt:,.0f} TL – {ust:,.0f} TL**")
        st.caption(
            "Aralık, modelin tipik hata payına (log-ölçekli RMSE) dayanır; "
            "gerçek kira bu bandın dışında olabilir."
        )


if __name__ == "__main__":
    main()

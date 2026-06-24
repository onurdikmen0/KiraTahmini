# 🏠 Kira Tahmini — Makine Öğrenmesi ile Konut Kira Fiyatı Tahmini

Türkiye genelindeki **gerçek kiralık konut ilanlarından** öğrenen, aylık kira
bedelini tahmin eden uçtan uca bir veri bilimi projesi. Web kazımadan model
eğitimine ve etkileşimli web arayüzüne kadar tüm hattı içerir.

> İstanbul Topkapı Üniversitesi — Bitirme Projesi · Onur Dikmen

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-EB5E28)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Test R²](https://img.shields.io/badge/Test%20R²-0.71-2E8B57)

---

## 📌 Proje Hakkında

Türkiye'de kira bedelleri çoğunlukla öznel değerlendirmelerle belirlenir ve
piyasada ciddi bir bilgi asimetrisi vardır. Bu proje, **emlakjet.com**'dan
toplanan gerçek ilan verisiyle nesnel ve ölçeklenebilir bir kira tahmini sunar.

Akış:

```
web kazıma → veri birleştirme → temizlik & öznitelik mühendisliği
   → coğrafi zenginleştirme → XGBoost model eğitimi → Streamlit arayüzü
```

## ✨ Öne Çıkanlar

- **77 ile ait 28.259 gerçek ilan** web kazıma ile toplandı (Ağustos 2025).
- **Veri sızıntısız** değerlendirme: konum kodlaması her fold'da yalnızca eğitim
  verisinden öğrenilir (eski sürümdeki sahte R²≈0.99 sorunu giderildi).
- Konum (il/ilçe/mahalle) için **hiyerarşik + smoothing'li hedef kodlama**.
- **Coğrafi öznitelikler**: 462 ilçe için enlem/boylam ve il merkezine uzaklık
  (OpenStreetMap/Nominatim ile geocoding).
- **Streamlit web arayüzü**: kullanıcı girdisinden anlık tahmin + güven aralığı.

## 📊 Sonuçlar

| Metrik | Değer |
|--------|------:|
| Test R² | **0,714** |
| 5-katlı CV R² | **0,716** |
| RMSE | 4.034 TL |
| MAE | 2.942 TL |
| MAPE | %16,2 |

> Temiz veri: 25.188 gözlem · 77 il. Öznitelik önemine göre kirayı en çok
> belirleyen faktör **konum** (ilçe/mahalle), ardından ısıtma tipi, banyo
> sayısı ve konut büyüklüğüdür.

## 🗂️ Veri Seti

Ham veri seti Kaggle'da açık erişimdedir:

**[Turkey House Rent Prices (28K Listings, 2025)](https://www.kaggle.com/datasets/onurdikmen0/turkey-house-rent-prices-28k-listings-2025)**

## 🛠️ Teknolojiler

`Python 3.11` · `pandas` · `scikit-learn` · `XGBoost` · `Streamlit` ·
`osmnx` / `geopandas` · `BeautifulSoup` / `requests`

## 📁 Proje Yapısı

```
kira_tahmin_projesi/
├── app.py                → Streamlit arayüzü
├── notebooks/            → iş akışı notebook'ları (1 → 4 sırayla)
│   ├── 1_web_scraping.ipynb
│   ├── 2_csv_merge.ipynb
│   ├── 3_veri_on_isleme.ipynb
│   └── 4_cografi_veri_isleme.ipynb
├── src/
│   ├── model_egitimi.py            → ana eğitim scripti (sızıntısız)
│   ├── cografi_zenginlestirme.py   → ilçe geocoding (osmnx/Nominatim)
│   ├── model_karsilastirma.py      → modellerin 5-fold CV kıyaslaması
│   └── veri_bilimci_el_cantasi.py  → yardımcı fonksiyonlar
├── data/                → raw / processed / geo
├── models/              → kira_model.joblib (model + encoder + metrikler)
└── requirements.txt
```

## 🚀 Kurulum ve Çalıştırma

```bash
# 1) Bağımlılıkları kur
pip install -r requirements.txt

# 2) (İsteğe bağlı) Modeli ham veriden yeniden eğit
python src/model_egitimi.py

# 3) Web arayüzünü başlat
streamlit run app.py
```

Modeli yeniden eğitmek istemezsen, hazır `models/kira_model.joblib` doğrudan
arayüz tarafından yüklenir.

## 🔬 Yöntem Notları

- Hedef değişken **log dönüşümlüdür** (`log1p`/`expm1`).
- Konum one-hot yerine **hiyerarşik hedef kodlama** ile temsil edilir
  (mahalle → ilçe → il → global fallback), yalnızca eğitim fold'undan öğrenilir.
- Aykırı değerler **IQR (1,5×)** yöntemiyle ayıklanır.
- Değerlendirme: hold-out test + **fold içinde encoder'ı yeniden fit eden 5-fold CV**.

## 🧭 Yol Haritası

- [x] Veri sızıntısını giderip modeli gerçekçi hale getirme
- [x] Coğrafi öznitelikler (ilçe lat/lon + il merkezine uzaklık)
- [x] Streamlit web arayüzü
- [ ] Daire dışındaki konut kategorileri için veri toplama
- [ ] Aynı projeyi **satılık** konutlar için tekrarlama
- [ ] Mahalle düzeyi geocoding + POI temelli öznitelikler (sahile/merkeze uzaklık)

## 👤 Yazar

**Onur Dikmen** — İstanbul Topkapı Üniversitesi, Yönetim Bilişim Sistemleri

---

## 🇬🇧 About (English)

End-to-end data science project that predicts monthly residential rent in Türkiye
from real listings scraped from emlakjet.com (28,259 listings across 77 provinces,
August 2025). Pipeline: web scraping → cleaning & feature engineering → geospatial
enrichment → leakage-free XGBoost training → Streamlit web app. **Test R² = 0.71**.
Dataset on [Kaggle](https://www.kaggle.com/datasets/onurdikmen0/turkey-house-rent-prices-28k-listings-2025).

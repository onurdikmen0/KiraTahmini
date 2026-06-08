
# CLAUDE.md

Bu dosya, bu repoda çalışırken Claude Code'a (ve diğer geliştiricilere) rehberlik eder.

## Proje Hakkında

**Kira Tahmin Projesi** — Türkiye genelinde gerçek emlak ilanı verisiyle, makine
öğrenmesi kullanarak kiralık konut fiyatı tahmini yapan bir veri bilimi projesi.

Akış: **web scraping → veri birleştirme → veri ön işleme / öznitelik üretimi →
coğrafi zenginleştirme → model eğitimi (XGBoost)**. Nihai hedef, Streamlit ile bir
web arayüzü sunmaktır. Referans: https://github.com/canbula/KiraTahmini

Veri kaynağı: emlakjet.com (kiralık daire ilanları).

## Ortam

- Python **3.11.5**, Windows + PowerShell. Bash da mevcut.
- Kurulu temel kütüphaneler: `pandas 2.2.2`, `numpy`, `scikit-learn 1.7.1`,
  `xgboost 3.0.4`, `streamlit 1.47.1`, `osmnx 2.0.6`, `geopandas 1.1.1`,
  `beautifulsoup4`, `requests`, `unidecode 1.4.0`, `seaborn`, `matplotlib`.
- Çalışma Jupyter notebook tabanlıdır; betikler genellikle notebook hücrelerinde
  yürütülür. Bağımlılıklar `requirements.txt` içinde.

## Klasör / Dosya Yapısı

Proje amaca göre klasörlere ayrılmıştır. **Notebook'lar `notebooks/` içinde
çalıştığı için veri yolları `../data/...` şeklinde relatiftir.**

```
kira_tahmin_projesi/
├── README.md  CLAUDE.md  .gitignore  requirements.txt
├── app.py       → Streamlit arayüzü (streamlit run app.py)
├── notebooks/   → iş akışı notebook'ları (1→4 sırayla)
├── src/         → yeniden kullanılan Python kodu (eğitim + yardımcılar)
├── data/        → raw / processed / geo
├── models/      → eğitilmiş model (kira_model.joblib)
└── other/       → arşiv (kullanılmıyor)
```

### İş akışı notebook'ları — `notebooks/` (sırayla)
- `1_web_scraping.ipynb` — emlakjet.com'dan ilan linklerini ve detaylarını çeker.
  URL listesi `np.array_split` ile 9 parçaya (`p1..p9`) bölünüp tek tek çekilir
  (`all_ads`). Çıktı: `../data/raw/kira/turkiye_kira_p1..p9.csv`.
- `2_csv_merge.ipynb` — `../data/raw/kira/` altındaki `p1..p9` CSV'lerini birleştirir.
  Çıktı: `../data/raw/turkey_rent.csv`.
- `3_veri_on_isleme.ipynb` — **ana notebook** (~206 hücre). Temizlik, eksik veri,
  öznitelik mühendisliği ve XGBoost model eğitimi burada.
  Çıktı: `../data/processed/temiz_turkey_rent.csv`.
- `4_cografi_veri_isleme.ipynb` — `osmnx`/`geopandas` ile il/ilçe/mahalle sınırlarını
  (polygon) ve centroid lat/lon bilgisini çeker. (Şu an eksik/çalışıyor.)

### Veri dosyaları — `data/`
- `data/raw/kira/` — `turkiye_kira_p1..p9.csv` (güncel scraping parçaları).
- `data/raw/turkey_rent.csv` — birleştirilmiş ham veri (~9 MB).
- `data/processed/temiz_turkey_rent.csv` — temizlenmiş + öznitelik üretilmiş veri
  (**~150 MB**; one-hot encoding sonrası genişler — `.gitignore`'da yorum satırı olarak hazır).
- `data/geo/istanbul_mahalleler.csv` — İl/İlçe/Mahalle + Polygon + Centroid.
- `data/geo/istanbul_mahalleler_latlon.csv` — İl/İlçe/Mahalle + Lat/Lon.

### Arşiv — `other/` (kullanılmıyor, sadece referans için saklanıyor)
Ana klasörü temiz tutmak için güncel iş akışında kullanılmayan her şey `other/`
altına taşındı. **Buradaki dosyalara iş akışında bağımlılık kurma.**
- `other/data/` — şehir bazında eski ham scraping çıktıları (`<sehir>_data.csv`).
- `other/old_doc/` — eski/arşiv notebook ve CSV'ler (`birlesik.csv`, eski scraping/preprocess).
- `other/osmnx_tutorial.ipynb`, `other/yeni.ipynb` — deneme/öğrenme amaçlı notebook'lar.

### Kod — `src/`
- `src/model_egitimi.py` — **ana eğitim scripti** (sızıntısız). `temizle()` ve
  `hazirla_X()` fonksiyonları Streamlit tarafından da yeniden kullanılır.
- `src/veri_bilimci_el_cantasi.py` — yeniden kullanılan yardımcı fonksiyonlar
  (Türkçe karakter dosya adı sorun çıkardığı için ASCII'ye çevrildi):
  - `kolon_duzenleme(df)` — kolon adlarını küçük harf, `_` ve Türkçe→ASCII (`unidecode`).
  - `bos_veri_orani(df)` — kolon bazında boş veri yüzdesi.
  - `outliers(df, col_name)` — IQR (1.5×) ile aykırı değer filtreleme.

## Veri İşleme Notları (veri_on_isleme.ipynb)

- Kolon adları `kolon_duzenleme` ile normalize edilir (örn. `Net Metrekare` →
  `net_metrekare`).
- `net_metrekare` / `brut_metrekare`: `" m²"` ve binlik `"."` temizlenip sayıya çevrilir.
- `oda_sayisi`: `"2+1"` → `oda` + `salon` olarak `+`'dan ayrılır; `"Stüdyo"` → 0.
- `binanin_yasi`: aralıklar tek değere maplenir (`age_map`, örn. `"5-10"→7`).
- Eksik veri: yalnız-NaN satırlar atılır; kategorik eksikler mode ile doldurulur;
  çok eksik kolonlar (balkon detayları, takas vb.) düşürülür.
- Öznitelik üretimi (hedef): `mahalle_ortalama_kira`, `oda_basi_alan`,
  `metrekare_basi_kira`, `burut_net_orani`, `fiyat_oda_orani`. İl/ilçe/mahalle
  ortalama kira `merge` ile eklenir.
- Encoding: kategorikler için `pd.get_dummies` / `OneHotEncoder` + `ColumnTransformer`.

## Modelleme

Eğitim artık notebook yerine **`src/model_egitimi.py`** scriptinde
(`python src/model_egitimi.py`). Script ham veriden başlar, temizler, eğitir ve
artefaktları `models/kira_model.joblib`'e kaydeder (model + encoder'lar + kolon
sırası + metrikler). Streamlit bu dosyayı ve scriptteki `temizle()` / `hazirla_X()`
fonksiyonlarını yeniden kullanır.

- Model: **XGBoost** (`xgb.XGBRegressor`), hedef **log dönüşümlü** (`log1p`/`expm1`).
- Konum (il/ilçe/mahalle) one-hot yerine **hiyerarşik + smoothing'li target encoding**
  ile kodlanır (mahalle→ilçe→il→global fallback), **yalnız train fold'undan** öğrenilir.
- Metrikler: hold-out test R², train R² (overfit kontrolü) ve **fold içinde encoder'ı
  yeniden fit eden 5-fold CV**.

### Veri sızıntısı düzeltmesi (tamamlandı)
Eski notebook R² ≈ 0.99 veriyordu çünkü hedeften türetilen kolonlar eğitime giriyordu.
`src/model_egitimi.py` bunları giderdi:
- `metrekare_basi_kira` ve `fiyat_oda_orani` (fiyatı doğrudan içeriyorlardı) **hiç üretilmiyor**.
- İl/ilçe/mahalle ortalama kira artık **sadece train'den** hesaplanıyor (eskiden tüm veriyle).
- **Güncel gerçek skor: Test R² ≈ 0.70, CV R² ≈ 0.70 (±0.005)** — gerçekçi ve kararlı.

## Streamlit Arayüzü

- `app.py` — `streamlit run app.py` ile çalışır. `models/kira_model.joblib`'i ve
  `src/model_egitimi.py`'deki `temizle()` / `hazirla_X()` fonksiyonlarını yükler.
- Konum (il/ilçe/mahalle) kademeli dropdown'larla **gerçek veriden** doldurulur
  (modeldeki `konum_agaci`); kullanıcı geçersiz konum giremez.
- Tahmin tek sayı + **çarpımsal aralık** (log-RMSE bandı) olarak gösterilir.

## Yol Haritası / Yapılacaklar

Tamamlananlar ✅
- ~~Veri sızıntısını gidererek modeli gerçekçi hale getir~~ (Test R² ≈ 0.70).
- ~~Streamlit ile web arayüzü oluştur~~ (`app.py`).

Kalanlar
- Emlakjet'te Daire dışındaki konut kategorileri için de veri çek.
- Aynı projeyi **satılık** evler için tekrarla.
- `4_cografi_veri_isleme.ipynb`'i tamamlayıp coğrafi öznitelik (lat/lon, merkeze
  uzaklık vb.) ekleyerek modeli güçlendir.

## Çalışma Notları

- Notebook'lar ve CSV'ler Türkçe karakter içerir; dosya okurken `encoding="utf-8"`
  kullan. CSV'ler büyük; `pd.read_csv(..., low_memory=False)` tercih edilir.
- `temiz_turkey_rent.csv` ~150 MB — gereksiz yere git'e commit'lemekten kaçın
  (`.gitignore` değerlendirilebilir).
- Yorum ve değişken adları Türkçedir; mevcut stile uy (snake_case, Türkçe yorum).

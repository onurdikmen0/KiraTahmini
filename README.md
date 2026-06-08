# Kira Tahmin Uygulaması

p1 : 47 dk sürdü
p2 : 51 dk
p3 : 41 dk
p4 : 34 dk
p5 : 39 dk
p6 : 40 dk
p7 : 38 dk
p8 : 37 dk
p9 : 39 dk 
6 saat 10 dk

Şuanki veriseti Türkiye geneli kiralık daireleri içeriyor.
Emlakjetden kiralık konut Daire haricindeki diğer kategoriler için de veri çekilsin.
Aynı proje Satılık evler içinde yapılsın.
---
## Project Abount
streamlit ile web arayüz oluştur,
web scraping ile webden veri çek

kira tahmin veri bilimi projesi

https://github.com/canbula/KiraTahmini/tree/master

### Kira Tahmini Uygulaması
Gerçek Verilerle Makine Öğrenmesi Modelleri Oluşturarak Kira Tahmini Yapılması

#### Proje Açıklaması
Bu proje, gerçek verilerle makine öğrenmesi modelleri oluşturarak kira tahmini yapılmasını amaçlamaktadır. Proje kapsamında, veri seti üzerinde veri ön işleme işlemleri gerçekleştirilmiş, veri seti üzerinde istatistiksel analizler yapılmış ve makine öğrenmesi modelleri uygulanmıştır.

##### Kullanılan Kütüphaneler
numpy
pandas
sklearn



Mahalleye ait ortalama kira fiyatı → modelin yerleşim etkisini öğrenmesini sağlar.
Oda başına alan, metrekare başına kira gibi yeni özellikler üret.
brüt/ net oranı → eğer brüt ve net varsa
fiyat / oda_sayisi → oda başına kira (tahmin için faydalı olabilir)

mahalle_ortalama_kira
oda_basi_alan
metrekare_basi_kira
burut_net_orani
fiyat_oda_orani

1. versiyon(çok kötü)
    R² Score: 0.000 bişeyler
    RMSE: 200000.00 bişeyer

2. verisyon 
    R² Score: 0.6042039394378662
    RMSE: 62214.635448582354

3. versiyon 
    R² Score: 0.9976809024810791
    RMSE: 388.3306419792288

--------------------------------------
Lat (Latitude – Enlem):
Lon (Longitude – Boylam):
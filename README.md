# Apple Hisse Senedi Fiyat Tahmini - LSTM Modeli

Bu proje Apple (AAPL) hisse senedi fiyatları için LSTM (Long Short-Term Memory) tabanlı bir zaman serisi tahmin modeli geliştirmeyi amaçlamaktadır. Projede veri işleme, model eğitimi ve test aşamaları ayrı modüller ve Notebook dosyaları olarak düzenlenmiştir.

---

## Proje Klasör Yapısı
.
├── data
│ ├── raw
│ │ └── AAPL_full.csv # İndirilen tam veri seti
│ └── sample
│ └── AAPL_sample.csv # Küçük örnek veri seti
├── models
│ └── lstm_seq60_units50_dr0.2_lr0.001_20250710-2132.h5 # Eğitilmiş model dosyaları
├── notebooks
│ ├── 01_train_LSTM.ipynb # Model eğitimi ve kayıt notebook'u
│ └── 02_test_LSTM.ipynb # Eğitilmiş modelin test ve tahmin notebook'u
├── src
│ ├── data_prep.py # Veri indirme, temizleme, ölçekleme fonksiyonları
│ └── model.py # LSTM modeli oluşturma, eğitim ve kayıt fonksiyonları
└── README.md # Proje açıklama dosyası

---

## Dosya Açıklamaları

### `src/data_prep.py`

- Yahoo Finance API kullanarak belirtilen hisse için veri indirir (`download_data`).
- Verideki eksik satırları siler (`clean_data`).
- Verileri MinMaxScaler ile [0,1] aralığına ölçekler (`scale_data`).
- Zaman serisi için kaydırmalı pencere (sliding window) yöntemiyle giriş ve hedef dizileri oluşturur (`create_sequences`).
- Verilerin tam ve küçük örnek (sample) CSV dosyalarını kaydeder (`save_csvs`).

### `src/model.py`

- LSTM tabanlı bir model mimarisi oluşturur (`build_lstm`).
- Modeli eğitir, erken durdurma uygular ve eğitilen modeli disk üzerine kaydeder (`train_and_save`).

### `notebooks/01_train_LSTM.ipynb`

- Veri indirir, temizler ve ölçekler.
- Kaydırmalı pencere yöntemi ile eğitim için giriş ve hedef dizilerini hazırlar.
- Eğitim ve test seti olarak veriyi böler.
- Modeli oluşturur ve eğitim yapar.
- Eğitim sırasında kayıp değerlerini grafik olarak gösterir.
- Eğitilen modeli `models` klasörüne kaydeder.
- Notebook çalıştırıldığında ara çıktı ve grafikler notebook içinde kayıtlıdır.

### `notebooks/02_test_LSTM.ipynb`

- Eğitilmiş modeli `models` klasöründen yükler.
- Test verisi üzerinde tahmin yapar.
- Tahmin sonuçlarının ölçeğini geri alır ve gerçek değerlerle karşılaştırır.
- Performans metriklerini hesaplar (MSE, RMSE, MAE).
- Gelecek 30 iş günü için tahmin yapar.
- Gerçek ve tahmin sonuçlarını grafikle gösterir.
- Notebook ara çıktı ve grafikleri kayıtlıdır.

---

## Kullanım

1. `notebooks/01_train_LSTM.ipynb` dosyasını açarak modeli eğitebilirsiniz.  
2. Eğitilen model `models` klasörüne otomatik kaydedilir.  
3. `notebooks/02_test_LSTM.ipynb` ile eğitilmiş modeli test edip tahmin sonuçlarını görebilirsiniz.  

---

## Kaynaklar ve Referanslar

- Sliding window tekniği ve zaman serisi LSTM uygulaması:  
  https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/  
- MinMaxScaler kullanımı:  
  https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html  
- Keras LSTM ve EarlyStopping dökümantasyonu:  
  https://keras.io/api/layers/recurrent_layers/lstm/  
  https://keras.io/api/callbacks/early_stopping/

---

## Sonuçlar ve Kısa Yorumlama

Farklı hiperparametrelerle eğitilen modellerin performansları değerlendirildiğinde,  
en iyi sonuçlar `seq_length=60` ve `units=50` parametrelerine sahip modelde gözlendi.  
Genel olarak, daha uzun diziler ve orta seviyede birim sayısı daha dengeli tahmin sağlıyor.  
Hiperparametre seçimi model başarımında belirleyici olmaktadır.

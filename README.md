# Finansal Zaman Serisi Tahminleri için Hibrit Yaklaşım

Bu proje hisse senedi piyasası hareketlerini tahmin etmek amacıyla **LSTM** ve **Random Forest** modellerini bir araya getiren hibrit bir yaklaşım sunmaktadır. Projede veri işleme, model eğitimi ve test aşamaları ayrı modüller ve Notebook dosyaları olarak düzenlenmiştir.

-----

## Teknik Genel Bakış

Finansal piyasalardaki öngörü gürültülü veriler nedeniyle zorlu bir problem teşkil eder. Bu proje hem kısa vadeli bağımlılıkları yakalama yeteneğine sahip olan **Uzun Kısa Süreli Bellek (LSTM) ağları** hem de doğrusal olmayan ilişkileri modelleyebilen **Rastgele Orman (Random Forest) algoritmaları** ile bu zorluğun üstesinden gelmeye çalışır. Her iki model de hisse senedi kapanış fiyatı tahminine odaklanırken farklı veri ön işleme stratejileri ve mimariler benimsenmiştir.

-----

## Mimari ve Akış

Projenin temel akışı ham veri alımı, modele özgü ön işleme, model eğitimi ve nihai performans ölçümü adımlarını içerir:

1.  **Veri Toplama ve Ön İşleme:** Gerekli finansal veriler toplanır ve her modelin gereksinimlerine göre özelleştirilmiş ön işlemelerden geçirilir.
2.  **Model Eğitimi:** LSTM ve Random Forest modelleri ayrılmış eğitim setleri üzerinde bağımsız olarak eğitilir.
3.  **Model Değerlendirme:** Eğitilmiş modeller bağımsız test setleri üzerinde tahmin performansı açısından değerlendirilir.

-----

## Kurulum ve Bağımlılıklar

Projeyi yerel ortamınızda çalıştırabilmek için aşağıdaki adımları izleyin:

1.  **Depoyu Klonlayın:**

    ```bash
    git clone https://github.com/acoplu/borsa-aslani.git
    cd borsa-aslani
    ```

2.  **Sanal Ortam Oluşturun (Önerilen):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```

3.  **Gerekli Kütüphaneleri Yükleyin:**

-----

## Proje Yapısı ve Modüller

```
borsa-aslani/
├── notebooks/
│   ├── 01_train_LSTM.ipynb         # LSTM modelinin eğitimi ve hiperparametre ayarlaması
│   ├── 02_test_LSTM.ipynb          # Eğitilmiş LSTM modelinin performans analizi
│   ├── 03_train_RF.ipynb           # Random Forest modelinin eğitimi ve hiperparametre optimizasyonu
│   ├── 04_test_RF.ipynb            # Eğitilmiş Random Forest modelinin performans analizi
├── src/
│   ├── data_prep_lstm.py           # LSTM için veri ön işleme boru hattı
│   ├── data_prep_rf.py             # Random Forest için veri ön işleme boru hattı
│   ├── lstm_model.py               # LSTM model mimarisi tanımı ve yardımcı fonksiyonları
│   ├── rf_model.py                 # Random Forest model tanımı ve yardımcı fonksiyonları
├── README.md                       # Proje tanıtım ve teknik detayları
└── requirements.txt                # Python bağımlılıkları listesi
```

### Notebooks: Deneysel Akış ve Analiz

  * **[01\_train\_LSTM.ipynb](https://github.com/acoplu/borsa-aslani/blob/main/notebooks/01_train_LSTM.ipynb)**: Bu not defteri **LSTM tabanlı regresyon modelinin** oluşturulması ve eğitimini detaylandırır. Zaman serisi verilerinin `sequence`'lara dönüştürülmesi, `MinMaxScaler` ile normalleştirilmesi ve `Sequential API` kullanılarak katmanların tanımlanması süreçleri incelenir. Modelin kaybı (MSE) ve optimizasyonu (Adam) görselleştirilir.
  * **[02\_test\_LSTM.ipynb](https://github.com/acoplu/borsa-aslani/blob/main/notebooks/02_test_LSTM.ipynb)**: Eğitilmiş LSTM modelinin test seti üzerindeki tahmin doğruluğunu değerlendirir. **Ortalama Kare Hata (RMSE)** ve **Ortalama Mutlak Hata (MAE)** gibi metrikler kullanılarak modelin performansı nicel olarak ölçülür. Gerçek ve tahmin edilen değerlerin zaman serisi grafikleri ile görsel analiz yapılır.
  * **[03\_train\_RF.ipynb](https://github.com/acoplu/borsa-aslani/blob/main/notebooks/03_train_RF.ipynb)**: **Random Forest Regressor** modelinin eğitimi ve hiperparametre optimizasyonunu kapsar. Özellik mühendisliği (örneğin hareketli ortalamalar, volatilite göstergeleri) ve veri ölçeklendirme stratejileri detaylandırılır. Modelin `n_estimators`, `max_depth` gibi ana hiperparametrelerinin ayarlanması tartışılır.
  * **[04\_test\_RF.ipynb](https://github.com/acoplu/borsa-aslani/blob/main/notebooks/04_test_RF.ipynb)**: Random Forest modelinin test seti üzerindeki tahmin performansını değerlendirmek için kullanılır. Aynı şekilde RMSE ve MAE gibi regresyon metrikleri sunulur. Modelin açıklanabilirlik özellikleri (örneğin, özellik önem dereceleri) incelenebilir.

### Kaynak (`src`) Dosyaları: Modüler Fonksiyonellik

  * **[data\_prep\_lstm.py](https://github.com/acoplu/borsa-aslani/blob/main/src/data_prep_lstm.py)**: Bu modül, LSTM ağları için özelleşmiş veri ön işleme fonksiyonlarını barındırır. Finansal zaman serilerini `look-back` pencereleri kullanarak sıralı girişlere dönüştürme (`create_sequences`), veriyi belirli bir aralığa ölçekleme (`scale_data`), ve eğitim/test setlerine ayırma gibi fonksiyonları içerir.
  * **[data\_prep\_rf.py](https://github.com/acoplu/borsa-aslani/blob/main/src/data_prep_rf.py)**: Random Forest modeline özel veri hazırlık adımlarını içeren modüldür. Bu, hisse senedi verilerinden türetilmiş ek teknik göstergelerin (örn., RSI, MACD, Bollinger Bantları) oluşturulması, `NaN` değerlerin işlenmesi ve kategorik özelliklerin tek-sıcak kodlaması (`one-hot encoding`) gibi adımları içerebilir.
  * **[lstm\_model.py](https://github.com/acoplu/borsa-aslani/blob/main/src/lstm_model.py)**: LSTM model mimarisinin programatik olarak tanımlandığı yerdir. `build_lstm_model` gibi fonksiyonlar, LSTM katmanlarının sayısı, `return_sequences` ayarı, `Dropout` katmanları ve çıktı katmanı gibi mimarinin temel bileşenlerini oluşturur. Modelin `compile` ayarları (optimizer, loss function) da burada yapılandırılabilir.
  * **[rf\_model.py](https://github.com/acoplu/borsa-aslani/blob/main/src/rf_model.py)**: Random Forest modelinin tanımını ve eğitim için yardımcı fonksiyonlarını içerir. `create_rf_model` gibi fonksiyonlar, `n_estimators`, `max_features`, `min_samples_leaf` gibi hiperparametrelerin ayarlanmasına olanak tanır. Ayrıca, modelin eğitim ve tahmin arayüzleri de burada tanımlanmıştır.

-----

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

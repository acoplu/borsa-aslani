import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# proje kök dizinin ayarlanması (notebook dışı çalıştırmak için)
project_root = os.path.abspath(os.path.join(os.getcwd(), "."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_prep_lstm import download_data, clean_data, scale_data, create_sequences

def compute_metrics(y_true, y_pred):
    # prediction'lar ile gerçek değerlerin karşılaştırılması için metrik ölçümleri
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae}

def plot_predictions(y_true, y_pred, title="Tahmin ve Gerçek Değerler"):
    # karşılaştırmaların grafik olarak çizilmesi
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Gerçek", linewidth=2)
    plt.plot(y_pred, label="Tahmin", linestyle='--', alpha=0.7)
    plt.title(title)
    plt.xlabel("Zaman")
    plt.ylabel("Kapanış Fiyatı")
    plt.legend()
    plt.grid(True)
    plt.show()

def run_demo_lstm(symbol="AAPL",
                  test_start="2024-01-01",
                  test_end="2025-07-09",
                  seq_length=60,
                  models_dir="models"):

    # veriyi indir ve temizle
    df = download_data(symbol, "2010-01-01", test_end)
    df = clean_data(df)

    test_df = df.loc[test_start:]

    # eğitimde kullanılan scaler'ı yükle
    scaler_path = os.path.join(project_root, "data", "scaler.save")
    scaler = joblib.load(scaler_path)
    print(f"Scaler yüklendi: {scaler_path}")

    # test verisini ölçeklendir
    scaled_test = scaler.transform(test_df.values)
    X_test, y_test = create_sequences(scaled_test, seq_length)

    # tüm LSTM modellerini bul
    model_files = glob.glob(os.path.join(models_dir, "*.h5"))

    results = []

    # close fiyatının scaler'daki min ve scale değerlerini al
    close_min, close_scale = scaler.min_[3], scaler.scale_[3]

    # her model için test yap
    for model_path in model_files:
        model_name = os.path.basename(model_path)
        print(f"\nModel testi: {model_name}")

        # modeli yükle ve tahmin yap
        model = load_model(model_path)
        scaled_pred = model.predict(X_test)

        # tahmin ve gerçek değerleri orijinal ölçeğe geri dönüştür
        y_test_orig = y_test * (1 / close_scale) + close_min
        y_pred_orig = scaled_pred * (1 / close_scale) + close_min

        # performans metriklerini hesapla
        metrics = compute_metrics(y_test_orig, y_pred_orig)
        print(f"MSE: {metrics['MSE']:.4f}, RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}")

        # sonuçları listeye ekle (grafik için tahminleri de saklıyoruz)
        results.append({"Model Dosyası": model_name, **metrics, "Tahminler": y_pred_orig})

        # gerçek ve tahmin değerlerini grafikle göster
        plot_predictions(y_test_orig, y_pred_orig, title=f"{symbol} - {model_name} Tahmin Sonuçları")

    # sonuçları dataframe'e çevir ve RMSE metriğine göre sırala
    results_df = pd.DataFrame(results).drop(columns=["Tahminler"]).sort_values("RMSE")

    print("\n=== Model Performans Tablosu (RMSE'ye göre sıralı) ===")
    print(results_df)

if __name__ == "__main__":
    # herhangi bir hisse senedi üzerinde buradan deneme yapılabilir
    run_demo_lstm(symbol="AAPL")

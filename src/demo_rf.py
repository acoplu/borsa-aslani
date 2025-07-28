import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# proje kök dizinin ayarlanması (notebook dışı çalıştırmak için)
project_root = os.path.abspath(os.path.join(os.getcwd(), "."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_prep_rf import download_data, clean_data, add_technical_indicators, prepare_features_targets

def compute_metrics(y_true, y_pred):
    # prediction'lar ile gerçek değerlerin karşılaştırılması için metrik ölçümleri
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae}

def plot_predictions(y_true, y_pred, title="Gerçek ve Tahmin Edilen Değerler", figsize=(12,6)):
    # gerçek ve tahmin edilen değerleri zaman ekseninde karşılaştırmalı olarak göster.
    # y_pred'in index'ini y_true ile eşitle
    if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
        y_pred = pd.Series(y_pred, index=y_true.index)
    
    plt.figure(figsize=figsize)
    plt.plot(y_true, label="Gerçek", linewidth=2)
    plt.plot(y_pred, label="Tahmin", alpha=0.7)
    plt.title(title)
    plt.xlabel("Tarih")
    plt.ylabel("Değer")
    plt.legend()
    plt.grid(True)
    plt.show()

def run_demo_rf(symbols, test_start='2024-01-01', test_end='2025-01-01', models_dir="models"):
    results = []

    for sym in symbols:
        # test verisinin indirilmesi
        df_test = download_data(sym, test_start, test_end)
        df_test = clean_data(df_test)
        df_test = add_technical_indicators(df_test)
        X_test, y_test = prepare_features_targets(df_test)

        rf_model_path = os.path.join(models_dir, f"rf_{sym}_best.joblib")

        # model dosyasının indirilmesi
        rf_model = joblib.load(rf_model_path)

        # tahmin yapılması
        y_pred_rf = rf_model.predict(X_test)

        # tahminler üzerinden performansın ölçümü
        rf_metrics = compute_metrics(y_test, y_pred_rf)
        rf_metrics.update({"Model": f"RF_{sym}"})
        results.append(rf_metrics)

        # grafiklerin oluşturulması
        plot_predictions(y_test, y_pred_rf, title=f"{sym} - RF Test Seti Tahmin Sonuçları")

    # sonuçları dataframe'e çevir ve RMSE metriğine göre sırala
    results_df = pd.DataFrame(results)
    print("\n=== Model Performans Tablosu ===")
    print(results_df.sort_values("RMSE"))

if __name__ == "__main__":
    # Dilersen buraya farklı semboller ekleyebilirsin
    run_demo_rf(symbols=['AAPL', 'MSFT', 'TSLA', 'GOOG'])

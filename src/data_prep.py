"""
veri çekme, temizleme, ölçekleme ve sequence oluşturma fonksiyonları.
"""
import yfinance as yf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

def download_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    # Yahoo Finance kullanılarak (burada Python API'si kullanılıyor)
    # "symbol" hissesine ait start-end yılları arasındaki günlük fiyat verilerini indirir
    df = yf.download(symbol, start=start, end=end, progress=False)
    return df[["Open","High","Low","Close","Volume"]]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # eksik olan verileri doğrudan dropla
    return df.dropna()

def scale_data(df: pd.DataFrame) -> (np.ndarray, MinMaxScaler):
    # veriyi normalize ettim (MinMaxScaler ile [0,1] aralığına ölçekleme)
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#:~:text=The%20transformation%20is%20given%20by%3A 
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(df.values)
    return scaled, scaler

def create_sequences(data: np.ndarray, seq_length: int) -> (np.ndarray, np.ndarray):
    # internette yaptığım araştırmada zaman serisi verilerinin kaydırmalı pencere (sliding window) tekniğiyle kullanıldığını gördüm
    # https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/#:~:text=As%20with%20one,with%20input%20and%20output%20components
    # aşağıda son seq_length günün verileri alınarak bir sonraki günün kapanış fiyatı hedef değişken olarak belirleniyor. bu işlem brownlee’nin sliding window yaklaşımı olarak geçiyor
    # https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/#:~:text=split%20a%20univariate%20sequence%20into,append%28seq_y%29%20return%20array%28X%29%2C%20array%28y
    # https://python.plainenglish.io/predicting-apple-stock-price-36f329cda530
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 3])  # close price
    return np.array(X), np.array(y)

def save_csvs(df: pd.DataFrame, symbol: str, sample_frac: float = 0.01):
    # raw CSV kaydetme
    raw_dir = os.path.join("..", "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, f"{symbol}_full.csv")
    df.to_csv(raw_path, index=True)
    print(f"Raw veri kaydedildi: {raw_path}")

    # sample CSV kaydetme
    sample_dir = os.path.join("..", "data", "sample")
    os.makedirs(sample_dir, exist_ok=True)
    sample_df = df.sample(frac=sample_frac, random_state=42).sort_index()
    sample_path = os.path.join(sample_dir, f"{symbol}_sample.csv")
    sample_df.to_csv(sample_path, index=True)
    print(f"Sample veri kaydedildi: {sample_path}")

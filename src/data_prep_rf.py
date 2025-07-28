"""
Veri çekme, temizleme, teknik göstergeler ekleme ve özellik-hedef ayırma işlemlerini içeren fonksiyonlar.
"""

import yfinance as yf
import pandas as pd
import numpy as np

def download_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    # Yahoo Finance API kullanılarak günlük fiyat verisi indirilir
    # Sütunlar: Open, High, Low, Close, Volume
    df = yf.download(symbol, start=start, end=end, progress=False)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Eksik veriler (NaN) varsa, model performansını etkilememesi için drop edilir
    return df.dropna()

# === Teknik Göstergeler ===

def SMA(series: pd.Series, window: int) -> pd.Series:
    # Simple Moving Average (Basit Hareketli Ortalama)
    return series.rolling(window=window).mean()

def EMA(series: pd.Series, window: int) -> pd.Series:
    # Exponential Moving Average (Üssel Hareketli Ortalama)
    return series.ewm(span=window, adjust=False).mean()

def RSI(series: pd.Series, window: int = 14) -> pd.Series:
    # Relative Strength Index göstergesi
    # https://www.investopedia.com/terms/r/rsi.asp
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))

def MACD(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    # MACD göstergesi (Moving Average Convergence Divergence)
    # https://www.investopedia.com/terms/m/macd.asp
    fast_ema = EMA(series, fast)
    slow_ema = EMA(series, slow)
    macd = fast_ema - slow_ema
    signal_line = EMA(macd, signal)
    hist = macd - signal_line
    return macd, signal_line, hist

def Bollinger_Bands(series: pd.Series, window: int = 20, n_std: int = 2):
    # Bollinger Bands göstergesi
    # Ortalama ± (standart sapma * katsayı)
    sma = SMA(series, window)
    std = series.rolling(window=window).std()
    upper_band = sma + n_std * std
    lower_band = sma - n_std * std
    return upper_band, lower_band

def Momentum(series: pd.Series, window: int = 10) -> pd.Series:
    # Momentum göstergesi: fiyat değişimi
    return series.diff(window)

def Rate_of_Change(series: pd.Series, window: int = 10) -> pd.Series:
    # Rate of Change (ROC): yüzde bazlı değişim oranı
    return series.pct_change(periods=window)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Girdi olarak gelen fiyat verisine çeşitli teknik analiz göstergeleri eklenir.
    - Hareketli ortalamalar
    - RSI, MACD, Bollinger Bandları
    - Momentum, ROC
    - Fiyat oranları (CO, HL)
    - Tarihsel (calendar) özellikler
    """
    df = df.copy()

    # hareketli ortalamalar
    df['SMA_5'] = SMA(df['Close'], 5)
    df['SMA_10'] = SMA(df['Close'], 10)
    df['EMA_10'] = EMA(df['Close'], 10)
    df['EMA_20'] = EMA(df['Close'], 20)

    # RSI
    df['RSI_14'] = RSI(df['Close'], 14)

    # MACD
    macd, macd_signal, macd_hist = MACD(df['Close'])
    df['MACD'] = macd
    df['MACD_signal'] = macd_signal
    df['MACD_hist'] = macd_hist

    # bollinger bandları
    upper_bb, lower_bb = Bollinger_Bands(df['Close'])
    df['BB_upper'] = upper_bb
    df['BB_lower'] = lower_bb

    # momentum ve ROC
    df['Momentum_10'] = Momentum(df['Close'], 10)
    df['ROC_10'] = Rate_of_Change(df['Close'], 10)

    # fiyat oranları
    df['HL_ratio'] = df['High'] / df['Low']       # gün içi dalgalanma
    df['CO_ratio'] = df['Close'] / df['Open']     # açılış vs kapanış oranı

    # tarihsel zaman bilgileri
    df['Day'] = df.index.day
    df['Weekday'] = df.index.weekday
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter

    # teknik göstergelerle birlikte eksik veri çıkabileceğinden dropna kullan
    df = df.dropna()
    return df

def prepare_features_targets(df: pd.DataFrame, target_col: str = 'Close') -> (pd.DataFrame, pd.Series):
    # özellik ve hedef değişken ayrımını yap
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def get_data_for_symbols(symbols: list, start: str, end: str) -> dict:
    # RF'de kullanacağım çoklu hisse senedi için veriyi indirir temizler ve teknik göstergeleri uygular.
    data = {}
    for sym in symbols:
        df = download_data(sym, start, end)
        df = clean_data(df)
        df = add_technical_indicators(df)
        data[sym] = df
    return data

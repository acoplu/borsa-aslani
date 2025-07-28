"""
lstm mimarisi, eğitim ve model kaydetme fonksiyonları.
"""
import os
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def build_lstm(input_shape, units: int, dropout: float):
    # LSTM tabanlı bir Seq model tanımladım 
    # katmanın birim sayısını "units" olarak ayarladım
    # https://arxiv.org/html/2409.14693v1#:~:text=Unlike%20traditional%20LSTMs%20that%20process,This%20makes%20them
    # burada overfit engellemek için "dropout" oranında bir dropout katmanı ekledim. 
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=units, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    return model

def train_and_save(model, X_train, y_train, save_path: str,
                   lr: float, batch_size: int=32, epochs: int=100):
    # model mean_squared_error kayıp fonksiyonu ve Adam optimizasyon algoritmasını kullanıyor
    model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=lr))
    # doğrulama kaybı 10 epoch boyunca iyileşme göstermediğinde early stopping ile eğitim sonlandırılıyor
    # https://keras.io/api/callbacks/early_stopping/#:~:text=Stop%20training%20when%20a%20monitored,metric%20has%20stopped%20improving
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1
    )
    # son olarak da modeli verilen path için kaydettim
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    return history

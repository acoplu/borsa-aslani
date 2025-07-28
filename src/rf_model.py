"""
random forest mimarisi, hiperparametre optimizasyonu, model eğitimi fonksiyonları.
"""

import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

def train_rf_model(X_train, y_train, param_distributions=None,
                   n_iter=20, cv=3, random_state=42, n_jobs=-1):
    if param_distributions is None:
        param_distributions = {
            'n_estimators': [100, 200, 300],         # ağaç sayısı (çok artarsa model yavaşlar ama genelde daha stabil olur)
            'max_depth': [None, 10, 20, 30],         # ağaçların maksimum derinliği (çok büyükse overfit riski)
            'max_features': ['sqrt', 'log2'],        # her split'te göz önünde bulundurulacak özellik sayısı
            'min_samples_split': [2, 5, 10],         # bir düğümün bölünebilmesi için gereken minimum örnek sayısı
            'min_samples_leaf': [1, 2, 4],           # bir yaprakta bulunması gereken minimum örnek sayısı
            'bootstrap': [True]                      # veri örnekleme yöntemi (bootstrap genellikle daha iyidir)
        }

    # kullanılacak olan randomforestregressor nesnesi oluşturulur
    rf = RandomForestRegressor(random_state=random_state)

    # bu fonksiyon hiperparametre optimizasyonu yapıyor aslında. search grid
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=n_iter,              # kaç farklı kombinasyon deneneceği
        cv=cv,                      # cross-validation fold sayısı
        verbose=2,                  # eğitim ilerlemesini göster (düzey 2: detaylı)
        random_state=random_state, # seed değeri
        n_jobs=n_jobs,              # paralel işlem sayısı
        scoring='neg_mean_squared_error'  # performans metriği olarak negatif MSE kullanılır (scikit-learn konvansiyonu)
    )

    # modelin eğitilmesi
    search.fit(X_train, y_train)

    # modellerden en iyi olanı al
    best_model = search.best_estimator_
    return best_model, search.best_params_, search.cv_results_

def save_model(model, save_path):
    # modeli kaydet
    # dosya yolu yoksa oluştur
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # modeli joblib ile diske yaz
    joblib.dump(model, save_path)
    print(f"Model kaydedildi: {save_path}")

def load_model(load_path):
    # modeli verilen pathten yükle
    model = joblib.load(load_path)
    print(f"Model yüklendi: {load_path}")
    return model

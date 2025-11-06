import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score
import json
import yaml

print("Starting model evaluation...")

# Загружаем данные и модель
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')
model = joblib.load('models/model.joblib')
print(f"Test data loaded: {X_test.shape}, {y_test.shape}")

# Предсказания
y_pred = model.predict(X_test)

# Вычисляем метрики
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metrics = {
    'accuracy': float(accuracy),
    'f1': float(f1)
}

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# Сохраняем метрики
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("Metrics saved to metrics.json")

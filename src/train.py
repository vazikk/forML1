import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import yaml
import os

print("Starting model training...")

# Загружаем данные
X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')
print(f"Training data loaded: {X_train.shape}, {y_train.shape}")

# Загружаем параметры
with open('params.yaml') as f:
    params = yaml.safe_load(f)

print(f"Training parameters: {params['train']}")

# Обучаем модель
model = RandomForestClassifier(
    n_estimators=params['train']['n_estimators'],
    max_depth=params['train']['max_depth'],
    random_state=params['train']['random_state']
)

model.fit(X_train, y_train)
print("Model trained successfully!")

# Сохраняем модель
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.joblib')

print("Model saved to models/model.joblib")

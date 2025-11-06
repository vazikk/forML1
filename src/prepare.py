import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import yaml
import os

print("Starting data preparation...")

# Генерируем синтетические данные
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
print(f"Generated dataset: {X.shape}, {y.shape}")

# Разделяем на train/test
with open('params.yaml') as f:
    params = yaml.safe_load(f)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=params['prepare']['test_size'],
    random_state=params['prepare']['random_state']
)

print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# Сохраняем данные
os.makedirs('data', exist_ok=True)
np.save('data/X_train.npy', X_train)
np.save('data/X_test.npy', X_test)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy', y_test)

print("Data preparation completed! Files saved to data/ directory")

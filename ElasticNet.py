import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("housing.csv")

# Handle missing values properly
df = df.copy()  # Prevent chained assignment warning
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

# One-hot encoding categorical features
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

# Define features and target
X = df.drop(columns=["median_house_value"])
y = df["median_house_value"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (Essential for ElasticNet)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Polynomial features (Captures interactions)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    "alpha": [0.1, 1, 10, 100],  # Regularization strength
    "l1_ratio": [0.1, 0.5, 0.9]  # Mix between Lasso (1) and Ridge (0)
}

elastic_net = ElasticNet(max_iter=50000, tol=0.01)  # Increased iterations & relaxed tolerance
grid_search = GridSearchCV(elastic_net, param_grid, cv=3, scoring="r2", verbose=1, n_jobs=-1)
grid_search.fit(X_train_poly, y_train)

# Best model after tuning
best_elastic_net = grid_search.best_estimator_
y_pred = best_elastic_net.predict(X_test_poly)

# Evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Best Alpha: {grid_search.best_params_['alpha']}")
print(f"Best L1 Ratio: {grid_search.best_params_['l1_ratio']}")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

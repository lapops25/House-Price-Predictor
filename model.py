import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("data/train.csv")


# ==========================================
# SELECT FEATURES
# ==========================================

features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt"
]

target = "SalePrice"

X = df[features].copy()
y = df[target]


# ==========================================
# HANDLE MISSING VALUES
# ==========================================

X["GarageCars"] = X["GarageCars"].fillna(0)
X["TotalBsmtSF"] = X["TotalBsmtSF"].fillna(0)


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ==========================================
# LINEAR REGRESSION
# ==========================================

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_pred)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))
linear_r2 = r2_score(y_test, linear_pred)

print("\nLinear Regression Evaluation:")
print("MAE:", linear_mae)
print("RMSE:", linear_rmse)
print("R² Score:", linear_r2)


# ==========================================
# RANDOM FOREST
# ==========================================

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# Train model
rf_model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")


# ==========================================
# RANDOM FOREST EVALUATION
# ==========================================

# Predict prices for unseen test data
rf_pred = rf_model.predict(X_test)

# Calculate evaluation metrics
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest Evaluation:")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R² Score:", rf_r2)


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

feature_importance = dict(
    zip(features, rf_model.feature_importances_)
)

print("\nFeature Importance:")

for feature, importance in feature_importance.items():
    print(f"{feature}: {importance:.4f}")


# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(rf_model, "house_price_model.pkl")

print("\nRandom Forest model saved successfully!")

print("\nFeatures used by the model:")
print(features)

print("\nNumber of features:", len(features))
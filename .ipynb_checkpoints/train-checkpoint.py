from preprocess import load_and_preprocess_data
from features import build_features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import joblib
import numpy as np
import os

# Create models directory if not exists
os.makedirs("models", exist_ok=True)

print("Loading and preprocessing data...")
df = load_and_preprocess_data()

print("Building features...")
X, tfidf = build_features(df["full_text"])

# Targets
le = LabelEncoder()
y_class = le.fit_transform(df["problem_class"])
y_score = df["problem_score"]

# ---------- CLASSIFICATION ----------
print("Training classification model...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

clf = LogisticRegression(max_iter=1000, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {acc:.4f}")

# ---------- REGRESSION ----------
print("Training regression model...")
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_score, test_size=0.2, random_state=42
)

reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
reg.fit(X_train_r, y_train_r)

y_pred_r = reg.predict(X_test_r)
mae = mean_absolute_error(y_test_r, y_pred_r)
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# ---------- SAVE MODELS ----------
print("Saving models...")
joblib.dump(tfidf, "models/tfidf.pkl")
joblib.dump(clf, "models/classifier.pkl")
joblib.dump(reg, "models/regressor.pkl")
joblib.dump(le, "models/label_encoder.pkl")

print("Training complete. Models saved successfully.")

import  joblib, tensorflow as tf
import sys, os
# Add project root (D:\VSCode\HealthCare) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils.data import load_csv
from utils.Preprocess import build_preprocessor, build_ml_pipeline
from utils.metrics import eval_classification

DATA_PATH = "data/risk.csv"   # update to your file
TARGET = "Heart_Attack_Risk"               # 0/1
CAT_COLS = ["State_Name", "Gender", "Health_Insurance"]  # update
NUM_COLS = ["Age", "Cholesterol_Level", "Systolic_BP"]      # update

os.makedirs("models", exist_ok=True)

df = load_csv(DATA_PATH)
X = df[CAT_COLS + NUM_COLS]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

preprocessor = build_preprocessor(CAT_COLS, NUM_COLS)
ml_estimator = LogisticRegression(max_iter=1000, class_weight="balanced")
risk_ml_pipeline = build_ml_pipeline(preprocessor, ml_estimator)

risk_ml_pipeline.fit(X_train, y_train)
y_pred_ml = risk_ml_pipeline.predict(X_test)
y_proba_ml = risk_ml_pipeline.predict_proba(X_test)[:, 1]
ml_metrics = eval_classification(y_test, y_pred_ml, y_proba_ml)
print(f"[ML] Metrics: {ml_metrics}")

ml_path = "models/risk_ml_pipeline.pkl"
joblib.dump(risk_ml_pipeline, ml_path)
print(f"[INFO] ML pipeline saved at: {os.path.abspath(ml_path)}")

# DL
# --- DL preprocessing ---
dl_preprocessor = build_preprocessor(CAT_COLS, NUM_COLS)
dl_preprocessor.fit(X_train)

# Transform into dense arrays (Keras needs NumPy arrays, not sparse matrices)
X_train_dl = dl_preprocessor.transform(X_train).toarray()
X_test_dl  = dl_preprocessor.transform(X_test).toarray()

input_dim = X_train_dl.shape[1]
risk_dl_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
risk_dl_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
risk_dl_model.fit(X_train_dl, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)

dl_proba = risk_dl_model.predict(X_test_dl).ravel()
dl_pred = (dl_proba > 0.5).astype(int)
dl_metrics = eval_classification(y_test, dl_pred, dl_proba)
print(f"[DL] Metrics: {dl_metrics}")

dl_path = "models/risk_dl_model.h5"
prep_path = "models/risk_preprocessor.pkl"
risk_dl_model.save(dl_path)
joblib.dump(dl_preprocessor, prep_path)
print(f"[INFO] DL model saved at: {os.path.abspath(dl_path)}")
print(f"[INFO] Preprocessor saved at: {os.path.abspath(prep_path)}")
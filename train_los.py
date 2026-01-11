import  joblib, tensorflow as tf
import sys, os
# Add project root (D:\VSCode\HealthCare) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils.data import load_csv
from utils.Preprocess import build_preprocessor
from utils.metrics import eval_regression

DATA_PATH = "data/los.csv"    # update to your file
TARGET = "Hospital_Stay"     # update
CAT_COLS = ["Location", "Time"]                 # update
NUM_COLS = ["MRI_Units", "CT_Scanners", "Hospital_Beds"]  # update

os.makedirs("models", exist_ok=True)

df = load_csv(DATA_PATH)
X = df[CAT_COLS + NUM_COLS]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = build_preprocessor(CAT_COLS, NUM_COLS)
ml_estimator = RandomForestRegressor(n_estimators=300, random_state=42)
los_ml_pipeline = Pipeline(steps=[("prep", preprocessor), ("model", ml_estimator)])

los_ml_pipeline.fit(X_train, y_train)
y_pred_ml = los_ml_pipeline.predict(X_test)
ml_metrics = eval_regression(y_test, y_pred_ml)
print(f"[ML] Metrics: {ml_metrics}")

ml_path = "models/los_ml_pipeline.pkl"
joblib.dump(los_ml_pipeline, ml_path)
print(f"[INFO] ML pipeline saved at: {os.path.abspath(ml_path)}")

# --- DL preprocessing ---
dl_preprocessor = build_preprocessor(CAT_COLS, NUM_COLS)
dl_preprocessor.fit(X_train)

# Convert sparse matrices to dense arrays
X_train_dl = dl_preprocessor.transform(X_train).toarray()
X_test_dl  = dl_preprocessor.transform(X_test).toarray()

input_dim = X_train_dl.shape[1]
los_dl_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="linear"),
])
los_dl_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
los_dl_model.fit(X_train_dl, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)

dl_pred = los_dl_model.predict(X_test_dl).ravel()
dl_metrics = eval_regression(y_test, dl_pred)
print(f"[DL] Metrics: {dl_metrics}")

dl_path = "models/los_dl_model.h5"
prep_path = "models/los_preprocessor.pkl"
los_dl_model.save(dl_path)
joblib.dump(dl_preprocessor, prep_path)
print(f"[INFO] DL model saved at: {os.path.abspath(dl_path)}")
print(f"[INFO] Preprocessor saved at: {os.path.abspath(prep_path)}")
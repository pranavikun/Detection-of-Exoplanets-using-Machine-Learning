import pandas as pd
import joblib
import time
import os
import json
from datetime import datetime

# Load model, scaler, and feature list
model = joblib.load('exo_best_model.pkl')
scaler = joblib.load('exo_scaler.pkl')
features = joblib.load('exo_features.pkl')

# Paths
original_csv_path = 'original_data.csv'
local_csv_path = 'local_data.csv'
json_path = 'results.json'

# Initialize
last_processed_original = -1
last_processed_local = -1

print(f"[{datetime.now().strftime('%a %b %d %H:%M:%S %Y')}] Monitoring 'original_data.csv' and 'local_data.csv' for new entries...\n")

def write_to_json(pred_confidence):
    """
    Append the latest prediction confidence to results.json for visualization.
    """
    entry = {"prediction": float(pred_confidence)}  
    try:
        with open(json_path, 'r') as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        results = []

    results.append(entry)
    with open(json_path, 'w') as f:
        json.dump(results[-50:], f)  # Keep only the latest 50 entries for better performance

def process_and_predict(df, csv_name, last_processed_index):
    """
    Process and predict on new data rows.
    """
    if len(df) > last_processed_index + 1:
        new_rows = df.iloc[last_processed_index + 1:]

        for i, row in new_rows.iterrows():
            try:
                row_df = pd.DataFrame([row])

                # One-hot encode
                row_encoded = pd.get_dummies(row_df, columns=['koi_tce_delivname'], prefix='tce_deliv')

                # Match feature columns
                for col in features:
                    if col not in row_encoded.columns:
                        row_encoded[col] = 0
                row_encoded = row_encoded[features]

                # Scale & Predict
                scaled = scaler.transform(row_encoded)
                pred_label = model.predict(scaled)[0]
                pred_prob = model.predict_proba(scaled)[0][1]  # Confidence of class 1

                label = "CONFIRMED" if pred_label == 1 else "NOT CONFIRMED"
                print(f"[{datetime.now().strftime('%a %b %d %H:%M:%S %Y')}] Row {i+1:<3} from {csv_name} - {label} (Confidence: {pred_prob:.4f})")

                write_to_json(pred_prob)

            except Exception as row_error:
                print(f"[{datetime.now().strftime('%a %b %d %H:%M:%S %Y')}] Error on row {i+1} from {csv_name}: {row_error}")

        last_processed_index = len(df) - 1
    else:
        print(f"[{datetime.now().strftime('%a %b %d %H:%M:%S %Y')}] No new rows detected in {csv_name}. Waiting...\n")

    return last_processed_index

# Main loop
while True:
    try:
        if os.path.exists(original_csv_path):
            df_original = pd.read_csv(original_csv_path)
            last_processed_original = process_and_predict(df_original, 'original_data.csv', last_processed_original)

        if os.path.exists(local_csv_path):
            df_local = pd.read_csv(local_csv_path)
            last_processed_local = process_and_predict(df_local, 'local_data.csv', last_processed_local)

    except Exception as e:
        print(f"[{datetime.now().strftime('%a %b %d %H:%M:%S %Y')}] Error during prediction: {e}")

    time.sleep(3)

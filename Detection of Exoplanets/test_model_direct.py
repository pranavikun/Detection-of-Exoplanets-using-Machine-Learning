import pandas as pd
import joblib

# Load saved items
model = joblib.load('exo_best_model.pkl')
scaler = joblib.load('exo_scaler.pkl')
features = joblib.load('exo_features.pkl')  # 👈 Load saved feature names

# Sample data (you can change this)
sample = {
    'koi_fpflag_nt': 0,
    'koi_fpflag_ss': 0,
    'koi_fpflag_co': 0,
    'koi_fpflag_ec': 0,
    'koi_period': 10.3,                 # Stable orbital period
    'koi_time0bk': 135.5,
    'koi_impact': 0.1,                  # Central transit
    'koi_duration': 3.5,
    'koi_depth': 950,                  # Typical of a Neptune/Earth-sized planet
    'koi_prad': 2.1,                   # Radius in Earth radii
    'koi_teq': 800,                    # Equilibrium temperature
    'koi_insol': 1.2,                  # Insolation similar to Earth
    'koi_model_snr': 50.0,            # Very strong signal-to-noise
    'koi_tce_plnt_num': 1,
    'koi_steff': 5700,                # Solar-type star
    'koi_slogg': 4.4,
    'koi_srad': 1.0,                  # Radius in Solar radii
    'koi_kepmag': 13.2,
    'tce_deliv_tce_deliv_2021': 1     # Ensure this matches your one-hot encoded model features
}

# Prepare input
X = pd.DataFrame([sample])

# Add missing columns if needed
for col in features:
    if col not in X.columns:
        X[col] = 0

X = X[features]  # Ensure correct order
scaled = scaler.transform(X)
pred = model.predict(scaled)

print("Prediction:", " CONFIRMED Exoplanet" if pred[0] == 1 else " NOT CONFIRMED")

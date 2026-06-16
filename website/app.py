from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Load ML model and related files
model = joblib.load('exo_best_model.pkl')
scaler = joblib.load('exo_scaler.pkl')
features = joblib.load('exo_features.pkl')

# Home Page
@app.route('/')
def home():
    return render_template('home.html')

# Introduction Page
@app.route('/introduction')
def introduction():
    return render_template('introduction.html')

# Features Page
@app.route('/features')
def features_page():
    return render_template('features.html')

# Predict Page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            if request.is_json:
                data = request.get_json()
                input_data = [data[feature] for feature in features]
            else:
                input_data = [float(request.form[feature]) for feature in features]

            input_array = np.array(input_data).reshape(1, -1)
            scaled = scaler.transform(input_array)
            pred = model.predict(scaled)[0]
            proba = model.predict_proba(scaled)[0]

            result = {
                "prediction": "✅ Confirmed Exoplanet" if pred == 1 else "❌ False Positive",
                "proba": proba.tolist()
            }

            return jsonify(result) if request.is_json else render_template(
                'predict.html',
                features=features,
                prediction=result["prediction"],
                proba=result["proba"]
            )

        except Exception as e:
            error = f"Error: {str(e)}"
            return jsonify({"error": error}) if request.is_json else render_template(
                'predict.html',
                features=features,
                prediction=error
            )

    return render_template('predict.html', features=features)

# Additional Informational Pages
@app.route('/finding_planets.html')
def finding_planets():
    return render_template('finding_planets.html')

@app.route('/Exoplanet_overview.html')
def exoplanet_overview():
    return render_template('Exoplanet_overview.html')

@app.route('/ML_importance.html')
def ml_importance():
    return render_template('ML_importance.html')

@app.route('/search_for_life.html')
def search_for_life():
    return render_template('search_for_life.html')

if __name__ == '__main__':
    app.run(debug=True)
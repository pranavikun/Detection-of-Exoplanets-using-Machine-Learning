from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True  # For development

# Load model assets
model = joblib.load('exo_best_model.pkl')
scaler = joblib.load('exo_scaler.pkl')
features = joblib.load('exo_features.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Handle both form and AJAX requests
            if request.is_json:
                data = request.get_json()
                input_data = [data[feature] for feature in features]
            else:
                input_data = [float(request.form[feature]) for feature in features]
            
            # Process in memory
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

if __name__ == '__main__':
    app.run(debug=True)
# convert.py
import joblib
import json
import os
import numpy as np
from sklearn.base import is_classifier
import warnings
warnings.filterwarnings("ignore")

def convert_to_edge_format():
    try:
        # 1. Create static directory if it doesn't exist
        os.makedirs('static', exist_ok=True)
        
        # 2. Load original assets with verification
        required_files = ['exo_best_model.pkl', 'exo_scaler.pkl', 'exo_features.pkl']
        for f in required_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Required file missing: {f}")

        model = joblib.load('exo_best_model.pkl')
        scaler = joblib.load('exo_scaler.pkl')
        features = joblib.load('exo_features.pkl')
        
        # 3. Prepare model metadata
        model_info = {
            'features': features,
            'is_classifier': is_classifier(model),
            'model_type': type(model).__name__,
            'scaler_params': {
                'mean': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else [0]*len(features),
                'scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else [1]*len(features)
            }
        }

        # 4. Handle different model types
        if hasattr(model, 'coef_'):
            model_info['weights'] = {
                'coef': model.coef_.tolist(),
                'intercept': model.intercept_.tolist() if hasattr(model, 'intercept_') else [0]
            }
        elif hasattr(model, 'feature_importances_'):
            model_info['feature_importances'] = model.feature_importances_.tolist()
        
        # 5. Save for JavaScript implementation
        output_path = os.path.join('static', 'model_params.json')
        with open(output_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f" Successfully converted to edge format")
        print(f"Saved to: {os.path.abspath(output_path)}")
        
        # 6. Verify the output file was created
        if not os.path.exists(output_path):
            raise RuntimeError("Output file was not created successfully")
            
        return True
        
    except Exception as e:
        print(f" Conversion failed: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure all required files exist in the same directory:")
        print("   - exo_best_model.pkl")
        print("   - exo_scaler.pkl") 
        print("   - exo_features.pkl")
        print("2. Check your Python version (3.8+ recommended)")
        print("3. Verify you have write permissions in this directory")
        return False

if __name__ == "__main__":
    convert_to_edge_format()
import streamlit as st
import json
import time
import pandas as pd

# Set up the page configuration
st.set_page_config(page_title="Monitoring Data for Exoplanet Detection", layout="wide")

# Title of the Dashboard
st.title("Monitoring Data for Exoplanet Detection")

# Create placeholders for the chart and text
chart = st.empty()
placeholder = st.empty()

# Initialize empty list to store all predictions for plotting
all_predictions = []
timestamps = []

# Function to load data from the results.json
def load_predictions():
    try:
        with open('results.json', 'r') as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        return []

# Loop to keep checking for new predictions
while True:
    results = load_predictions()

    if results:
        # Loop through all the rows in the results (including new ones)
        for latest in results:
            pred_confidence = latest['prediction']
            
            # Use the current timestamp or a default one if missing
            timestamp = latest.get('timestamp', str(time.time()))  # Use current time if 'timestamp' is missing

            # Update the list of all predictions
            all_predictions.append(pred_confidence)
            timestamps.append(timestamp)

            # Create a DataFrame for visualization
            df = pd.DataFrame({
                'Timestamp': timestamps,
                'Prediction Confidence': all_predictions
            })

            # Update the chart with all predictions (this will continuously expand as predictions are added)
            chart.line_chart(df.set_index('Timestamp'))

        # Display the latest prediction with its label
        label = "CONFIRMED" if pred_confidence > 0.5 else "NOT CONFIRMED"
        placeholder.write(f"Latest Prediction: {label} (Confidence: {pred_confidence:.4f})")
    else:
        # If no predictions, show waiting message
        placeholder.write("Waiting for predictions from edge device...")

    # Refresh every 1 second to get new data and update the chart
    time.sleep(1)
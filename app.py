from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


app = Flask(__name__)
app.secret_key = "replace-with-a-random-secret"  # for flash messages

# Load model & scaler (ensure these files exist in the same folder)
MODEL_PATH = "detector_model.pkl"
SCALER_PATH = "scaler.pkl"

try:
    detector = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    detector = None
    scaler = None
    print("Error loading model/scaler:", e)

REQUIRED_FEATURES = ["Transaction_Amount", "Average_Transaction_Amount", "Frequency_of_Transactions"]

def predict_row(values):
    """
    values: 2D array-like shape (1, 3)
    returns: "Anomaly Detected" or "Anomaly Not Detected"
    """
    X = np.array(values, dtype=float)
    X_scaled = scaler.transform(X)
    pred = detector.predict(X_scaled)[0]
    return "Anomaly Detected" if pred == -1 else "Anomaly Not Detected"

@app.route("/", methods=["GET", "POST"])
def index():
    if detector is None or scaler is None:
        return render_template("index.html", error="Model or scaler not found. Make sure detector_model.pkl and scaler.pkl are in the project folder.")

    # Manual input form submission
    if request.method == "POST" and request.form.get("action") == "manual":
        try:
            amount = float(request.form.get("amount", "").strip())
            avg = float(request.form.get("avg_amount", "").strip())
            freq = float(request.form.get("frequency", "").strip())
        except ValueError:
            flash("Please enter valid numeric values for all fields.", "error")
            return redirect(url_for("index"))

        try:
            result = predict_row([[amount, avg, freq]])
            return render_template("index.html", manual_result=result, manual_values=(amount, avg, freq))
        except Exception as e:
            flash(f"Prediction error: {e}", "error")
            return redirect(url_for("index"))

    # CSV upload form submission
    if request.method == "POST" and request.form.get("action") == "upload":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("Please select a CSV file to upload.", "error")
            return redirect(url_for("index"))

        try:
            df = pd.read_csv(file)
        except Exception as e:
            flash(f"Failed to read CSV: {e}", "error")
            return redirect(url_for("index"))

        # Check required columns
        missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
        if missing:
            flash(f"Missing columns in CSV: {', '.join(missing)}", "error")
            return redirect(url_for("index"))

        try:
            X = df[REQUIRED_FEATURES].astype(float).values
            X_scaled = scaler.transform(X)
            preds = detector.predict(X_scaled)
            df["Prediction"] = ["Anomaly Detected" if p == -1 else "Anomaly Not Detected" for p in preds]
            # summary counts
            counts = df["Prediction"].value_counts().to_dict()
            # convert df to html table (bootstrap-less simple styling)
            table_html = df.to_html(classes="results-table", index=False, border=0, justify="center")
            return render_template("index.html", table_html=table_html, counts=counts)
        except Exception as e:
            flash(f"Error during prediction: {e}", "error")
            return redirect(url_for("index"))

    # GET
    return render_template("index.html")

if __name__ == "__main__":
    # Run on localhost:5000
    app.run(debug=True)


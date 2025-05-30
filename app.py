from flask import Flask, render_template, request, send_from_directory, url_for
import os, uuid, json, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import Dans_Diffraction as dif
from scipy.signal import find_peaks
import requests


def download_model(url, path):
    if not os.path.exists(path):
        print(f"Downloading {path} from {url}...")
        r = requests.get(url, stream=True)
        with open(path, 'wb') as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print(f"{path} downloaded.")
    else:
        print(f"{path} already exists. Skipping download.")

def is_valid_h5(filepath):
    """Check for valid HDF5 signature."""
    with open(filepath, 'rb') as f:
        sig = f.read(8)
        return sig == b'\x89HDF\r\n\x1a\n'


# Check if they are valid .h5 files
if not is_valid_h5("organic_model.h5"):
    raise Exception("❌ organic_model.h5 is not a valid HDF5 file!")
if not is_valid_h5("inorganic_model.h5"):
    raise Exception("❌ inorganic_model.h5 is not a valid HDF5 file!")


download_model(os.getenv("ORGANIC_MODEL_URL"), "organic_model.h5")
download_model(os.getenv("INORGANIC_MODEL_URL"), "inorganic_model.h5")



app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

organic_model = load_model("organic_model.h5")
inorganic_model = load_model("inorganic_model.h5")
scaler_y = joblib.load("scaler_y.pkl")
print("✅ Models loaded successfully.")

with open("organic_columns.json") as f:
    organic_columns = [float(x) for x in json.load(f)]
with open("inorganic_columns.json") as f:
    inorganic_columns = [float(x) for x in json.load(f)]

@app.route("/", methods=["GET", "POST"])
def index():
    compound_name = None  # Default for GET request

    if request.method == "POST":
        file = request.files["input_file"]
        model_type = request.form["model_type"]
        high_accuracy = "high_accuracy" in request.form
        compound_name = request.form.get("compound_name", "Unknown Compound")

        filename = f"{uuid.uuid4()}_{file.filename}"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        model = organic_model if model_type == "organic" else inorganic_model
        columns = organic_columns if model_type == "organic" else inorganic_columns
        theta_min, theta_max = (3, 110) if high_accuracy else ((5, 100) if model_type == "organic" else (10, 100))
        inverse = model_type == "inorganic"

        try:
            if filename.endswith(".cif"):
                df, x, y = process_cif_to_row(theta_min, theta_max, columns, save_path)
                cif_url = url_for("serve_uploads", filename=filename)
            elif filename.endswith(".xlsx"):
                df, x, y = process_excel_to_row(theta_min, theta_max, columns, save_path)
                cif_url = None
            else:
                return render_template("index.html", bandgap=None, plot_url=None, cif_url=None,
                                       compound_name=compound_name, error="Unsupported file format")

            input_array = df.values.reshape(1, len(columns), 1)
            prediction = model.predict(input_array, verbose=0)
            bandgap = scaler_y.inverse_transform(prediction)[0][0] if inverse else prediction[0][0]

            plot_path = os.path.join(UPLOAD_FOLDER, filename + ".png")
            plot_xrd(x, y, plot_path)
            plot_url = url_for("serve_uploads", filename=filename + ".png")

            return render_template("index.html",
                                   bandgap=f"{bandgap:.4f}",
                                   plot_url=plot_url,
                                   cif_url=cif_url,
                                   compound_name=compound_name)

        except Exception as e:
            return f"Error: {str(e)}", 500

    return render_template("index.html",
                           bandgap=None,
                           plot_url=None,
                           cif_url=None,
                           compound_name=None)


@app.route("/uploads/<path:filename>")
def serve_uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

def process_excel_to_row(theta_min, theta_max, col_list, path):
    df = pd.read_excel(path)
    theta = df.iloc[:, 0].values
    intensity = df.iloc[:, 1].values
    norm = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    row = {col: np.nan for col in col_list}
    for t, i in zip(theta, norm):
        if theta_min <= t <= theta_max:
            closest = min(col_list, key=lambda x: abs(x - t))
            row[closest] = i
    return pd.DataFrame([row]).fillna(0), theta, norm

def process_cif_to_row(theta_min, theta_max, col_list, path):
    xtl = dif.Crystal(path)
    xtl.Scatter.setup_scatter("x-ray", wavelength_a=1.5406, output=False)
    theta, intensity, _ = xtl.Scatter.powder()
    mask = (theta >= theta_min) & (theta <= theta_max)
    theta = theta[mask]
    intensity = intensity[mask]
    norm = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    row = {col: np.nan for col in col_list}
    for t, i in zip(theta, norm):
        if theta_min <= t <= theta_max:
            closest = min(col_list, key=lambda x: abs(x - t))
            row[closest] = i
    return pd.DataFrame([row]).fillna(0), theta, norm

def plot_xrd(x, y, out_path):
    peaks, _ = find_peaks(y, height=0.05)
    peak_positions = [round(x[i], 2) for i in peaks]
    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(x, y, label="Normalized Intensity", color="royalblue", linewidth=1.5)
    plt.plot(x[peaks], y[peaks], "ro", markersize=4, label="Detected Peaks")
    plt.fill_between(x, 0, y, color="lightblue", alpha=0.3)
    plt.xlabel("2θ (Theta)")
    plt.ylabel("Normalized Intensity")
    plt.title("XRD Pattern")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return peak_positions

if __name__ == "__main__":
    app.run(debug=True)

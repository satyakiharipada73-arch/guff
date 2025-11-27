import requests
from pathlib import Path
from flask import Flask, request, jsonify

# === CONFIG ===
FILE_ID = "14sDrw5IHnQo91Kjz1NbCTTWeagzlISEc"  # from your Drive link
MODEL_PATH = Path("model.bin")
PORT = 8000
# ==============

def download_from_gdrive(file_id: str, dst: Path):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    # First request (may return a confirmation token for big files)
    response = session.get(URL, params={"id": file_id}, stream=True)
    token = None
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break

    # If there is a token, confirm the download
    if token:
        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)

    # Stream to disk
    with open(dst, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

def ensure_model():
    if not MODEL_PATH.exists():
        print("Downloading model from Google Drive...")
        download_from_gdrive(FILE_ID, MODEL_PATH)
        print("Download complete.")
    else:
        print("Model file already exists, skipping download.")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("input", "")

    # TODO: load and use your real model here with MODEL_PATH
    # e.g. output = model.generate(text)
    output = f"Echo: {text}"

    return jsonify({"output": output})

if __name__ == "__main__":
    ensure_model()
    app.run(host="0.0.0.0", port=PORT)

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

    print(f"[INFO] Starting download from Google Drive (file_id={file_id})")
    # First request (may return a confirmation token for big files)
    response = session.get(URL, params={"id": file_id}, stream=True)
    token = None
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break

    # If there is a token, confirm the download
    if token:
        print("[INFO] Got download warning token, confirming download...")
        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)

    # Stream to disk with simple progress info
    downloaded = 0
    with open(dst, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (10 * 1024 * 1024) < 32768:  # every ~10MB
                    print(f"[INFO] Downloaded ~{downloaded / (1024*1024):.1f} MB...")

    print("[INFO] Download finished and written to disk.")

def ensure_model():
    if not MODEL_PATH.exists():
        print("[INFO] Model file not found, starting download...")
        download_from_gdrive(FILE_ID, MODEL_PATH)
        print("[INFO] Model download complete.")
    else:
        print(f"[INFO] Model file already exists at {MODEL_PATH}, skipping download.")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    print("[INFO] /predict endpoint called")
    data = request.get_json(force=True)
    text = data.get("input", "")
    print(f"[INFO] Received input: {text!r}")

    # TODO: load and use your real model here with MODEL_PATH
    # e.g. output = model.generate(text)
    output = f"Echo: {text}"

    print(f"[INFO] Sending output: {output!r}")
    return jsonify({"output": output})

if __name__ == "__main__":
    print(f"[INFO] Starting server on port {PORT}")
    ensure_model()
    app.run(host="0.0.0.0", port=PORT)

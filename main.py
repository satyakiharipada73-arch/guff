import requests
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
from llama_cpp import Llama

# === CONFIG ===
FILE_ID = "14sDrw5IHnQo91Kjz1NbCTTWeagzlISEc"  # your Drive file id
MODEL_PATH = Path("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
PORT = 8000
# ==============

llm = None  # global model instance


def download_from_gdrive(file_id: str, dst: Path):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    print(f"[INFO] Starting download from Google Drive (file_id={file_id})")
    response = session.get(URL, params={"id": file_id}, stream=True)
    token = None
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break

    if token:
        print("[INFO] Got download warning token, confirming download...")
        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)

    downloaded = 0
    with open(dst, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (10 * 1024 * 1024) < 32768:
                    print(f"[INFO] Downloaded ~{downloaded / (1024*1024):.1f} MB...")

    print("[INFO] Download finished and written to disk.")


def ensure_model():
    if not MODEL_PATH.exists():
        print("[INFO] Model file not found, starting download...")
        download_from_gdrive(FILE_ID, MODEL_PATH)
        print("[INFO] Model download complete.")
    else:
        print(f"[INFO] Model file already exists at {MODEL_PATH}, skipping download.")


def load_llm():
    global llm
    if llm is None:
        print(f"[INFO] Loading GGUF model from {MODEL_PATH} ...")
        llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=4096,
            n_threads=4,
            chat_format="chatml",  # Qwen 2.5 uses ChatML-style formatting
            verbose=True,
        )
        print("[INFO] Model loaded.")
    return llm


app = Flask(__name__)

# --- Simple browser chat page (normal web window) ---

HTML_PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Qwen2.5 Coder Chat</title>
  <style>
    body { font-family: sans-serif; max-width: 800px; margin: 20px auto; }
    textarea { width: 100%; height: 120px; }
    pre { background: #f5f5f5; padding: 10px; white-space: pre-wrap; }
  </style>
</head>
<body>
  <h1>Qwen2.5 Coder Chat</h1>
  <form id="chat-form">
    <label>Message:</label><br>
    <textarea id="input" placeholder="Ask something..."></textarea><br><br>
    <button type="submit">Send</button>
  </form>
  <h2>Response</h2>
  <pre id="output"></pre>

  <script>
    const form = document.getElementById('chat-form');
    const inputEl = document.getElementById('input');
    const outputEl = document.getElementById('output');

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const text = inputEl.value;
      outputEl.textContent = "Loading...";
      try {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ input: text })
        });
        const data = await res.json();
        outputEl.textContent = data.output || JSON.stringify(data, null, 2);
      } catch (err) {
        outputEl.textContent = "Error: " + err;
      }
    });
  </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    # Browser UI
    return render_template_string(HTML_PAGE)


# --- API endpoint for Postman/clients ---

@app.route("/predict", methods=["POST"])
def predict():
    print("[INFO] /predict endpoint called")
    data = request.get_json(force=True)
    text = data.get("input", "")
    print(f"[INFO] Received input: {text!r}")

    llm = load_llm()

    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": text},
        ],
        max_tokens=256,
        temperature=0.7,
    )

    reply = resp["choices"][0]["message"]["content"]
    print(f"[INFO] Model reply: {reply!r}")

    return jsonify({"output": reply})


if __name__ == "__main__":
    print(f"[INFO] Starting server on port {PORT}")
    ensure_model()
    load_llm()
    app.run(host="0.0.0.0", port=PORT)

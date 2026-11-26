import os
import sys
import tempfile
import threading

from flask import Flask, request, send_file, render_template_string
from huggingface_hub import snapshot_download

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

MODEL_DIR = os.path.join(os.getcwd(), "mithi-v2-flash")

tts = None
model_lock = threading.Lock()
VOICES = ["satvik", "vaidehi"]

HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Mithi Hindi TTS</title>
    <style>
        body {
            font-family: Arial;
            background: #f1f3f6;
            padding: 20px;
        }
        .box {
            max-width: 600px;
            margin: auto;
            background: white;
            padding: 20px;
            border-radius: 12px;
        }
        textarea, select, button {
            width: 100%;
            padding: 13px;
            margin-top: 12px;
            box-sizing: border-box;
            font-size: 16px;
        }
        button {
            background: #1769ff;
            color: white;
            border: none;
            border-radius: 7px;
        }
    </style>
</head>
<body>
<div class="box">
    <h2>Mithi Hindi AI Voice Generator</h2>
    <form method="POST" action="/generate">
        <textarea name="text" rows="9"
        placeholder="अपनी कहानी यहाँ लिखें..." required></textarea>

<select name="voice">
    <option value="satvik">Male - Satvik</option>
    <option value="vaidehi">Female - Vaidehi</option>
</select>

        <button type="submit">Generate Audio</button>
    </form>
</div>
</body>
</html>
"""


def load_model():
    global tts

    if tts is not None:
        return tts

    with model_lock:
        if tts is not None:
            return tts

        print("Downloading Mithi model files...")

        snapshot_download(
            repo_id="Oomni/mithi-v2-flash",
            local_dir=MODEL_DIR
        )

        os.chdir(MODEL_DIR)
        sys.path.insert(0, MODEL_DIR)

        import torch
        torch.set_num_threads(1)

        from run import Mithi

        print("Loading Mithi model...")
        tts = Mithi(device="cpu")

        print("Mithi model loaded successfully!")

    return tts


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/generate", methods=["POST"])
def generate():
    text = request.form.get("text", "").strip()
    voice = request.form.get("voice", "satvik")

    if not text:
        return "Please enter some text.", 400

    if voice not in VOICES:
        return "Invalid voice selected.", 400

    output_path = None

    try:
        model = load_model()

        audio = model.synthesize(
            text,
            lang="hi",
            voice=voice
        )

        temp_file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        )
        output_path = temp_file.name
        temp_file.close()

        model.save(audio, output_path)

        response = send_file(
            output_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="mithi_audio.wav"
        )

        response.call_on_close(
            lambda: os.remove(output_path)
            if os.path.exists(output_path) else None
        )

        return response

    except Exception as e:
        print("ERROR:", repr(e))

        if output_path and os.path.exists(output_path):
            os.remove(output_path)

        return "Audio generation failed: " + str(e), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

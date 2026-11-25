import os
import uuid
import subprocess
import tempfile

from flask import Flask, request, send_file, render_template_string
from run import Mithi

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

print("Loading Mithi AI model...")

import torch
torch.set_num_threads(2)

tts = Mithi(device="cpu")

print("Mithi AI model loaded successfully!")

VOICES = {
    "satvik": "Satvik (Male)",
    "anirban": "Anirban (Male)",
    "nakul": "Nakul (Male)",
    "vasant": "Vasant (Male)",
    "siddhant": "Siddhant (Male)",
    "charvi": "Charvi (Female)",
    "ahana": "Ahana (Female)",
    "vaidehi": "Vaidehi (Female)",
    "malini": "Malini (Female)",
    "jahnavi": "Jahnavi (Female)"
}

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Mithi Hindi AI Voice Generator</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            background: #f1f3f6;
            font-family: Arial;
            padding: 20px;
        }

        .box {
            background: white;
            max-width: 600px;
            margin: auto;
            padding: 20px;
            border-radius: 12px;
        }

        textarea, select {
            width: 100%;
            padding: 12px;
            margin: 10px 0;
            box-sizing: border-box;
            font-size: 16px;
        }

        textarea {
            height: 250px;
        }

        button {
            background: #087b46;
            color: white;
            padding: 15px;
            border: none;
            width: 100%;
            font-size: 17px;
            border-radius: 8px;
        }

        h2 {
            text-align: center;
        }
    </style>
</head>

<body>
<div class="box">

<h2>🎙️ Mithi Hindi AI Voice Generator</h2>

<form action="/generate" method="POST">

<label>अपनी कहानी यहाँ Paste करें:</label>

<textarea name="text" maxlength="20000"
placeholder="अपनी कहानी यहाँ लिखें..." required></textarea>

<label>अपनी पसंद की आवाज चुनें:</label>

<select name="voice">
{% for key, name in voices.items() %}
<option value="{{ key }}">{{ name }}</option>
{% endfor %}
</select>

<button type="submit">Generate Audio</button>

</form>

</div>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML, voices=VOICES)


@app.route("/generate", methods=["POST"])
def generate():

    text = request.form.get("text", "").strip()
    voice = request.form.get("voice", "satvik")

    if not text:
        return "कृपया कहानी लिखें।", 400

    if len(text) > 20000:
        return "अधिकतम 20000 characters की अनुमति है।", 400

    if voice not in VOICES:
        return "Invalid voice selection.", 400

    with tempfile.TemporaryDirectory() as folder:

        wav_file = os.path.join(folder, "story.wav")
        mp3_file = os.path.join(folder, "story.mp3")

        try:
            audio = tts.synthesize(
                text,
                lang="hi",
                voice=voice
            )

            tts.save(audio, wav_file)

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i", wav_file,
                    "-codec:a", "libmp3lame",
                    "-b:a", "128k",
                    mp3_file
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )

            return send_file(
                mp3_file,
                as_attachment=True,
                download_name="Mithi_Hindi_Audio.mp3",
                mimetype="audio/mpeg"
            )

        except Exception as e:
            return f"Audio generation failed: {str(e)}", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

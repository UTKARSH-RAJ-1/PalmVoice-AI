# PalmVoice AI

PalmVoice AI is an experimental sign-language translation project with two complementary demos:

- **Sign to text:** recognize static ASL alphabet hand shapes from a webcam.
- **Text to sign:** display a typed message as hand poses in a browser avatar viewer.

The current implementation is an MVP focused on fingerspelling. It uses MediaPipe hand landmarks and the `smart_gestures` ASL model for webcam predictions, plus a small local landmark dataset for the browser demo and centroid-classifier training path.

## Repository layout

```text
PalmVoice-AI/
├── sign-translator/
│   ├── app.py                 # Flask web server and JSON endpoints
│   ├── data/                  # Landmark dataset used by the local demo
│   ├── models/                # Generated/local classifier artifact
│   ├── src/
│   │   ├── landmarks.py       # Landmark generation and CSV helpers
│   │   ├── live_demo.py       # Webcam ASL recognition
│   │   ├── recognize.py       # Model prediction helpers
│   │   ├── text_to_sign.py    # Landmark-to-pose export
│   │   ├── train_model.py     # Local centroid classifier
│   │   └── tts_stt.py         # Speech helper utilities
│   ├── web/
│   │   ├── index.html          # Browser text-to-sign UI
│   │   └── poses.json          # Browser-ready hand poses
│   ├── requirements.txt
│   └── README.md
├── Sign_Language_Translator_Build_Guide.pdf
└── README.md
```

## Requirements

- Windows, macOS, or Linux
- Python 3.11 (recommended)
- A webcam for the live recognition demo
- A working microphone/speaker setup if using the speech utilities

## Installation

From the repository root:

```powershell
cd sign-translator
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS/Linux, use `.venv/bin/python` instead of `.\.venv\Scripts\python.exe`.

## Run the browser demo

```powershell
cd sign-translator
.\.venv\Scripts\python.exe app.py
```

Open <http://localhost:5000> in a browser. The app exposes:

| Endpoint | Purpose |
| --- | --- |
| `/` | Text-to-sign browser interface |
| `/health` | Simple health check |
| `/api/poses` | Browser-ready pose data |
| `/api/demo` | Small end-to-end demo response |
| `/api/train` | Retrains the local centroid classifier |

The UI can load a local `.vrm` humanoid avatar. Avatar files are not uploaded to the server; only use assets you are licensed to use.

## Run live ASL recognition

```powershell
cd sign-translator
.\.venv\Scripts\python.exe src\live_demo.py
```

Press `q` in the camera window to stop. The live demo requires a camera and may need platform-specific permissions.

## Train and regenerate assets

```powershell
cd sign-translator
.\.venv\Scripts\python.exe src\train_model.py
.\.venv\Scripts\python.exe src\text_to_sign.py
```

The generated classifier is written to `models/gesture_model.pkl`; browser poses are written to `web/poses.json`.

## Known limitations

- This is static fingerspelling, not continuous sign-language understanding.
- The bundled dataset is intentionally small and is not a production training corpus.
- Accuracy depends on lighting, camera quality, hand position, and the installed model.
- The browser avatar viewer needs a compatible VRM file for full humanoid animation; it has a procedural fallback.
- Speech-to-text and text-to-speech helpers are provided as building blocks and may require system audio configuration.

## Development notes

Keep generated environments and caches out of commits. Before opening a pull request, run the application health check and the relevant scripts from the `sign-translator` directory. See [CONTRIBUTING.md](CONTRIBUTING.md) for the project workflow.

## License

This project is released under the [MIT License](LICENSE). Third-party models, datasets, and avatar assets retain their own licenses.


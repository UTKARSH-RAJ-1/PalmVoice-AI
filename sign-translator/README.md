# Sign Language Translator

This project follows the build guide and uses a real ASL hand-sign model from the internet via the `smart_gestures` PyPI package. The primary goal is a working fingerspelling MVP for sign-to-text recognition, plus a browser-based text-to-sign demo.

## Real dataset/model source

We used the internet-sourced ASL model package:

- `smart_gestures` on PyPI
- Includes a trained ASL alphabet model and class list for static hand landmarks
- Model is loaded through `smart_gestures.alphabet.asl_model.model.ASLModel`

This meets the project requirement to use data from the internet instead of relying only on a synthetic local mockup.

## Project structure

- `app.py` — Flask server and API routes
- `src/live_demo.py` — live webcam ASL recognition using MediaPipe + `smart_gestures`
- `src/recognize.py` — model-based sign classification utilities
- `src/landmarks.py` — synthetic fallback and pose export helpers
- `src/train_model.py` — lightweight local trainer for the generated landmark CSV
- `src/text_to_sign.py` — builds `web/poses.json`
- `src/tts_stt.py` — TTS wrapper built around `pyttsx3`
- `web/index.html` — browser demo that plays letter poses on a procedural fallback or uploaded VRM humanoid avatar
- `data/landmarks.csv` — generated dataset used by the local trainer and fallback path

## Setup

```bash
cd sign-translator
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the live ASL webcam demo

```bash
cd sign-translator
.\.venv\Scripts\python.exe src\live_demo.py
```

## Run the browser demo

```bash
cd sign-translator
.\.venv\Scripts\python.exe app.py
```

Then open http://localhost:5000.

## Notes

- The project is intentionally static-fingerspelling focused and matches the recommended MVP in the guide.
- `smart_gestures` supplies the real internet ASL model; a local synthetic fallback still exists for environments without a webcam or dataset.
- The browser supports `.vrm` humanoid avatars through Three.js and `three-vrm`. Click **Load 3D avatar** and choose a VRM file. The procedural character remains as a fallback when no avatar is loaded.
- You can also drag a `.vrm` file onto the viewer. The page validates the extension, loads the model locally with no upload to the server, and shows an explicit error if the file is invalid.
- Use an avatar that you are licensed to use. VRM files are recommended because their standardized humanoid bones allow the sign pose to drive the arm, hand, head, and finger rig.
- For a production-ready version, you would expand the dataset and add a richer alphabet vocabulary, better word assembly, and full-body motion capture.

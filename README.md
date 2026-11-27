# VoiceGuard AI — Your Voice, Your Security

A local Flask prototype for screening recordings for audio patterns associated with synthetic speech. It includes a responsive dashboard, local accounts, microphone capture, analysis history, JSON reports, charts, SQLite storage, and a PyTorch CNN training pipeline.

## Run the app

Python 3.10–3.12 is recommended (audio and ML wheels can lag the newest Python releases).

```powershell
cd voiceguard-ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 and create a local account. The first account is created locally; there is no email service. Browser microphone capture requires localhost or HTTPS and microphone permission.

## Classifier and dataset

This prototype includes a CNN trained on a 600-clip English subset of [MLAAD-tiny](https://huggingface.co/datasets/mueller91/MLAAD-tiny): 300 authentic and 300 synthetic clips from six TTS systems. Dataset attribution and license notices are in `dataset/licenses/`. MLAAD-tiny's synthetic clips are CC BY-NC 4.0, so this bundled sample is for non-commercial educational use; see all included notices before redistribution.

The trained checkpoint is `models/voiceguard_cnn.pt`; validation metrics are in `models/voiceguard_metrics.json`. To retrain or use a different labelled dataset, follow [TRAINING.md](TRAINING.md). The included small subset is for a classroom demonstration, not deployment or forensic use.

## Features

- WAV, MP3, OGG, FLAC, M4A and WebM upload (25 MB limit) and in-browser playback.
- Microphone recording and provisional short-segment capture.
- CNN inference after training; prediction confidence, risk category, timestamp and explanatory caveat.
- Private local analysis history with JSON report download and deletion.
- Registration, login and logout with password hashing.
- Separate user and administrator sign-in, with administrator credentials configured through environment variables.
- Dashboard counts, recent activity, model status and safety guidance.

## Administrator sign-in

For a hosted Render deployment, open the service's **Environment** settings and add `ADMIN_EMAIL` and `ADMIN_PASSWORD`. Use a strong, unique password and keep it private. Save the settings and redeploy. On the site, choose **Admin sign in**; administrators can view account totals and the latest analyses across user accounts. Admin accounts are not created through public registration.

## Limitations

This is a student prototype, not a forensic or identity-verification system. Scores can be wrong and model confidence is not a guarantee. It does not monitor calls or other applications. Audio is sent to the local Flask process for analysis and is not persisted as an audio file; only the result metadata is stored. Change `VOICEGUARD_SECRET` before exposing the app beyond your own machine.

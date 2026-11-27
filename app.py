"""VoiceGuard AI: local prototype for audio deepfake risk screening.

Predictions are only made after a classifier has been trained with labelled data.
Without trained weights the API returns UNCERTAIN rather than a fabricated result.
"""
from __future__ import annotations

import io
import json
import os
import secrets
import sqlite3
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import numpy as np
from flask import Flask, g, jsonify, redirect, render_template, request, send_file, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "voiceguard.sqlite3"
MODEL_PATH = ROOT / "models" / "voiceguard_cnn.pt"
SECRET_PATH = ROOT / "data" / "session_secret.key"
ALLOWED = {"wav", "mp3", "ogg", "flac", "m4a", "webm"}


def session_secret():
    configured=os.environ.get("VOICEGUARD_SECRET")
    if configured: return configured
    SECRET_PATH.parent.mkdir(parents=True,exist_ok=True)
    if not SECRET_PATH.exists(): SECRET_PATH.write_text(secrets.token_hex(32),encoding="utf-8")
    return SECRET_PATH.read_text(encoding="utf-8").strip()


app = Flask(__name__)
app.secret_key = session_secret()
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


def db():
    if "db" not in g:
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_=None):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def init_db():
    # The deploy filesystem may not contain an empty data/ directory from the repo.
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
          CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, password TEXT NOT NULL, created_at TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS analyses(id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, filename TEXT NOT NULL, created_at TEXT NOT NULL, label TEXT NOT NULL, confidence REAL, risk TEXT NOT NULL, details TEXT NOT NULL, FOREIGN KEY(user_id) REFERENCES users(id));
        """)


def login_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Please sign in first."}), 401
        return fn(*args, **kwargs)
    return wrapped


def classify_audio(raw: bytes, filename: str):
    """Run the trained CNN when available; otherwise safely return uncertain.

    Audio is decoded/resampled and converted to log-mel features by the same
    utility used in train.py. The model is trained only from the user's labelled
    real/fake folders; no synthetic predictions are generated here.
    """
    if not MODEL_PATH.exists():
        return {"label": "UNCERTAIN", "confidence": None, "risk": "UNRATED",
                "reason": "No trained model is installed. Follow TRAINING.md with labelled audio, then restart the app."}
    try:
        from ml import predict_bytes
        label, confidence, duration = predict_bytes(raw, filename, MODEL_PATH)
        if confidence < 0.65:
            label = "UNCERTAIN"
        risk = "HIGH" if label == "AI-GENERATED VOICE" and confidence >= .8 else "ELEVATED" if label == "AI-GENERATED VOICE" else "LOW" if label == "REAL VOICE" else "UNRATED"
        return {"label": label, "confidence": round(float(confidence), 4), "risk": risk,
                "reason": "Model estimate from audio features; not proof of authenticity.", "duration": round(duration, 2)}
    except Exception as exc:
        app.logger.exception("Audio classification failed")
        return {"label": "UNCERTAIN", "confidence": None, "risk": "UNRATED", "reason": f"Audio could not be analysed: {str(exc)[:180]}"}


@app.get("/")
def home():
    if session.get("user_id"):
        return render_template("index.html", page="dashboard", email=session.get("email"))
    return render_template("index.html", page="welcome", email="")


@app.post("/api/register")
def register():
    data = request.get_json(silent=True) or {}
    email, password = str(data.get("email", "")).strip().lower(), str(data.get("password", ""))
    if "@" not in email or len(password) < 8:
        return jsonify(error="Enter a valid email and a password of at least 8 characters."), 400
    try:
        cur = db().execute("INSERT INTO users(email,password,created_at) VALUES(?,?,?)", (email, generate_password_hash(password), datetime.now(timezone.utc).isoformat()))
        db().commit()
    except sqlite3.IntegrityError:
        return jsonify(error="An account with that email already exists."), 409
    session.update(user_id=cur.lastrowid, email=email)
    return jsonify(ok=True)


@app.post("/api/login")
def login():
    data = request.get_json(silent=True) or {}
    row = db().execute("SELECT * FROM users WHERE email=?", (str(data.get("email", "")).strip().lower(),)).fetchone()
    if not row or not check_password_hash(row["password"], str(data.get("password", ""))):
        return jsonify(error="Email or password is incorrect."), 401
    session.update(user_id=row["id"], email=row["email"])
    return jsonify(ok=True)


@app.post("/api/logout")
def logout():
    session.clear()
    return jsonify(ok=True)


@app.post("/api/analyze")
@login_required
def analyze():
    f = request.files.get("audio")
    if not f or not f.filename or "." not in f.filename or f.filename.rsplit(".", 1)[1].lower() not in ALLOWED:
        return jsonify(error="Choose WAV, MP3, OGG, FLAC, M4A, or WebM audio."), 400
    raw = f.read()
    if len(raw) < 256:
        return jsonify(error="The audio file is empty or too small."), 400
    result = classify_audio(raw, f.filename)
    stamp = datetime.now(timezone.utc).isoformat()
    safe_name = secure_filename(f.filename)[:180] or "recording"
    cur = db().execute("INSERT INTO analyses(user_id,filename,created_at,label,confidence,risk,details) VALUES(?,?,?,?,?,?,?)",
        (session["user_id"], safe_name, stamp, result["label"], result["confidence"], result["risk"], json.dumps(result)))
    db().commit()
    result.update(id=cur.lastrowid, filename=safe_name, created_at=stamp)
    return jsonify(result)


@app.get("/api/history")
@login_required
def history():
    rows = db().execute("SELECT * FROM analyses WHERE user_id=? ORDER BY id DESC LIMIT 100", (session["user_id"],)).fetchall()
    return jsonify([dict(r) for r in rows])


@app.delete("/api/history/<int:analysis_id>")
@login_required
def delete_analysis(analysis_id):
    db().execute("DELETE FROM analyses WHERE id=? AND user_id=?", (analysis_id, session["user_id"]))
    db().commit()
    return jsonify(ok=True)


@app.get("/api/report/<int:analysis_id>")
@login_required
def report(analysis_id):
    row = db().execute("SELECT * FROM analyses WHERE id=? AND user_id=?", (analysis_id, session["user_id"])).fetchone()
    if not row:
        return jsonify(error="Analysis not found."), 404
    payload = dict(row)
    payload["details"] = json.loads(payload["details"])
    return jsonify(payload)


@app.get("/api/stats")
@login_required
def stats():
    rows = db().execute("SELECT label,risk,COUNT(*) n FROM analyses WHERE user_id=? GROUP BY label,risk", (session["user_id"],)).fetchall()
    counts = {"total": 0, "real": 0, "fake": 0, "uncertain": 0}
    for r in rows:
        counts["total"] += r["n"]
        counts[{"REAL VOICE":"real", "AI-GENERATED VOICE":"fake"}.get(r["label"], "uncertain")] += r["n"]
    counts["model_ready"] = MODEL_PATH.exists()
    metrics_path = MODEL_PATH.with_name("voiceguard_metrics.json")
    if metrics_path.exists():
        try:
            counts["validation_accuracy"] = json.loads(metrics_path.read_text(encoding="utf-8")).get("validation_accuracy")
        except (OSError, ValueError):
            counts["validation_accuracy"] = None
    return jsonify(counts)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Audio must be 25 MB or smaller."), 413


@app.errorhandler(Exception)
def unexpected_error(exc):
    # Keep API failures machine-readable so the browser can show a useful message.
    if isinstance(exc, HTTPException):
        return exc
    app.logger.exception("Unhandled request error", exc_info=exc)
    if request.path.startswith("/api/"):
        return jsonify(error="The server could not complete this request. Check the service logs."), 500
    return "The server encountered an error. Please try again.", 500


# Gunicorn imports the Flask application instead of running this file as a script.
init_db()


if __name__ == "__main__":
    # Bind all interfaces so the desktop's browser bridge can reach this process.
    # Keep this local prototype off untrusted networks unless access is intended.
    app.run(host=os.environ.get("VOICEGUARD_HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "5000")), debug=False)

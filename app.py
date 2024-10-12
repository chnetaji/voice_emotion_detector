import sqlite3
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import librosa
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = "your_secret_key"

model = load_model("LSTM_model.keras")
emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad"]


def init_db():
    try:
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL
            )
        """
        )

        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()


init_db()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/input")
def input():
    return render_template("input.html")


@app.route("/sign_in")
def sign_in():
    return render_template("sign_in.html")


@app.route("/sign_up")
def sign_up():
    return render_template("sign_up.html")


# Sign Up Route
@app.route("/sign_up", methods=["POST"])
def register_user():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    existing_user = cursor.fetchone()

    if existing_user:
        conn.close()
        return jsonify({"status": "error", "message": "Email already registered"}), 409

    cursor.execute(
        "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
        (name, email, password),
    )

    conn.commit()
    conn.close()
    return jsonify({"status": "success", "message": "Account created successfully"})


# Sign In Route
@app.route("/sign_in", methods=["POST"])
def authenticate_user():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE email = ? AND password = ?", (email, password)
    )
    user = cursor.fetchone()
    conn.close()

    if user:
        session["user_id"] = user[0]
        return jsonify({"status": "success", "message": "Sign in successful"}), 200
    else:
        return jsonify({"status": "error", "message": "Invalid email or password"}), 401


@app.route("/log_out")
def log_out():
    session.pop("user_id", None)
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "audio" not in request.files:
        return "No file part"

    file = request.files["audio"]

    if file.filename == "":
        return "No selected file"

    audio_data, sr = librosa.load(file, sr=None)

    emotions = process_audio_for_prediction(audio_data, sr)
    total = len(emotions)
    predictions = {
        emotion: f"{emotions.count(emotion) / total * 100:.2f} %"
        for emotion in emotions
    }

    return render_template("results.html", predictions=predictions)


def predict(mfcc):
    predictions = model.predict(mfcc)
    predicted_label_index = np.argmax(predictions, axis=1)[0]
    return emotion_labels[predicted_label_index]


def process_audio_for_prediction(audio_data, sr):
    duration = librosa.get_duration(y=audio_data, sr=sr)
    predictions = []

    for start in range(0, int(duration), 3):
        end = min(start + 3, duration)
        chunk = audio_data[int(start * sr) : int(end * sr)]
        mfcc = np.mean(librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=40).T, axis=0)
        mfcc = mfcc.reshape(1, 40, 1)
        pred = predict(mfcc)
        predictions.append(pred)

    return predictions


if __name__ == "__main__":
    init_db()
    app.run(debug=True)

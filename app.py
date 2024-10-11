from flask import Flask, request, render_template
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from collections import Counter


app = Flask(__name__)

# Load your trained Keras model
model = load_model("LSTM_model.keras")  # Update the path to your model file
emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad"]


# Function to simulate predictions (use your actual model in place of predict)
def predict(mfcc):
    # Make prediction and get the index of the highest probability
    predictions = model.predict(mfcc)
    predicted_label_index = np.argmax(predictions, axis=1)[0]
    return emotion_labels[predicted_label_index]


# Function to process audio file and make predictions
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

    counts = Counter(predictions)
    t = len(counts)

    return predictions


# Route for home page
@app.route("/")
def home():
    return render_template("app.html")


# Route to handle file upload and process the audio file
@app.route("/upload", methods=["POST"])
def upload_file():
    if "audio" not in request.files:
        return "No file part"

    file = request.files["audio"]

    if file.filename == "":
        return "No selected file"

    # Load the audio file directly from the file stream
    audio_data, sr = librosa.load(file, sr=None)

    # Process the audio data
    predictions = process_audio_for_prediction(audio_data, sr)

    return render_template("results.html", predictions=predictions)


if __name__ == "__main__":
    app.run()

import librosa
import numpy as np
import io
import speech_recognition as sr

def evaluate(audio_file):
    # Convert uploaded Streamlit file to BytesIO for librosa
    audio_bytes = audio_file.read()
    audio_buffer = io.BytesIO(audio_bytes)

    # Load audio
    y, sr_audio = librosa.load(audio_buffer, sr=None)

    # Extract audio features
    rms = np.mean(librosa.feature.rms(y=y))                    # Clarity (energy)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))       # Clarity (voicing)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr_audio))  # Sharpness

    # Calculate Clarity Score (scaled to 0–10)
    clarity = np.clip((rms + zcr + spectral_centroid / 1000) * 10, 0, 10)

    # Confidence — measure variation in pitch and loudness
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr_audio)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0
    confidence = 10 - min(pitch_std, 10)  # lower std = more confident

    # Accuracy — based on speech recognizer ability to transcribe
    recognizer = sr.Recognizer()
    audio_buffer.seek(0)  # rewind before using again
    with sr.AudioFile(audio_buffer) as source:
        audio_data = recognizer.record(source)
        try:
            transcript = recognizer.recognize_google(audio_data)
            accuracy = 10.0
        except sr.UnknownValueError:
            transcript = "[Could not understand speech]"
            accuracy = 4.0
        except sr.RequestError:
            transcript = "[API unavailable]"
            accuracy = 0.0

    # Final score as average
    final_score = round((clarity + confidence + accuracy) / 3, 2)

    return transcript, round(clarity, 2), round(confidence, 2), round(final_score, 2)

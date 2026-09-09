# inference.py - Detection, prediction, fusion
import numpy as np
from calibration import extract_vocal_features, normalize_features, stress_score_from_zscores
import cv2
import torch
import librosa
import numpy as np
import sounddevice as sd
import queue
from torchvision import transforms

FACE_EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
AUDIO_EMOTIONS = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
SAMPLE_RATE = 22050
AUDIO_DURATION = 3

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def detect_face_emotion(frame, face_model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None, None, frame
    x, y, bw, bh = faces[0]
    face = frame[y:y+bh, x:x+bw]
    if face.size == 0:
        return None, None, frame
    tensor = face_transform(face).unsqueeze(0)
    with torch.no_grad():
        output = face_model(tensor)
        probs = torch.softmax(output, dim=1)
        idx = probs.argmax().item()
        confidence = probs.max().item()
    emotion = FACE_EMOTIONS[idx]
    cv2.rectangle(frame, (x, y), (x+bw, y+bh), (30, 30, 30), 2)
    cv2.putText(frame, f"{emotion} {confidence:.0%}",
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    return emotion, confidence, frame

audio_result_queue = queue.Queue()
vocal_stress_queue = queue.Queue()

def record_and_predict(audio_model, vocal_baseline=None):
    while True:
        audio = sd.rec(int(AUDIO_DURATION * SAMPLE_RATE),
                      samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)

        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
        max_len = 130
        if mfcc.shape[1] < max_len:
            mfcc = np.pad(mfcc, ((0,0),(0, max_len - mfcc.shape[1])))
        else:
            mfcc = mfcc[:, :max_len]
        tensor = torch.FloatTensor(mfcc).unsqueeze(0)
        with torch.no_grad():
            output = audio_model(tensor)
            idx = output.argmax().item()
        audio_result_queue.put(AUDIO_EMOTIONS[idx])

        if vocal_baseline is not None:
            raw = extract_vocal_features(audio, sr=SAMPLE_RATE)
            z = normalize_features(raw, vocal_baseline)
            vs = stress_score_from_zscores(z)
            if not (np.isnan(vs) or np.isinf(vs)):
                vocal_stress_queue.put(vs)

EMOTION_MAP = {
    'angry': 'Angry', 'disgust': 'Disgust', 'fear': 'Fearful',
    'happy': 'Happy', 'neutral': 'Neutral', 'sad': 'Sad', 'surprise': 'Surprised',
}

def fuse_emotions(face_emotion, audio_emotion):
    if face_emotion is None and audio_emotion is None:
        return "Awaiting signal", False
    if face_emotion is None:
        return audio_emotion, False
    if audio_emotion is None:
        return face_emotion, False
    face_mapped = EMOTION_MAP.get(face_emotion.lower(), face_emotion)
    if face_mapped.lower() == audio_emotion.lower():
        return face_mapped, True
    return f"{face_emotion} / {audio_emotion}", False
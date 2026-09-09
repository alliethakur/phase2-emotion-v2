# calibration.py - Per-person baseline calibration + adaptive normalization
import numpy as np
import mediapipe as mp
import librosa
import sounddevice as sd
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]
LEFT_BROW = [70, 63, 105, 66, 107]
RIGHT_BROW = [336, 296, 334, 293, 300]

def extract_facial_features(image_rgb):
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = image_rgb.shape
    pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks])

    def eye_aspect_ratio(eye_idx):
        p = pts[eye_idx]
        vertical1 = np.linalg.norm(p[1] - p[5])
        vertical2 = np.linalg.norm(p[2] - p[4])
        horizontal = np.linalg.norm(p[0] - p[3])
        return (vertical1 + vertical2) / (2.0 * horizontal + 1e-6)

    def mouth_openness():
        p = pts[MOUTH]
        vertical = np.linalg.norm(p[4] - p[5])
        horizontal = np.linalg.norm(p[0] - p[1])
        return vertical / (horizontal + 1e-6)

    def brow_position():
        left_brow_y = pts[LEFT_BROW][:, 1].mean()
        left_eye_y = pts[LEFT_EYE][:, 1].mean()
        right_brow_y = pts[RIGHT_BROW][:, 1].mean()
        right_eye_y = pts[RIGHT_EYE][:, 1].mean()
        return ((left_eye_y - left_brow_y) + (right_eye_y - right_brow_y)) / 2

    return {
        "left_ear": eye_aspect_ratio(LEFT_EYE),
        "right_ear": eye_aspect_ratio(RIGHT_EYE),
        "mouth_openness": mouth_openness(),
        "brow_position": brow_position(),
    }

def capture_baseline(neutral_frames):
    all_features = []
    for frame in neutral_frames:
        feats = extract_facial_features(frame)
        if feats is not None:
            all_features.append(feats)
    if not all_features:
        return None
    keys = all_features[0].keys()
    baseline = {}
    for k in keys:
        values = [f[k] for f in all_features]
        baseline[k] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return baseline

def extract_vocal_features(audio, sr=22050):
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
    )
    f0_clean = f0[~np.isnan(f0)]
    pitch_mean = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0.0
    pitch_std = float(np.std(f0_clean)) if len(f0_clean) > 0 else 0.0
    rms = librosa.feature.rms(y=audio)[0]
    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))
    return {
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
    }

def capture_vocal_baseline(neutral_audio_clips, sr=22050, window_sec=1.5, hop_sec=1.0):
    all_features = []
    for clip in neutral_audio_clips:
        window_samples = int(window_sec * sr)
        hop_samples = int(hop_sec * sr)
        if len(clip) < window_samples:
            all_features.append(extract_vocal_features(clip, sr=sr))
            continue
        for start in range(0, len(clip) - window_samples, hop_samples):
            window = clip[start:start + window_samples]
            all_features.append(extract_vocal_features(window, sr=sr))
    if not all_features:
        return None
    keys = all_features[0].keys()
    baseline = {}
    for k in keys:
        values = [f[k] for f in all_features]
        baseline[k] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return baseline

def record_until_silence(sample_rate=22050, silence_threshold=0.002,
                          silence_duration=2.0, max_duration=15, min_speech_duration=1.0):
    recorded_chunks = []
    speech_started = False
    silence_start = None
    speech_start_time = None
    start_time = time.time()

    def callback(indata, frames, time_info, status):
        recorded_chunks.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', callback=callback):
        while True:
            time.sleep(0.1)
            if not recorded_chunks:
                continue
            recent = recorded_chunks[-1]
            energy = float(np.sqrt(np.mean(recent ** 2)))
            if energy > silence_threshold:
                if not speech_started:
                    speech_started = True
                    speech_start_time = time.time()
                silence_start = None
            elif speech_started:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > silence_duration:
                    if time.time() - speech_start_time > min_speech_duration:
                        break
            if time.time() - start_time > max_duration:
                break

    if not recorded_chunks:
        return np.array([], dtype='float32')
    return np.concatenate(recorded_chunks, axis=0).flatten()

def normalize_features(raw_features, baseline, eps=1e-6, min_std_ratio=0.05):
    if raw_features is None or baseline is None:
        return None
    normalized = {}
    for key, value in raw_features.items():
        if key not in baseline:
            continue
        mean = baseline[key]["mean"]
        std = baseline[key]["std"]
        safe_std = max(std, abs(mean) * min_std_ratio, eps)
        normalized[key] = float((value - mean) / safe_std)
    return normalized

def stress_score_from_zscores(z_dict):
    if not z_dict:
        return 0.0
    return float(np.mean([abs(v) for v in z_dict.values()]))
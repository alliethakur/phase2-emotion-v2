import streamlit as st
import cv2
import threading
import queue
import time

from models import load_models
from styles import inject_css
from inference import detect_face_emotion, record_and_predict, fuse_emotions, audio_result_queue, vocal_stress_queue
from calibration import capture_baseline, capture_vocal_baseline, record_until_silence
import numpy as np
from datetime import datetime
from storage import init_db, log_session_start, log_reading
from calibration import capture_baseline, capture_vocal_baseline, record_until_silence, extract_facial_features, normalize_features, stress_score_from_zscores


NEUTRAL_SCRIPT = "The quick brown fox jumps over the lazy dog while everything around it stays calm and ordinary."

def run_calibration(cap, frame_placeholder, face_seconds=5):
    calib_placeholder = st.empty()
    calib_placeholder.info(f"Calibrating... please stay neutral and quiet for {face_seconds} seconds.")
    neutral_frames = []
    start = time.time()
    while time.time() - start < face_seconds:
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            neutral_frames.append(frame_rgb)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        time.sleep(0.1)
    face_baseline = capture_baseline(neutral_frames)

    calib_placeholder.markdown(f"""
    <div class="card"><div class="card-label">Voice Calibration</div>
    <div class="emotion-sub">Please read this sentence aloud:</div>
    <div class="emotion-value" style="font-size:1.1rem;">{NEUTRAL_SCRIPT}</div></div>
    """, unsafe_allow_html=True)

    neutral_audio = record_until_silence()
    neutral_audio = np.nan_to_num(neutral_audio, nan=0.0, posinf=0.0, neginf=0.0)

    calib_placeholder.markdown("""
    <div class="card"><div class="card-label">Voice Calibration</div>
    <div class="emotion-value">Processing...</div></div>
    """, unsafe_allow_html=True)

    vocal_baseline = capture_vocal_baseline([neutral_audio])

    calib_placeholder.empty()
    return face_baseline, vocal_baseline

def interpret_stress(face_stress, vocal_stress):
    def phrase(value, modality):
        if value < 0.8:
            return f"{modality} looks like their normal calm state"
        elif value < 1.8:
            return f"{modality} is a little different from usual"
        else:
            return f"{modality} is notably different from their normal"

    face_phrase = phrase(face_stress, "Face")
    vocal_phrase = phrase(vocal_stress, "Voice") if vocal_stress > 0 else "Voice hasn't been heard yet"
    return f"{face_phrase}. {vocal_phrase}."

def stress_level_label(rolling_stress):
    if rolling_stress < 0.8:
        return "Calm", "stress-low"
    elif rolling_stress < 1.8:
        return "Mild deviation", "stress-mid"
    elif rolling_stress < 3.0:
        return "Elevated", "stress-mid"
    else:
        return "High", "stress-high"

def main():
    st.set_page_config(page_title="De-Escalation Model", layout="wide", initial_sidebar_state="collapsed")
    inject_css()

    init_db()

    if "session_id" not in st.session_state:
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    st.markdown("""
    <div class="app-header">
        <span class="app-title">De-Escalation Model</span>
        <span class="app-subtitle">Real-time multimodal emotion recognition</span>
    </div>
    """, unsafe_allow_html=True)

    face_model, audio_model = load_models()

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="card-label">Live Feed</div>', unsafe_allow_html=True)
        frame_placeholder = st.empty()

    with col2:
        st.markdown('<div class="card-label" style="margin-bottom:0.75rem">Face</div>', unsafe_allow_html=True)
        face_emotion_placeholder = st.empty()

        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

        st.markdown('<div class="card-label" style="margin-bottom:0.75rem">Voice</div>', unsafe_allow_html=True)
        audio_emotion_placeholder = st.empty()

        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

        st.markdown('<div class="card-label" style="margin-bottom:0.75rem">Fused Output</div>', unsafe_allow_html=True)
        fused_placeholder = st.empty()

        st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
        if "running" not in st.session_state:
            st.session_state.running = True

        if st.session_state.running:
            if st.button("Stop session"):
                st.session_state.running = False
                st.rerun()
        else:
            if st.button("Start session"):
                st.session_state.running = True
                if "face_baseline" in st.session_state:
                    del st.session_state["face_baseline"]
                if "vocal_baseline" in st.session_state:
                    del st.session_state["vocal_baseline"]
                st.rerun()


    cap = cv2.VideoCapture(0)

    if "face_baseline" not in st.session_state:
        face_baseline, vocal_baseline = run_calibration(cap, frame_placeholder)
        st.session_state.face_baseline = face_baseline
        st.session_state.vocal_baseline = vocal_baseline
        log_session_start(st.session_state.session_id, face_baseline, vocal_baseline)

    if "audio_thread_started" not in st.session_state:
        audio_thread = threading.Thread(target=record_and_predict, args=(audio_model, st.session_state.vocal_baseline), daemon=True)
        audio_thread.start()
        st.session_state.audio_thread_started = True

    current_audio_emotion = None

    current_face_confidence = 0.0
    prev_time = time.time()

    frame_counter = 0
    last_face_stress = 0.0

    current_vocal_stress = 0.0

    from collections import deque
    stress_window = deque(maxlen=30)

    chart_counter = 0

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Cannot access webcam.")
            break

        result = detect_face_emotion(frame, face_model)
        face_emotion, face_confidence, annotated_frame = result

        frame_counter += 1
        if frame_counter % 5 == 0:
            frame_rgb_for_stress = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw = extract_facial_features(frame_rgb_for_stress)
            z = normalize_features(raw, st.session_state.face_baseline)
            fs = stress_score_from_zscores(z)
            if not (np.isnan(fs) or np.isinf(fs)):
                last_face_stress = fs
        face_stress = last_face_stress

        try:
            current_audio_emotion = audio_result_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            v = vocal_stress_queue.get_nowait()
            if not (np.isnan(v) or np.isinf(v)):
                current_vocal_stress = v
        except queue.Empty:
            pass

        fused, confirmed = fuse_emotions(face_emotion, current_audio_emotion)

        combined_stress = (face_stress + current_vocal_stress) / 2
        if np.isnan(combined_stress) or np.isinf(combined_stress):
            combined_stress = 0.0
        log_reading(
            st.session_state.session_id,
            face_stress, current_vocal_stress, combined_stress,
            face_emotion, current_audio_emotion
        )

        stress_window.append(combined_stress)
        rolling_stress = float(np.mean(stress_window))

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        print(f"FPS: {fps:.1f}")

        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        face_conf_pct = int((face_confidence or 0) * 100)
        face_emotion_placeholder.markdown(f"""
        <div class="card">
            <div class="emotion-value">{face_emotion or "—"}</div>
            <div class="emotion-sub">{"Detected" if face_emotion else "No face detected"}</div>
            <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{face_conf_pct}%"></div></div>
        </div>
        """, unsafe_allow_html=True)

        audio_emotion_placeholder.markdown(f"""
        <div class="card">
            <div class="emotion-value">{current_audio_emotion or "—"}</div>
            <div class="status-row">
                <div class="status-dot {'inactive' if not current_audio_emotion else ''}"></div>
                <span class="status-text">{"Analyzing 3s windows" if not current_audio_emotion else "Active"}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        confirmed_badge = '<div class="fused-confirmed">Both modalities agree</div>' if confirmed else ''
        fused_placeholder.markdown(f"""
        <div class="fused-card">
            <div class="fused-value">{fused}</div>
            {confirmed_badge}
        </div>
        """, unsafe_allow_html=True)

        time.sleep(0.03)

    cap.release()

if __name__ == "__main__":
    main()
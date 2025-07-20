import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
from datetime import datetime
from collections import deque

st.set_page_config(layout="wide")
st.title("🎯 Mock Interview Distraction Tracker")

# Session state
if "running" not in st.session_state:
    st.session_state.running = False
if "report_path" not in st.session_state:
    st.session_state.report_path = ""
if "frame_placeholder" not in st.session_state:
    st.session_state.frame_placeholder = st.empty()
if "distraction_start_time" not in st.session_state:
    st.session_state.distraction_start_time = None
if "distraction_label" not in st.session_state:
    st.session_state.distraction_label = None

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_points, image_w, image_h):
    p = [(int(landmarks[i].x * image_w), int(landmarks[i].y * image_h)) for i in eye_points]
    A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    return (A + B) / (2.0 * C)

def compute_score(gaze, head, blink):
    return round((0.4 * gaze + 0.4 * head + 0.2 * (0 if blink else 1)) * 100, 2)

def classify_gaze_direction(landmarks):
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    nose_tip = landmarks[1]

    eye_center_x = (left_eye.x + right_eye.x) / 2
    eye_center_y = (left_eye.y + right_eye.y) / 2

    dx = nose_tip.x - eye_center_x
    dy = nose_tip.y - eye_center_y

    if dx > 0.04:
        return "Right"
    elif dx < -0.04:
        return "Left"
    elif dy > 0.04:
        return "Down"
    elif dy<-0.04:
        return "Up"
    else:
        return "Center"

def run_tracking():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    score_history = deque(maxlen=10)

    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"reports/report_{timestamp}.csv"
    st.session_state.report_path = report_path
    csv_file = open(report_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["Timestamp", "Event", "Duration(s)"])

    cap = cv2.VideoCapture(0)

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        current_event = "Focused"
        direction = None

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            left = eye_aspect_ratio(lm, LEFT_EYE, w, h)
            right = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            blink = (left + right) / 2 < 0.2

            nose = lm[1]
            head = 1.0 if np.linalg.norm([nose.x * w - w / 2, nose.y * h - h / 2]) < 0.3 * w else 0.0

            iris_x = (lm[468].x + lm[473].x) / 2
            gaze = 1.0 if 0.5 < iris_x < 0.7 else 0.0

            score = compute_score(gaze, head, blink)
            score_history.append(score)
            smooth_score = int(np.mean(score_history))

            direction = classify_gaze_direction(lm)

            if direction != "Center":
                current_event = f"Distraction ({direction})"
            else:
                current_event = "Focused"

            color = (0, 255, 0) if smooth_score >= 80 else (0, 0, 255)
            cv2.putText(frame, f"Score: {smooth_score}%", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if current_event.startswith("Distraction"):
                if st.session_state.distraction_start_time is None:
                    st.session_state.distraction_start_time = time.time()
                    st.session_state.distraction_label = current_event
            else:
                if st.session_state.distraction_start_time is not None:
                    duration = round(time.time() - st.session_state.distraction_start_time, 2)
                    writer.writerow([now, st.session_state.distraction_label, duration])
                    writer.writerow([now, "Focused", 0])
                    csv_file.flush()
                    st.session_state.distraction_start_time = None
                    st.session_state.distraction_label = None

        st.session_state.frame_placeholder.image(frame, channels="BGR")

    # Final flush
    if st.session_state.distraction_start_time is not None:
        duration = round(time.time() - st.session_state.distraction_start_time, 2)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         st.session_state.distraction_label, duration])
        csv_file.flush()
        st.session_state.distraction_start_time = None
        st.session_state.distraction_label = None

    cap.release()
    csv_file.close()
    face_mesh.close()
    st.success("🔴 Tracking stopped and report saved.")

# UI
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("▶️ Start"):
        st.session_state.running = True
        run_tracking()

with col2:
    if st.button("⏹️ Stop"):
        st.session_state.running = False

with col3:
    if st.session_state.report_path and os.path.exists(st.session_state.report_path):
        with open(st.session_state.report_path, 'rb') as f:
            st.download_button("📥 Download Report", f, file_name=os.path.basename(st.session_state.report_path))

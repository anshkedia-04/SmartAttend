import streamlit as st
st.set_page_config(layout="centered")

import os
import pickle
import numpy as np
import pandas as pd
import cv2
import torch
from datetime import datetime
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import time


# =========================
# Constants
# =========================
DATASET_DIR = 'dataset'
STUDENTS_FILE = 'students.csv'
ATTENDANCE_DIR = 'attendance'
PASSWORD = "teacher123"
EMBEDDINGS_FILE = "student_embeddings.pkl"
NUM_SLOTS = 6
SLOTS = [f"Slot {i+1}" for i in range(NUM_SLOTS)]

# =========================
# Ensure directories
# =========================
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

if not os.path.exists(STUDENTS_FILE):
    pd.DataFrame(columns=["Roll", "Name"]).to_csv(STUDENTS_FILE, index=False)

# =========================
# Device configuration
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# Load FaceNet Models (cached)
# =========================
@st.cache_resource
def load_facenet_models():
    mtcnn_model = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
    resnet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return mtcnn_model, resnet_model

mtcnn, resnet = load_facenet_models()

# =========================
# Load student embeddings from pickle (cached)
# =========================
@st.cache_resource
def load_student_embeddings():
    if not os.path.exists(EMBEDDINGS_FILE):
        st.error("❌ Embeddings file not found. Please run the embedding generator script first.")
        st.stop()
    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)
    return data["student_embeddings"]

student_embeddings = load_student_embeddings()

# =========================
# Attendance utilities
# =========================
def get_attendance_file():
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")

def load_attendance():
    file = get_attendance_file()
    if not os.path.exists(file):
        df = pd.read_csv(STUDENTS_FILE)
        for slot in SLOTS:
            df[slot] = "Absent"
        df.to_csv(file, index=False)
    return pd.read_csv(file)

def save_attendance(df):
    df.to_csv(get_attendance_file(), index=False)

# =========================
# Recognition & Attendance marking
# =========================
def recognize_and_mark_attendance(slot):
    df = load_attendance()
    marked = []

    cap = cv2.VideoCapture(0)
    st.warning("Press 'Q' in webcam window to stop recognition.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face = mtcnn(img)
        if face is not None:
            with torch.no_grad():
                emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy()[0]
            scores = {name: cosine_similarity([emb], [vec])[0][0] for name, vec in student_embeddings.items()}
            identified = max(scores, key=scores.get)
            score = scores[identified]

            x, y, w, h = 50, 50, 200, 200  # Dummy bounding box

            if score >= 0.8:
                try:
                    roll = identified.split('_')[0]
                    name = '_'.join(identified.split('_')[1:])
                    student_row = df[df["Roll"].astype(str) == str(roll)]
                except Exception as e:
                    st.error(f"Parsing error: {e}")
                    continue

                if not student_row.empty:
                    already_present = df.at[student_row.index[0], slot] == 'Present'
                    if not already_present:
                        df.at[student_row.index[0], slot] = 'Present'
                        marked_time = datetime.now().strftime("%H:%M:%S")
                        marked.append((roll, name))
                        st.success(f"✅ Marked: {roll} - {name} at {marked_time} in {slot}")
                    else:
                        st.info(f"⚠️ {name} already marked Present in {slot}.")

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name}, {slot}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imshow("Mark Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    save_attendance(df)

    if not marked:
        st.error("❌ No known face detected.")

# =========================
# Teacher Panel
# =========================
def view_attendance():
    st.subheader("📋 Attendance Dashboard")
    df = load_attendance()
    st.dataframe(df)
    time.sleep(10)
    st.experimental_rerun()

# =========================
# Streamlit UI
# =========================
st.title("🎓 SmartAttend - AI-Based Attendance System")

tab = st.selectbox("Choose Mode", ["📌 Mark Attendance", "🔐 Teacher Panel"])

if tab == "📌 Mark Attendance":
    slot = st.selectbox("Select Slot", SLOTS)
    if st.button("📷 Start Camera and Mark Attendance"):
        recognize_and_mark_attendance(slot)

elif tab == "🔐 Teacher Panel":
    pwd = st.text_input("Enter Teacher Password", type="password")
    if pwd == PASSWORD:
        view_attendance()
    elif pwd:
        st.error("Incorrect password.")

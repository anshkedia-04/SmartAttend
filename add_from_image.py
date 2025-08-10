import streamlit as st
import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from io import BytesIO

HAAR_FILE = 'haarcascade_frontalface_default.xml'
DATASET_PATH = 'dataset'
STUDENTS_FILE = 'students.csv'

def save_student_from_uploaded_image(uploaded_file, roll, name):
    face_cascade = cv2.CascadeClassifier(HAAR_FILE)

    student_folder = os.path.join(DATASET_PATH, f"{roll}_{name}")
    os.makedirs(student_folder, exist_ok=True)

    img = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    img_array = np.array(img)

    faces = face_cascade.detectMultiScale(img_array, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        st.error("❌ No face detected in the uploaded image.")
        return

    (x, y, w, h) = faces[0]
    face_img = img_array[y:y+h, x:x+w]
    face_resized = cv2.resize(face_img, (100, 100))

    for i in range(1, 140): 
        img_path = os.path.join(student_folder, f"{i}.jpg")
        cv2.imwrite(img_path, face_resized)

    # Update students.csv
    if not os.path.exists(STUDENTS_FILE) or os.stat(STUDENTS_FILE).st_size == 0:
        df = pd.DataFrame(columns=["Roll", "Name"])
    else:
        df = pd.read_csv(STUDENTS_FILE)

    df = df[df["Roll"] != roll]
    df = pd.concat([df, pd.DataFrame([{"Roll": roll, "Name": name}])], ignore_index=True)
    df.to_csv(STUDENTS_FILE, index=False)

    st.success(f"✅ 139 face images saved for {name} and entry added to students.csv.")

# --- Streamlit UI ---
st.title("🧑‍🎓 Add Student from Uploaded Image")

uploaded_file = st.file_uploader("Upload Student Face Image", type=['jpg', 'jpeg', 'png'])
roll = st.text_input("Enter Roll Number")
name = st.text_input("Enter Name")

if st.button("Save Student"):
    if uploaded_file and roll and name:
        save_student_from_uploaded_image(uploaded_file, roll, name)
    else:
        st.warning("Please upload an image and enter both Roll and Name.")

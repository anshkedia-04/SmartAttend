import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Config
SOURCE_DATASET = '105_classes_pins_dataset'   # path to unzipped Pins dataset
DEST_DATASET = 'dataset'  # path to save processed dataset
STUDENTS_FILE = 'students.csv'
HAAR_FILE = 'haarcascade_frontalface_default.xml'

# Create destination dataset folder
os.makedirs(DEST_DATASET, exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier(HAAR_FILE)

# Load existing students.csv or create new one
if os.path.exists(STUDENTS_FILE) and os.stat(STUDENTS_FILE).st_size > 0:
    df_students = pd.read_csv(STUDENTS_FILE)
else:
    df_students = pd.DataFrame(columns=["Roll", "Name"])

# Start processing each person
start_roll = 2000  # roll numbers for Kaggle entries

for i, person_name in enumerate(sorted(os.listdir(SOURCE_DATASET)), start=1):
    person_path = os.path.join(SOURCE_DATASET, person_name)
    if not os.path.isdir(person_path):
        continue

    roll = str(start_roll + i)
    clean_name = person_name.replace(" ", "")
    folder_name = f"{roll}_{clean_name}"
    dest_folder = os.path.join(DEST_DATASET, folder_name)
    os.makedirs(dest_folder, exist_ok=True)

    print(f"🔄 Processing: {folder_name}")

    count = 0
    for file in sorted(os.listdir(person_path)):
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(person_path, file)

        try:
            img = Image.open(img_path).convert("L")  # Grayscale
            img_np = np.array(img)
            faces = face_cascade.detectMultiScale(img_np, 1.3, 5)

            if len(faces) == 0:
                continue

            (x, y, w, h) = faces[0]
            face = img_np[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (100, 100))

            count += 1
            cv2.imwrite(os.path.join(dest_folder, f"{count}.jpg"), face_resized)

            if count >= 30:  # Limit to 30 images per person
                break

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

    if count == 0:
        print(f"⚠️ No face found for {person_name}, skipping.")
        continue

    # Add entry to students.csv
    df_students = df_students[df_students["Roll"] != roll]
    df_students = pd.concat([df_students, pd.DataFrame([{"Roll": roll, "Name": clean_name}])], ignore_index=True)

df_students.to_csv(STUDENTS_FILE, index=False)
print("✅ Dataset conversion complete.")

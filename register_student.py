import cv2
import os
import pandas as pd

# Constants
haar_file = 'haarcascade_frontalface_default.xml'
dataset_path = 'dataset'
students_file = 'students.csv'

# Ensure dataset folder exists
os.makedirs(dataset_path, exist_ok=True)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(haar_file)

# Get student info
roll = input("Enter Roll Number: ").strip()
name = input("Enter Name: ").strip()
student_folder = os.path.join(dataset_path, f"{roll}_{name}")
os.makedirs(student_folder, exist_ok=True)

# Start camera and collect 30 face samples
cap = cv2.VideoCapture(0)
count = 0

print("📸 Capturing face images. Press 'Q' to stop early.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        count += 1
        img_path = os.path.join(student_folder, f"{count}.jpg")
        cv2.imwrite(img_path, face_img)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Image {count}/30", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Register Student", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 30:
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ {count} images saved in {student_folder}")

# Create students.csv if empty or non-existent
if not os.path.exists(students_file) or os.stat(students_file).st_size == 0:
    df = pd.DataFrame(columns=["Roll", "Name"])
else:
    df = pd.read_csv(students_file)

# Add new entry using concat (for pandas >= 2.0)
df = pd.concat([df, pd.DataFrame([{"Roll": roll, "Name": name}])], ignore_index=True)
df.to_csv(students_file, index=False)
print(f"📄 Student data saved to {students_file}")

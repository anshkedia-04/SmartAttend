import os
import pickle
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PIL import Image
import torch
from tqdm import tqdm

# Configuration
DATASET_DIR = 'dataset'  # Folder with balanced classes (30 images each)
EMBEDDINGS_FILE = "student_embeddings.pkl"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize FaceNet components
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# If embeddings file doesn't exist, create it
if not os.path.exists(EMBEDDINGS_FILE):
    print("🔄 Generating embeddings for dataset...")
    X = []
    y = []
    student_embeddings = {}

    for folder in tqdm(os.listdir(DATASET_DIR)):
        person_path = os.path.join(DATASET_DIR, folder)
        if not os.path.isdir(person_path):
            continue
        person_embeddings = []
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
            except:
                continue
            face = mtcnn(img)
            if face is not None:
                with torch.no_grad():
                    emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy()[0]
                    X.append(emb)
                    y.append(folder)
                    person_embeddings.append(emb)
        if person_embeddings:
            student_embeddings[folder] = np.mean(person_embeddings, axis=0)

    # Save embeddings to .pkl
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump({"X": np.array(X), "y": np.array(y), "student_embeddings": student_embeddings}, f)
    print(f"✅ Saved embeddings to {EMBEDDINGS_FILE}")
else:
    print(f"📂 Found {EMBEDDINGS_FILE}, skipping embedding generation.")

# Load embeddings
with open(EMBEDDINGS_FILE, "rb") as f:
    data = pickle.load(f)
X, y, student_embeddings = data["X"], data["y"], data["student_embeddings"]

# Train-test split & SVM
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

train_acc = accuracy_score(y_train, clf.predict(X_train))
test_acc = accuracy_score(y_test, clf.predict(X_test))

print(f"\n✅ Final Training Accuracy: {train_acc * 100:.2f}%")
print(f"✅ Final Validation/Test Accuracy: {test_acc * 100:.2f}%")

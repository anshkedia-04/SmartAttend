# model_training_mobilenetv2.ipynb

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Constants
DATASET_DIR = 'dataset'
IMG_SIZE = (128, 128)  # You can also try (96, 96)
MODEL_PATH = 'trained_model/mobilenetv2_face_model.keras'
LABELS_PATH = 'trained_model/label_classes.npy'
os.makedirs('trained_model', exist_ok=True)

# Step 1: Load and preprocess data
images, labels = [], []
for folder in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
        labels.append(folder)

X = np.array(images)
y = np.array(labels)

# Step 2: Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, stratify=y_categorical, random_state=42)

# Step 4: Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)

# Step 5: Build Model
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  # Initially freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
output = Dense(len(le.classes_), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Step 7: Train initial model
history = model.fit(datagen.flow(X_train, y_train, batch_size=16),
                    validation_data=(X_test, y_test),
                    epochs=30,
                    callbacks=[early_stop])

# Step 8: Fine-tune last 40 layers
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

fine_tune_history = model.fit(datagen.flow(X_train, y_train, batch_size=16),
                              validation_data=(X_test, y_test),
                              epochs=10,
                              callbacks=[early_stop])

# Step 9: Save model and label classes
model.save(MODEL_PATH)
np.save(LABELS_PATH, le.classes_)

# Step 10: Plot accuracy
plt.plot(history.history['accuracy'], label='Train Acc (initial)')
plt.plot(history.history['val_accuracy'], label='Val Acc (initial)')
if 'accuracy' in fine_tune_history.history:
    plt.plot(fine_tune_history.history['accuracy'], label='Train Acc (fine-tuned)')
    plt.plot(fine_tune_history.history['val_accuracy'], label='Val Acc (fine-tuned)')

plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# Step 11: Evaluate final accuracy
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"✅ Final Training Accuracy: {train_acc * 100:.2f}%")
print(f"✅ Final Validation/Test Accuracy: {test_acc * 100:.2f}%")

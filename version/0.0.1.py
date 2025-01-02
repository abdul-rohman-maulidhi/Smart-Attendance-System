import cv2
import os
import csv
import numpy as np
from datetime import datetime
from fuzzywuzzy import fuzz, process

# Direktori penyimpanan
DATASET_DIR = "face_dataset"
MODEL_DIR = "trained_model"
ATTENDANCE_FILE = "attendance.csv"

# Inisialisasi Haar Cascades dan Recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Pastikan direktori ada
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Nama", "Status Kehadiran", "Jam"])

def collect_images(person_name, num_samples=100):
    cap = cv2.VideoCapture(0)
    count = 0
    person_dir = os.path.join(DATASET_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)

    print(f"Collecting images for '{person_name}'. Press 'q' to quit early.")
    while cap.isOpened() and count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (200, 200))
            file_path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(file_path, face_resized)
            count += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"Image {count}/{num_samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Collecting Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collected {count} images for '{person_name}'.")

def train_recognizer():
    faces, labels = [], []
    label_map = {}

    print("Training the recognizer...")
    for label, person_name in enumerate(os.listdir(DATASET_DIR)):
        label_map[label] = person_name
        person_dir = os.path.join(DATASET_DIR, person_name)
        for file_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            labels.append(label)

    recognizer.train(faces, np.array(labels))
    recognizer.write(os.path.join(MODEL_DIR, "face_recognizer.yml"))

    with open(os.path.join(MODEL_DIR, "labels.txt"), "w") as f:
        for label, name in label_map.items():
            f.write(f"{label},{name}\n")
    print("Model trained and saved successfully.")

def recognize_faces_and_attendance():
    recognizer.read(os.path.join(MODEL_DIR, "face_recognizer.yml"))

    # Load label map
    label_map = {}
    with open(os.path.join(MODEL_DIR, "labels.txt"), "r") as f:
        for line in f:
            label, name = line.strip().split(",")
            label_map[int(label)] = name

    cap = cv2.VideoCapture(0)
    eye_detected = False
    smile_detected = False
    detected_name = None

    print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_resized = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            label, confidence = recognizer.predict(face_resized)
            detected_name = label_map.get(label, "Unknown")

            # Fuzzy matching for name similarity
            if confidence <= 50.0:
                matched_name = process.extractOne(detected_name, list(label_map.values()), scorer=fuzz.partial_ratio)[0]
                cv2.putText(frame, f"{matched_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Eye and Smile Detection
                roi_gray = gray[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
                smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 15)

                if len(eyes) > 0:
                    eye_detected = True
                    cv2.putText(frame, "Eye Blink Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if len(smiles) > 0:
                    smile_detected = True
                    cv2.putText(frame, "Smile Detected!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if eye_detected and smile_detected:
                    # Log attendance
                    with open(ATTENDANCE_FILE, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([matched_name, "Hadir", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                    print(f"{matched_name} marked as present.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

        cv2.imshow("Smart Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    while True:
        print("\n--- Smart Attendance System ---")
        print("1: Collect Face Data")
        print("2: Train Recognizer")
        print("3: Run Attendance System")
        print("4: Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            name = input("Enter the person's name: ")
            collect_images(name)
        elif choice == "2":
            train_recognizer()
        elif choice == "3":
            recognize_faces_and_attendance()
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")

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

# Fungsi untuk mengecek senyuman gigi yang akurat
def check_toothy_smile(smiles, face_width):
    for (sx, sy, sw, sh) in smiles:
        if sw > face_width * 0.5:  # Senyuman lebih dari 50% lebar wajah
            return True
    return False

# Fungsi untuk mendeteksi kedipan mata
def check_eye_blink(eyes_detected, previous_status):
    if eyes_detected == 0:  # Mata tertutup
        if previous_status == "OPEN":
            return "CLOSED", True  # Kedipan terdeteksi
        return "CLOSED", False
    return "OPEN", False

# Fungsi utama untuk deteksi wajah, kedipan mata, dan senyuman
def recognize_faces_and_attendance():
    recognizer.read(os.path.join(MODEL_DIR, "face_recognizer.yml"))

    # Load label map
    label_map = {}
    with open(os.path.join(MODEL_DIR, "labels.txt"), "r") as f:
        for line in f:
            label, name = line.strip().split(",")
            label_map[int(label)] = name

    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")

    eye_status = "OPEN"
    blink_detected = False
    smile_detected = False
    detected_name = None

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

            # Fuzzy name matching
            if confidence <= 50.0:
                matched_name = process.extractOne(detected_name, list(label_map.values()), scorer=fuzz.partial_ratio)[0]

                # Tampilkan persegi wajah dan nama
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"Name: {matched_name}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Area wajah untuk deteksi mata dan senyuman
                roi_gray = gray[y:y + h, x:x + w]

                # Deteksi mata
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10, minSize=(30, 30))
                eye_status, blink_detected = check_eye_blink(len(eyes), eye_status)
                if blink_detected:
                    cv2.putText(frame, "Eye Blink: Detected", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Deteksi senyuman
                smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20, minSize=(30, 30))
                smile_detected = check_toothy_smile(smiles, w)
                if smile_detected:
                    cv2.putText(frame, "Toothly Smile: Detected", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Jika kedipan mata dan senyuman terdeteksi
                if blink_detected and smile_detected:
                    cv2.putText(frame, "Detected", (x + w - 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    # Log kehadiran
                    with open(ATTENDANCE_FILE, "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([matched_name, "Hadir", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                    print(f"{matched_name} marked as present.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return  # Tutup webcam dan keluar dari fungsi

        # Tampilkan frame
        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Fungsi untuk mengumpulkan gambar wajah
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

# Fungsi untuk melatih model pengenalan wajah
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

# Menu utama
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

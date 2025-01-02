import cv2
import os
import csv
import numpy as np
from datetime import datetime
from fuzzywuzzy import fuzz, process
from version.mail import extract_attendance_from_doc, send_email_notification

# Direktori penyimpanan
DATASET_DIR = "face_dataset"
MODEL_DIR = "trained_model"
ATTENDANCE_FILE = "attendance.csv"

# Inisialisasi Haar Cascades dan Recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Variabel untuk deteksi
EYE_CLOSED_FRAMES = 2  # Ambang batas frame mata tertutup untuk satu kedipan
eye_blink_count = 0    # Counter kedipan mata
blink_counter = 0       # Counter untuk hitung mata tertutup berturut-turut
smile_detected = False # Flag untuk senyuman

# Fungsi untuk mengecek kedipan mata
def check_eye_blink(eyes_detected, eye_closed_frames):
    global eye_blink_count, blink_counter
    if eyes_detected == 0:  # Mata tertutup
        blink_counter += 1
    else:  # Mata terbuka
        if blink_counter >= eye_closed_frames:
            eye_blink_count += 1  # Increment kedipan valid
        blink_counter = 0  # Reset counter setelah mata terbuka kembali

    return eye_blink_count

# Fungsi untuk mengecek senyuman gigi yang akurat
def check_toothy_smile(smiles, face_width):
    for (sx, sy, sw, sh) in smiles:
        if sw > face_width * 0.5:  # Senyuman lebih dari 50% lebar wajah
            return True
    return False

# Fungsi utama untuk deteksi wajah, kedipan mata, dan senyuman
def recognize_faces_and_attendance():
    global eye_blink_count, smile_detected, blink_counter
    
    # Reset variabel untuk sesi baru
    eye_blink_count = 0
    smile_detected = False
    blink_counter = 0
    
    recognizer.read(os.path.join(MODEL_DIR, "face_recognizer.yml"))

    # Load label map
    label_map = {}
    with open(os.path.join(MODEL_DIR, "labels.txt"), "r") as f:
        for line in f:
            label, name = line.strip().split(",")
            label_map[int(label)] = name

    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_resized = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            try:
                label, confidence = recognizer.predict(face_resized)
                if confidence > 50.0:  # Confidence too low, treat as "Unknown"
                    detected_name = "Unknown"
                else:
                    detected_name = label_map.get(label, "Unknown")

            except Exception as e:  # Handle cases where prediction fails
                detected_name = "Unknown"

            # Display face rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, f"Name: {detected_name}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if detected_name == "Unknown" else (0, 255, 0), 2)

            # Additional checks for blinks and smiles if the face is recognized
            if detected_name != "Unknown":
                roi_gray = gray[y:y + h, x:x + w]

                # Eye detection
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10, minSize=(30, 30))
                current_blink_count = check_eye_blink(len(eyes), EYE_CLOSED_FRAMES)
                cv2.putText(frame, f"Eye Blink Count: {current_blink_count}", (x, y - 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Smile detection
                smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20, minSize=(30, 30))
                if check_toothy_smile(smiles, w):
                    smile_detected = True
                    cv2.putText(frame, "Toothy Smile: Detected", (x, y - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Verification logic
                if current_blink_count >= 3 and smile_detected:
                    cv2.putText(frame, "Verified", (x + w - 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                    # Log kehadiran
                    with open(ATTENDANCE_FILE, "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([detected_name, "Hadir", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                    print(f"{detected_name} marked as present.")
                    
                    # Mengirim email notifikasi
                    # email_subject = "Kehadiran Anda Tercatat"
                    # email_body = f"Halo {detected_name},\n\nKehadiran Anda telah berhasil tercatat pada {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
                    # recipient_email = "rohmanmaulidhi@gmail.com"  # Ganti dengan email peserta
                    # send_email_notification(recipient_email, email_subject, email_body)
                    
                    cap.release()
                    cv2.destroyAllWindows()
                    return  # Exit after attendance is marked

        # Tampilkan frame
        cv2.imshow("Smart Attendance System", frame)
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

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
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

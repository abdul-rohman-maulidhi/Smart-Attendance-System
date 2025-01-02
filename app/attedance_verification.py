import cv2  # OpenCV, pustaka untuk pemrosesan gambar dan video
import os  # Modul untuk berinteraksi dengan sistem file
import csv  # Modul untuk membaca dan menulis file CSV

from datetime import datetime # Untuk mengambil tanggal dan waktu secara real-time

from app.face_recognition import get_user_info_by_id
from app.speech import speak_text, listen_for_phrase
from app.fuzzy_logic import check_eye_blink, fuzzy_smile_check
from app.notification_email import send_email_notification

# Direktori penyimpanan
DATASET_DIR = "./data/face_dataset"  # Direktori untuk menyimpan dataset wajah
MODEL_DIR = "./data/trained_model"  # Direktori untuk menyimpan model yang telah dilatih

ATTENDANCE_FILE = "./data/attendance/attendance.csv"  # File CSV untuk mencatat kehadiran
USERS_FILE = "./data/users/users.csv"  # File CSV untuk menyimpan data pengguna

# Inisialisasi Haar Cascades dan Recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  
# Detektor wajah menggunakan Haar Cascade bawaan OpenCV

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")  
# Detektor mata menggunakan Haar Cascade bawaan OpenCV

smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")  
# Detektor senyuman menggunakan Haar Cascade bawaan OpenCV

recognizer = cv2.face.LBPHFaceRecognizer_create()  
# Inisialisasi Local Binary Pattern Histogram (LBPH) Recognizer untuk pengenalan wajah

# Variabel untuk deteksi
EYE_CLOSED_FRAMES = 2  # Ambang batas frame mata tertutup untuk satu kedipan
eye_blink_count = 0    # Counter kedipan mata
blink_counter = 0       # Counter untuk hitung mata tertutup berturut-turut
smile_detected = False # Flag untuk senyuman

# Fungsi utama untuk deteksi wajah, kedipan mata, dan senyuman
def recognize_faces_and_attendance():
    # Fungsi utama untuk deteksi wajah, kedipan mata, dan senyuman, serta mencatat kehadiran.

    global eye_blink_count, smile_detected, blink_counter  # Gunakan variabel global untuk kedipan dan senyuman.

    # Reset variabel untuk sesi baru
    eye_blink_count = 0  # Mengatur ulang jumlah kedipan mata.
    smile_detected = False  # Mengatur ulang status senyuman.
    blink_counter = 0  # Mengatur ulang counter kedipan berturut-turut.

    recognizer.read(os.path.join(MODEL_DIR, "face_recognizer.yml"))  # Memuat model pengenalan wajah dari file.

    # Tampilkan instruksi kepada pengguna untuk memverifikasi kehadiran
    print("please blink 3x then smile for first verification!")
    speak_text("please blink 3x then smile for first verification!")  # Menggunakan text-to-speech untuk instruksi.

    # Load label map untuk menerjemahkan ID ke nama pengguna.
    label_map = {}
    with open(os.path.join(MODEL_DIR, "labels.txt"), "r") as f:
        for line in f:
            label, user_id = line.strip().split(",")  # Membaca label dan ID dari file.
            label_map[int(label)] = user_id  # Menyimpan ke dalam dictionary.

    cap = cv2.VideoCapture(0)  # Mengaktifkan kamera.
    print("Press 'q' to quit.")  # Petunjuk untuk keluar dari aplikasi.

    while cap.isOpened():  # Loop selama kamera terbuka.
        ret, frame = cap.read()  # Membaca frame dari kamera.
        if not ret:  # Jika gagal membaca frame, keluar dari loop.
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Mengonversi frame ke grayscale.
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Mendeteksi wajah dalam frame.

        for (x, y, w, h) in faces:  # Iterasi melalui setiap wajah yang terdeteksi.
            face_resized = cv2.resize(gray[y:y + h, x:x + w], (200, 200))  # Memperbesar ukuran wajah untuk prediksi.
            try:
                # Memprediksi wajah menggunakan recognizer.
                label, confidence = recognizer.predict(face_resized)
                if confidence > 50.0:  # Jika confidence rendah, wajah dianggap "Unknown".
                    detected_id = "Unknown"
                else:
                    detected_id = label_map.get(label, "Unknown")  # Mencocokkan label dengan ID pengguna.
            except Exception as e:
                detected_id = "Unknown"  # Jika terjadi error, wajah dianggap "Unknown".

            # Ambil data pengguna berdasarkan ID.
            user_info = get_user_info_by_id(detected_id)
            if user_info:  # Jika data pengguna ditemukan.
                user_name, email = user_info  # Mengambil nama dan email.
                display_text = f"{detected_id} | {user_name}"  # Format teks untuk ditampilkan.
            else:
                display_text = f"Unknown"  # Tampilkan sebagai "Unknown" jika data tidak ditemukan.

            # Tampilkan persegi wajah dan informasi pengguna di layar.
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Gambar kotak di sekitar wajah.
            cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        (0, 255, 0) if detected_id != "Unknown" else (0, 0, 255), 2)
            if detected_id != "Unknown":
                cv2.putText(frame, f"Confidence: {confidence:.2f}%", (x, y + h + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Pemeriksaan tambahan untuk kedipan mata dan senyuman jika wajah dikenali.
            if detected_id != "Unknown":
                roi_gray = gray[y:y + h, x:x + w]  # Region of Interest (ROI) untuk wajah.

                # Deteksi mata.
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10, minSize=(30, 30))
                current_blink_count = check_eye_blink(len(eyes), EYE_CLOSED_FRAMES)  # Periksa kedipan mata.
                cv2.putText(frame, f"Eye Blink Count: {current_blink_count}", (x, y - 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Deteksi senyuman.
                smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20, minSize=(30, 30))
                smile_detected = fuzzy_smile_check(smiles, w)  # Gunakan fuzzy logic untuk memeriksa senyuman.
                if smile_detected:  # Jika senyuman terdeteksi.
                    cv2.putText(frame, "Toothy Smile: Detected", (x, y - 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Logika verifikasi kehadiran.
                if current_blink_count >= 3 and smile_detected:  # Jika kedipan dan senyuman terdeteksi.
                    print("Please say 'I am here' to confirm your attendance.")
                    speak_text("Please say 'I am here' to confirm your attendance.")  # Instruksi konfirmasi.
                    response = listen_for_phrase()  # Dengarkan respons pengguna.
                    if response and "i am here" in response:  # Jika respons valid.
                        cv2.putText(frame, "Verified", (x + w - 30, y + 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

                        # Catat kehadiran ke file.
                        with open(ATTENDANCE_FILE, "a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([detected_id, user_name, "Hadir", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                        print(f"{user_name} (ID: {detected_id}) marked as present.")

                        # Kirim email notifikasi.
                        email_subject = "Kehadiran Anda Tercatat"
                        email_body = (
                            f"Halo {user_name},\n\n"
                            f"Kehadiran Anda telah berhasil dicatat pada {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n\n"
                            f"ID: {detected_id}\n\n"
                            f"Terima kasih."
                        )
                        send_email_notification(email, email_subject, email_body)  # Kirim email.

                        print("Your attendance has been recorded, please check your email to see it.")
                        speak_text("Your attendance has been recorded, please check your email to see it.")

                        cap.release()  # Lepaskan kamera.
                        cv2.destroyAllWindows()  # Tutup semua jendela OpenCV.
                        return  # Keluar setelah kehadiran tercatat.
                    else:
                        print("Kehadiran Anda gagal dicatat. Silakan coba lagi.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return  # Keluar jika gagal mencatat kehadiran.

        # Tampilkan frame di jendela OpenCV.
        cv2.imshow("Smart Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Tekan 'q' untuk keluar.
            break

    cap.release()  # Lepaskan kamera.
    cv2.destroyAllWindows()  # Tutup semua jendela OpenCV.
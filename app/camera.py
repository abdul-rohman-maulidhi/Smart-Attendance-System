import cv2  # OpenCV, pustaka untuk pemrosesan gambar, objek, video bahkan dalam deteksi dan pengenalan
import os # Modul untuk berinteraksi dengan sistem file
import csv # Modul untuk membaca dan menulis file CSV

# Direktori penyimpanan
DATASET_DIR = "./data/face_dataset" # Direktori untuk menyimpan dataset wajah

USERS_FILE = "./data/users/users.csv" # File CSV untuk menyimpan data pengguna

# Inisialisasi Haar Cascades dan Recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  
# Detektor wajah menggunakan Haar Cascade bawaan OpenCV

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")  
# Detektor mata menggunakan Haar Cascade bawaan OpenCV

smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")  
# Detektor senyuman menggunakan Haar Cascade bawaan OpenCV

recognizer = cv2.face.LBPHFaceRecognizer_create()  
# Inisialisasi Local Binary Pattern Histogram (LBPH) Recognizer untuk pengenalan wajah


def collect_images():
    # Fungsi untuk mengumpulkan gambar wajah dan menyimpannya di dataset lokal.

    # Input nama, ID, dan email di dalam fungsi.
    person_id = input("Enter the person's ID (NIP/NIM): ").strip()  # Meminta input ID pengguna.
    person_name = input("Enter the person's name: ").strip()  # Meminta input nama pengguna.
    email = input("Enter the person's Email: ").strip()  # Meminta input email pengguna.

    # Buat direktori berdasarkan ID pengguna.
    cap = cv2.VideoCapture(0)  # Mengaktifkan kamera untuk pengumpulan gambar.
    count = 0  # Counter untuk menghitung jumlah gambar yang berhasil diambil.
    person_dir = os.path.join(DATASET_DIR, person_id)  # Direktori penyimpanan gambar pengguna berdasarkan ID.
    os.makedirs(person_dir, exist_ok=True)  # Membuat direktori jika belum ada.

    # Simpan data pengguna ke dalam `users.csv`.
    users_file = USERS_FILE  # Lokasi file penyimpanan data pengguna.
    if not os.path.exists(users_file):  # Jika file belum ada, buat file baru dengan header.
        with open(users_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Name", "Email"])  # Menulis header ke file CSV.

    # Tambahkan data pengguna ke file.
    with open(users_file, "a", newline="") as f:  # Membuka file CSV dalam mode append.
        writer = csv.writer(f)
        writer.writerow([person_id, person_name, email])  # Menambahkan data pengguna ke file.

    # Mulai pengumpulan gambar.
    print(f"Collecting images for '{person_name}' (ID: {person_id}). Press 'q' to quit early.")  
    # Memberikan informasi kepada pengguna tentang proses pengumpulan gambar.
    
    while cap.isOpened() and count < 100:  # Mengambil hingga 100 gambar atau sampai 'q' ditekan.
        ret, frame = cap.read()  # Membaca frame dari kamera.
        if not ret:  # Jika gagal membaca frame, keluar dari loop.
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Mengonversi frame ke grayscale.
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Mendeteksi wajah pada frame.

        for (x, y, w, h) in faces:  # Iterasi melalui setiap wajah yang terdeteksi.
            face = gray[y:y + h, x:x + w]  # Memotong ROI (Region of Interest) wajah.
            face_resized = cv2.resize(face, (200, 200))  # Mengubah ukuran wajah ke 200x200 piksel.
            file_path = os.path.join(person_dir, f"{count}.jpg")  # Path penyimpanan file gambar.
            cv2.imwrite(file_path, face_resized)  # Menyimpan gambar wajah ke file.
            count += 1  # Increment counter setelah berhasil menyimpan gambar.

            # Menampilkan progress pengumpulan gambar di layar.
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Gambar kotak di sekitar wajah.
            cv2.putText(frame, f"Image {count}/100", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Tampilkan jumlah gambar yang terkumpul.

        cv2.imshow("Collecting Faces", frame)  # Menampilkan frame di jendela OpenCV.
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Keluar jika tombol 'q' ditekan.
            break

    cap.release()  # Melepaskan kamera setelah selesai.
    cv2.destroyAllWindows()  # Menutup semua jendela OpenCV.
    print(f"Collected {count} images for '{person_name}' (ID: {person_id}').")  
    # Memberikan laporan akhir tentang jumlah gambar yang berhasil dikumpulkan.
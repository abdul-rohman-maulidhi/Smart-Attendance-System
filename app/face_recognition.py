import cv2  # OpenCV, pustaka untuk pemrosesan gambar dan video
import os  # Modul untuk berinteraksi dengan sistem file
import csv  # Modul untuk membaca dan menulis file CSV
import numpy as np  # Pustaka untuk operasi numerik dan array

# Direktori penyimpanan
DATASET_DIR = "./data/face_dataset"  # Direktori untuk menyimpan dataset wajah
MODEL_DIR = "./data/trained_model"  # Direktori untuk menyimpan model yang telah dilatih

ATTENDANCE_FILE = "./data/attendance/attendance.csv" # File CSV untuk mencatat kehadiran
USERS_FILE = "./data/users/users.csv" # File CSV untuk menyimpan data pengguna

# Inisialisasi Haar Cascades dan Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Inisialisasi Local Binary Pattern Histogram (LBPH) Recognizer untuk pengenalan wajah

# Fungsi untuk mendapatkan informasi pengguna berdasarkan ID
def get_user_info_by_id(user_id):
    users_file = USERS_FILE  # Lokasi file pengguna
    if not os.path.exists(users_file):  # Periksa apakah file pengguna ada
        return None  # Jika tidak ada, kembalikan None

    with open(users_file, "r") as f:  # Buka file pengguna dalam mode baca
        reader = csv.DictReader(f)  # Membaca file sebagai dictionary
        for row in reader:  # Iterasi setiap baris dalam file
            if row["ID"] == user_id:  # Jika ID cocok dengan user_id
                return row["Name"], row["Email"]  # Kembalikan nama dan email
    return None  # Jika tidak ditemukan, kembalikan None

def train_recognizer():
    # Fungsi untuk melatih model pengenalan wajah menggunakan data gambar yang telah dikumpulkan.

    faces, labels = [], []  # List untuk menyimpan gambar wajah dan label masing-masing.
    label_map = {}  # Peta label untuk menghubungkan label numerik dengan ID pengguna.

    print("Training the recognizer...")  # Memberikan informasi bahwa proses pelatihan sedang dimulai.

    # Iterasi melalui setiap direktori pengguna di folder dataset.
    for label, user_id in enumerate(os.listdir(DATASET_DIR)):  
        # Enumerasi memberikan label numerik untuk setiap ID pengguna.
        label_map[label] = user_id  # Menyimpan hubungan label numerik dengan ID pengguna.
        user_dir = os.path.join(DATASET_DIR, user_id)  # Mendapatkan path direktori pengguna.

        # Iterasi melalui setiap file gambar di direktori pengguna.
        for file_name in os.listdir(user_dir):
            img_path = os.path.join(user_dir, file_name)  # Path gambar wajah.
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Membaca gambar dalam mode grayscale.
            faces.append(img)  # Menambahkan gambar ke dalam daftar `faces`.
            labels.append(label)  # Menambahkan label numerik ke daftar `labels`.

    # Melatih model pengenalan wajah menggunakan gambar dan label yang dikumpulkan.
    recognizer.train(faces, np.array(labels))  
    # Menyimpan model yang telah dilatih ke file `face_recognizer.yml`.
    recognizer.write(os.path.join(MODEL_DIR, "face_recognizer.yml"))

    # Menyimpan peta label ke file `labels.txt`.
    with open(os.path.join(MODEL_DIR, "labels.txt"), "w") as f:
        for label, user_id in label_map.items():  
            # Menyimpan setiap pasangan label dan ID pengguna ke dalam file.
            f.write(f"{label},{user_id}\n")  

    print("Model trained and saved successfully.")  # Memberikan informasi bahwa pelatihan selesai dan model telah disimpan.
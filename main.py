import sys

from app.camera import collect_images
from app.face_recognition import train_recognizer
from app.attedance_verification import recognize_faces_and_attendance
from app.attendance import display_attendance, open_attendance_with_excel


def main():
    if len(sys.argv) > 1:  # Mengecek apakah ada argumen tambahan yang diberikan saat menjalankan skrip
        action = sys.argv[1]  # Mengambil argumen pertama setelah nama file
        # Mengecek nilai argumen untuk menentukan tindakan
        if action == "collect_images":
            collect_images()  # Memanggil fungsi untuk mengumpulkan data wajah
        elif action == "train_recognizer":
            train_recognizer()  # Memanggil fungsi untuk melatih model pengenal wajah
        elif action == "run_attendance":
            recognize_faces_and_attendance()  # Memanggil fungsi untuk menjalankan sistem absensi
        elif action == "view_attendance":
            display_attendance()  # Memanggil fungsi untuk menampilkan data absensi di terminal
        elif action == "open_excel":
            open_attendance_with_excel()  # Memanggil fungsi untuk membuka data absensi menggunakan Excel
    else:
        # Jika tidak ada argumen tambahan, gunakan pendekatan berbasis menu interaktif
        while True:  # Loop tak terbatas sampai pengguna memilih keluar
            print("\n--- Smart Attendance System ---")  # Menampilkan judul menu
            print("1: Collect Face Data")  # Opsi untuk mengumpulkan data wajah
            print("2: Train Recognizer")  # Opsi untuk melatih model pengenal wajah
            print("3: Run Attendance System")  # Opsi untuk menjalankan sistem absensi
            print("4: View Attendance Records in Terminal")  # Opsi untuk melihat data absensi di terminal
            print("5: Open Attendance Records in Excel")  # Opsi untuk membuka data absensi menggunakan Excel
            print("6: Exit")  # Opsi untuk keluar dari program
            choice = input("Enter your choice: ")  # Meminta input pengguna untuk memilih opsi menu

            # Mengecek input pengguna dan menjalankan fungsi yang sesuai
            if choice == "1":
                collect_images()  # Memanggil fungsi untuk mengumpulkan data wajah
            elif choice == "2":
                train_recognizer()  # Memanggil fungsi untuk melatih model pengenal wajah
            elif choice == "3":
                recognize_faces_and_attendance()  # Memanggil fungsi untuk menjalankan sistem absensi
            elif choice == "4":
                display_attendance()  # Memanggil fungsi untuk menampilkan data absensi di terminal
            elif choice == "5":
                open_attendance_with_excel()  # Memanggil fungsi untuk membuka data absensi di Excel
            elif choice == "6":
                print("Exiting...")  # Menampilkan pesan keluar
                break  # Menghentikan loop, keluar dari program
            else:
                print("Invalid choice. Try again.")  # Menampilkan pesan jika input tidak valid

# Menu utama
if __name__ == "__main__":
    main()
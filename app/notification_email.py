import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Fungsi untuk mengirim notifikasi email
def send_email_notification(to_email, subject, body):
    # Konfigurasi email
    from_email = "abdulrohmanm1304@gmail.com"  # Alamat email pengirim
    password = "wzqz ijmp mbyw hnmb"  # Password aplikasi email pengirim

    try:
        # Setup email server
        server = smtplib.SMTP('smtp.gmail.com', 587)  # Menggunakan server SMTP Gmail pada port 587
        server.starttls()  # Memulai koneksi TLS (untuk keamanan)
        server.login(from_email, password)  # Login ke akun email menggunakan kredensial

        # Buat pesan email
        msg = MIMEMultipart()  # Membuat objek email multipart
        msg['From'] = from_email  # Menentukan alamat pengirim
        msg['To'] = to_email  # Menentukan alamat penerima
        msg['Subject'] = subject  # Menentukan subjek email
        msg.attach(MIMEText(body, 'plain'))  # Menambahkan isi email dengan format teks biasa

        # Kirim email
        server.send_message(msg)  # Mengirim email
        print(f"Email berhasil dikirim ke {to_email}")  # Konfirmasi bahwa email berhasil dikirim
        server.quit()  # Menutup koneksi ke server email
    except Exception as e:
        # Jika ada kesalahan, tangkap dan tampilkan pesan kesalahan
        print(f"Gagal mengirim email: {e}")


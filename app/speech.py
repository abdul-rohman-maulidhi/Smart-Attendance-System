import pyttsx3  # Pustaka untuk teks ke suara (text-to-speech)
import speech_recognition as sr # Pustaka untuk pengenalan suara

# Inisialisasi Text-to-Speech
engine = pyttsx3.init()  # Menginisialisasi engine Text-to-Speech
engine.setProperty('rate', 150)  # Mengatur kecepatan bicara (kata per menit)
engine.setProperty('volume', 0.9)  # Mengatur volume suara (0.0 hingga 1.0)

# Fungsi untuk mengucapkan teks
def speak_text(text):
    engine.say(text)  # Menggunakan engine Text-to-Speech untuk mengucapkan teks
    engine.runAndWait()  # Menjalankan proses Text-to-Speech hingga selesai

# Fungsi untuk mendengarkan dan mengenali ucapan
def listen_for_phrase():
    recognizer = sr.Recognizer()  # Membuat objek recognizer untuk mengenali suara
    with sr.Microphone() as source:  # Menggunakan mikrofon sebagai sumber input suara
        print("Listening for 'I am here'...")  # Memberi tahu pengguna bahwa aplikasi sedang mendengarkan
        try:
            recognizer.adjust_for_ambient_noise(source, duration=1)  
            # Menyesuaikan sensitivitas deteksi suara berdasarkan kebisingan lingkungan selama 1 detik
            audio = recognizer.listen(source, timeout=10)  
            # Mendengarkan input suara dengan batas waktu 10 detik
            recognized_text = recognizer.recognize_google(audio)  
            # Menggunakan API Google untuk mengenali teks dari audio
            print(f"Recognized: {recognized_text}")  
            # Menampilkan teks yang dikenali ke layar
            return recognized_text.lower()  
            # Mengembalikan teks yang dikenali dalam huruf kecil untuk mempermudah perbandingan
        except sr.UnknownValueError:
            # Jika tidak dapat mengenali ucapan
            print("Could not understand the audio.")
        except sr.RequestError as e:
            # Jika ada kesalahan dalam permintaan ke layanan pengenalan ucapan
            print(f"Could not request results; {e}")
        except sr.WaitTimeoutError:
            # Jika waktu mendengarkan habis tanpa menerima suara
            print("Listening timed out.")
    return None  # Mengembalikan None jika tidak ada teks yang dikenali

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email_notification(to_email, subject, body):
    # Konfigurasi email
    from_email = "abdulrohmanm1304@gmail.com"
    password = "wzqz ijmp mbyw hnmb"

    try:
        # Setup email server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)

        # Buat pesan email
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Kirim email
        server.send_message(msg)
        print(f"Email berhasil dikirim ke {to_email}")
        server.quit()
    except Exception as e:
        print(f"Gagal mengirim email: {e}")


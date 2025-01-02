from . import camera, attendance, attedance_verification, face_recognition, fuzzy_logic, notification_email, speech

from .camera import collect_images
from .attendance import display_attendance, open_attendance_with_excel
from .attedance_verification import recognize_faces_and_attendance
from .face_recognition import train_recognizer
from .speech import speak_text, listen_for_phrase
from .notification_email import send_email_notification
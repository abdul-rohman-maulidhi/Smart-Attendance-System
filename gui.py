import tkinter as tk
from tkinter import messagebox, ttk
import subprocess
import sys

class SmartAttendanceSystemGUI:
    def __init__(self, master):
        self.master = master
        master.title("Smart Attendance System")
        master.geometry("500x600")
        master.configure(bg='#f0f0f0')

        # Custom style
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Modern theme
        self.style.configure('TButton', 
            font=('Arial', 12, 'bold'), 
            padding=10,
            background='#4CAF50'
        )

        # Title
        self.title_label = tk.Label(
            master, 
            text="Smart Attendance System", 
            font=('Arial', 18, 'bold'), 
            bg='#f0f0f0', 
            fg='#333333'
        )
        self.title_label.pack(pady=20)

        # Button frame for better organization
        self.button_frame = tk.Frame(master, bg='#f0f0f0')
        self.button_frame.pack(expand=True, fill='both', padx=50, pady=20)

        # Create buttons with custom colors and styles
        button_configs = [
            ("Collect Face Data", self.collect_images, '#2196F3'),  # Blue
            ("Train Recognizer", self.train_recognizer, '#FFC107'),  # Amber
            ("Run Attendance System", self.run_attendance, '#4CAF50'),  # Green
            ("View Attendance Records", self.view_attendance_terminal, '#9C27B0'),  # Purple
            ("Open Attendance in Excel", self.open_attendance_excel, '#FF5722')  # Deep Orange
        ]

        for text, command, color in button_configs:
            btn = ttk.Button(
                self.button_frame, 
                text=text, 
                command=command,
                style='Custom.TButton'
            )
            self.style.configure(f'{text}.TButton', background=color)
            btn.pack(fill='x', pady=10)
            btn.configure(style=f'{text}.TButton')

        # Exit button
        self.exit_btn = ttk.Button(
            self.button_frame, 
            text="Exit", 
            command=master.quit,
            style='Exit.TButton'
        )
        self.style.configure('Exit.TButton', background='#F44336')  # Red
        self.exit_btn.pack(fill='x', pady=10)

    def run_script(self, script_function):
        try:
            result = subprocess.run(
                [sys.executable, 'main.py', script_function], 
                capture_output=True, 
                text=True
            )
            if result.returncode == 0:
                messagebox.showinfo("Success", result.stdout)
            else:
                messagebox.showerror("Error", result.stderr)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def collect_images(self):
        self.run_script("collect_images")

    def train_recognizer(self):
        self.run_script("train_recognizer")

    def run_attendance(self):
        self.run_script("run_attendance")

    def view_attendance_terminal(self):
        self.run_script("view_attendance")

    def open_attendance_excel(self):
        self.run_script("open_excel")

def main():
    root = tk.Tk()
    gui = SmartAttendanceSystemGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
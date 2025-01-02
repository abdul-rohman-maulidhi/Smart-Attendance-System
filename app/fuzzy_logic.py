import numpy as np # Pustaka untuk operasi numerik dan array
import skfuzzy as fuzz  # Pustaka untuk logika fuzzy
import skfuzzy.control as ctrl  # Modul kontrol untuk logika fuzzy

# Variabel untuk deteksi
EYE_CLOSED_FRAMES = 2  # Ambang batas frame mata tertutup untuk satu kedipan
eye_blink_count = 0    # Counter kedipan mata
blink_counter = 0       # Counter untuk hitung mata tertutup berturut-turut
smile_detected = False # Flag untuk senyuman

# Fuzzy Logic untuk deteksi kedipan mata
def fuzzy_eye_blink(eyes_detected, eye_closed_frames):
    # Variabel fuzzy
    blink_duration = ctrl.Antecedent(np.arange(0, 10, 1), 'blink_duration')  # Durasi kedipan dalam frame
    blink_confidence = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'blink_confidence')  # Kepercayaan kedipan

    # Membership functions
    blink_duration['short'] = fuzz.trapmf(blink_duration.universe, [0, 0, 2, 4])
    blink_duration['medium'] = fuzz.trimf(blink_duration.universe, [2, 4, 6])
    blink_duration['long'] = fuzz.trapmf(blink_duration.universe, [4, 6, 10, 10])

    blink_confidence['low'] = fuzz.trapmf(blink_confidence.universe, [0, 0, 0.3, 0.5])
    blink_confidence['medium'] = fuzz.trimf(blink_confidence.universe, [0.3, 0.5, 0.7])
    blink_confidence['high'] = fuzz.trapmf(blink_confidence.universe, [0.5, 0.7, 1.0, 1.0])

    # Aturan fuzzy
    rules = [
        ctrl.Rule(blink_duration['short'], blink_confidence['low']),
        ctrl.Rule(blink_duration['medium'], blink_confidence['medium']),
        ctrl.Rule(blink_duration['long'], blink_confidence['high']),
    ]

    # Sistem kontrol fuzzy
    system = ctrl.ControlSystem(rules)  # Inisialisasi sistem kontrol fuzzy
    simulation = ctrl.ControlSystemSimulation(system)  # Simulasi sistem kontrol

    # Hitung kepercayaan berdasarkan input
    simulation.input['blink_duration'] = blink_counter  # Masukkan jumlah frame kedipan
    simulation.compute()  # Jalankan simulasi fuzzy
    confidence = simulation.output['blink_confidence']  # Dapatkan output kepercayaan

    return confidence  # Kembalikan kepercayaan

# Panggil fungsi fuzzy dalam pendeteksian kedipan mata
def check_eye_blink(eyes_detected, eye_closed_frames):
    global eye_blink_count, blink_counter
    if eyes_detected == 0:  # Jika mata tertutup
        blink_counter += 1  # Tambahkan counter kedipan
    else:  # Jika mata terbuka
        if blink_counter >= eye_closed_frames:  # Jika jumlah frame mata tertutup melewati ambang batas
            confidence = fuzzy_eye_blink(eyes_detected, eye_closed_frames)  # Hitung kepercayaan fuzzy
            if confidence > 0.5:  # Jika kepercayaan melebihi ambang batas
                eye_blink_count += 1  # Tambahkan hitungan kedipan valid
        blink_counter = 0  # Reset counter setelah mata terbuka

    return eye_blink_count  # Kembalikan jumlah kedipan


# Fungsi fuzzy untuk memeriksa senyuman berdasarkan ukuran senyuman
def fuzzy_smile_check(smiles, face_width):
    if len(smiles) > 0:  # Periksa jika ada senyuman yang terdeteksi
        smile_ratio = smiles[0][2] / face_width  # Rasio lebar senyuman terhadap lebar wajah
    else:
        smile_ratio = 0  # Jika tidak ada senyuman, rasio 0

    # Variabel fuzzy
    smile_size = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'smile_size')  # Ukuran senyuman relatif
    smile_confidence = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'smile_confidence')  # Kepercayaan senyuman

    # Membership functions
    smile_size['small'] = fuzz.trapmf(smile_size.universe, [0, 0, 0.3, 0.5])
    smile_size['medium'] = fuzz.trimf(smile_size.universe, [0.3, 0.5, 0.7])
    smile_size['large'] = fuzz.trapmf(smile_size.universe, [0.5, 0.7, 1.0, 1.0])

    smile_confidence['low'] = fuzz.trapmf(smile_confidence.universe, [0, 0, 0.3, 0.5])
    smile_confidence['medium'] = fuzz.trimf(smile_confidence.universe, [0.3, 0.5, 0.7])
    smile_confidence['high'] = fuzz.trapmf(smile_confidence.universe, [0.5, 0.7, 1.0, 1.0])

    # Aturan fuzzy
    rules = [
        ctrl.Rule(smile_size['small'], smile_confidence['low']),
        ctrl.Rule(smile_size['medium'], smile_confidence['medium']),
        ctrl.Rule(smile_size['large'], smile_confidence['high']),
    ]

    # Sistem kontrol fuzzy
    system = ctrl.ControlSystem(rules)  # Inisialisasi sistem kontrol fuzzy
    simulation = ctrl.ControlSystemSimulation(system)  # Simulasi sistem kontrol

    # Hitung kepercayaan berdasarkan input
    simulation.input['smile_size'] = smile_ratio  # Masukkan rasio senyuman
    simulation.compute()  # Jalankan simulasi fuzzy
    confidence = simulation.output['smile_confidence']  # Dapatkan output kepercayaan

    return confidence > 0.5  # Kembalikan True jika kepercayaan melebihi ambang batas
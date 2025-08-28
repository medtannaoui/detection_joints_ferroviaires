###################### Application de visualisation des detections  ######################################


import os
import traceback
import sys
import numpy as np
import sounddevice as sd

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QFrame,
    QRadioButton, QHBoxLayout
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from PyQt5.QtMultimedia import QSound
from PyQt5.QtCore import QTimer

import time
from importlib import reload

import functions as fct
import models
reload(models)
reload(fct)
import librosa as lb
from tensorflow.keras.models import load_model
from metric_seq2seq import AUCSeq, F1ScoreSeq, PrecisionSeq, RecallSeq
import pickle
#from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)

############################ Les constantes ###############################################
nb_mels = 64
F_max = 22000

# ----------------- Fonction de prédiction ------------------
def predict_from_audio(file_path):
    wave, sr = lb.load(file_path, sr=None)
    wave -= np.mean(wave)
    N_window = fct.get_near_pof2(0.02, sr)
    mels = fct.get_MELgrame(wave, Nt = N_window, NB_FEATURES=nb_mels, FMAX=F_max, power=2.0, FS=sr,
                 overlap=2, energie="both", derive1=True, derive2=True)[0].T
    

    #choix du modèle 

    with open(os.path.join("models","audios","audio_1.pkl"), "rb") as f:
        data = pickle.load(f)
    
    pca, scaler, hist_metr = data[3]

    custom_objects = {
    "AUCSeq": AUCSeq,
    "F1ScoreSeq": F1ScoreSeq,
    "PrecisionSeq": PrecisionSeq,
    "RecallSeq": RecallSeq
}

    model = load_model(os.path.join("models","audios","audio_1.keras"),
                       custom_objects=custom_objects,compile=False)

    X, _, _, _, _ = fct.pretrait(mels, np.zeros(len(mels)), pca=None, scaler=scaler,apply_pca=False,apply_norm=True,seq=True,seq_before=9,seq_apres=10,overlap=0.5)

    y_pred = model.predict(X)
    print(y_pred)
    y_pred_classes = (y_pred[:,1] > 0.5).astype(int)
    print(y_pred_classes)
    '''
    y_pred_classes = np.zeros(len(mels))
    y_pred_classes[100:200] = 1'''
    
    return wave, sr, y_pred_classes


# --------------------- Application PyQt ---------------------
class AudioPredictorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Détection de défauts audio")
        self.setGeometry(300, 300, 800, 600)
        self.setStyleSheet("background-color: #f4f6f7;")
        self.setup_ui()
        self.last_wave = None
        self.last_sr = None
        self.last_y_pred = None
        self.last_audio_path = None
        self.playhead_line = None
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.update_playhead)
        self.play_start_time = None
        self.zoom_start = None
        self.zoom_end = None

    def setup_ui(self):
        layout = QVBoxLayout()

        # Import
        self.load_btn = QPushButton("🎵 Importer un fichier audio")
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.load_btn.clicked.connect(self.load_audio)
        layout.addWidget(self.load_btn)

        # Résultats
        self.result_label = QLabel("Résultats :")
        self.result_label.setFont(QFont("Segoe UI", 11))
        self.result_label.setStyleSheet("padding: 10px; background-color: #ecf0f1; border: 1px solid #bdc3c7; border-radius: 5px;")
        self.result_label.setAlignment(Qt.AlignTop)
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)

        # Boutons Radio
        self.radio_signal = QRadioButton("Signal audio")
        self.radio_spectro = QRadioButton("Spectrogramme MEL")
        self.radio_signal.setChecked(True)

        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.radio_signal)
        radio_layout.addWidget(self.radio_spectro)
        layout.addLayout(radio_layout)

        # Bouton Affichage
        self.plot_btn = QPushButton("📈 Afficher le signal")
        self.plot_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        self.plot_btn.clicked.connect(self.plot_signal)
        layout.addWidget(self.plot_btn)
        self.play_zoom_btn = QPushButton("🔊 Écouter la zone zoomée")
        self.play_zoom_btn.setStyleSheet("""
            QPushButton {
                background-color: #e67e22;
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #d35400;
            }
        """)
        self.play_zoom_btn.clicked.connect(self.play_zoomed_audio)
        layout.addWidget(self.play_zoom_btn)

        # Bouton Écoute
        self.play_btn = QPushButton("▶️ Écouter l'audio")
        self.play_btn.setStyleSheet("""
            QPushButton {
                background-color: #e67e22;
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #d35400;
            }
        """)
        self.play_btn.clicked.connect(self.play_audio)
        layout.addWidget(self.play_btn)

        # Graphique avec zoom
        self.canvas = FigureCanvas(plt.Figure(figsize=(10, 3)))
        self.ax = self.canvas.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def load_audio(self):
        audio_path, _ = QFileDialog.getOpenFileName(
            self, "Choisir un fichier audio", "", "Fichiers audio (*.wav *.mp3)"
        )
        if audio_path:
            try:
                wave, sr, y_pred = predict_from_audio(audio_path)
                #self.result_label.setText(f"<b>Résultats (bincount de y_pred)</b> :<br>{result}")
                self.last_wave = wave
                self.last_sr = sr
                self.last_y_pred = y_pred
                self.last_audio_path = audio_path
            except Exception as e:
                exc_type, exc_value, exc_tb = sys.exc_info()
                ligne = exc_tb.tb_lineno  # numéro de ligne
                self.result_label.setText(
                    f"<b>Erreur</b> : {str(e)}<br><b>Ligne :</b> {ligne}"
                )
                print("Trace complète :")
                traceback.print_exc()


    def plot_signal(self):
        if self.last_wave is not None and self.last_y_pred is not None:
            self.ax.clear()

            if self.radio_signal.isChecked():
                time = np.arange(len(self.last_wave)) / self.last_sr
                self.ax.plot(time, self.last_wave, color="blue", label="Signal audio")
                seg_len = len(self.last_wave) // len(self.last_y_pred)

                for i, cls in enumerate(self.last_y_pred):
                    if cls != 0:
                        start = i * seg_len
                        end = (i + 1) * seg_len
                        self.ax.axvspan(start / self.last_sr, end / self.last_sr, color='red', alpha=0.3)

                self.ax.set_title("Signal audio")
                self.ax.set_xlabel("Temps (s)")
                self.ax.set_ylabel("Amplitude")

            elif self.radio_spectro.isChecked():
                N_window = fct.get_near_pof2(0.03, self.last_sr)
                
                mels = fct.get_MELgrame(audio = self.last_wave , Nt= N_window, NB_FEATURES=nb_mels, FMAX=F_max, power=2.0, FS=self.last_sr,
                 overlap=2, energie="both", derive1=True, derive2=True)[0].T
                mels = mels[:-1]
                self.ax.imshow(mels.T, origin="lower", aspect="auto", cmap="magma", 
                               extent=[0, len(self.last_wave) / self.last_sr, 0, F_max])
                self.ax.set_title("Spectrogramme MEL")
                self.ax.set_xlabel("Temps (s)")
                self.ax.set_ylabel("Fréquence (Hz)")

            self.canvas.draw()
        else:
            self.result_label.setText("Veuillez d'abord importer un fichier audio.")

    def play_audio(self):
        if self.last_audio_path:
            if self.last_audio_path.endswith(".wav"):
                QSound.play(self.last_audio_path)
            else:
                self.result_label.setText("⚠️ La lecture n'est disponible que pour les fichiers .wav.")
        else:
            self.result_label.setText("Aucun fichier audio chargé.")
    


    def play_zoomed_audio(self):
        if self.last_wave is None or self.last_sr is None:
            self.result_label.setText("Aucun fichier audio chargé.")
            return

        x_min, x_max = self.ax.get_xlim()
        idx_start = max(0, int(x_min * self.last_sr))
        idx_end = min(len(self.last_wave), int(x_max * self.last_sr))

        if idx_end <= idx_start:
            self.result_label.setText("Zoom incorrect ou plage invalide.")
            return

        self.zoom_start = x_min
        self.zoom_end = x_max
        wave_zoomed = self.last_wave[idx_start:idx_end]

        try:
            # Stoppe le son et le timer au cas où
            sd.stop()
            self.play_timer.stop()
            self.remove_playhead()

            # Joue le son
            if np.max(np.abs(wave_zoomed)) > 0:
                wave_zoomed = wave_zoomed / np.max(np.abs(wave_zoomed))
            sd.play(wave_zoomed, samplerate=self.last_sr)

            # Crée la ligne de lecture (playhead)
            self.playhead_line = self.ax.axvline(self.zoom_start, color='black', linestyle='-', linewidth=2)
            self.canvas.draw()

            # Démarre le timer pour bouger la ligne toutes les 30 ms
            self.play_start_time = time.time()
            self.play_timer.start(30)

            self.result_label.setText(f"Lecture de l'extrait de {x_min:.2f}s à {x_max:.2f}s...")

        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            ligne = exc_tb.tb_lineno
            self.result_label.setText(f"Erreur de lecture audio : {e}<br>Ligne : {ligne}")
            traceback.print_exc()
    
    
    def update_playhead(self):
        elapsed = time.time() - self.play_start_time
        current_time = self.zoom_start + elapsed

        if current_time >= self.zoom_end:
            self.play_timer.stop()
            self.remove_playhead()
            return

        if self.playhead_line:
            self.playhead_line.set_xdata([current_time])

            self.canvas.draw_idle()
    

    def remove_playhead(self):
        if self.playhead_line:
            self.playhead_line.remove()
            self.playhead_line = None
            self.canvas.draw_idle()





# --------------------- Lancemenet de l'application  ---------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioPredictorApp()
    window.show()
    sys.exit(app.exec_())

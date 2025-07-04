import os
import threading
import customtkinter as ctk
from tkinter import filedialog
from faster_whisper import WhisperModel
import torch

# ----------- Paramètres disponibles ----------
MODELS = {
    "Base": "base",
    "Small": "small",
    "Medium": "medium",
    "Large v3 (CPU lourd)": "large-v3"
}
LANGS = {
    "Français": "fr",
    "Anglais": "en",
    "Espagnol": "es",
    "Allemand": "de",
    "Italien": "it",
    "Portugais": "pt",
    "Néerlandais": "nl",
    "Russe": "ru",
    "Arabe": "ar",
    "Chinois": "zh",
    "Japonais": "ja"
}
DEFAULT_LANG = "Français"
DEFAULT_MODEL = "Large v3 (CPU lourd)"

# ------------ Début patch VAD Silero (.onnx) - facultatif mais conseillé -----------
import sys, shutil, pathlib

def _ensure_vad_assets():
    """
    Copie les .onnx du dossier 'assets' (packagé avec PyInstaller)
    dans le cache ~/.cache/faster-whisper/assets/
    afin que faster-whisper les trouve quel que soit l'environnement.
    """
    if hasattr(sys, "_MEIPASS"):
        src_assets = pathlib.Path(sys._MEIPASS, "assets")
    else:
        src_assets = pathlib.Path(__file__).with_suffix("").parent / "assets"
    dst_assets = pathlib.Path.home() / ".cache" / "faster-whisper" / "assets"
    dst_assets.mkdir(parents=True, exist_ok=True)
    for fname in ("silero_encoder_v5.onnx", "silero_decoder_v5.onnx"):
        src = src_assets / fname
        dst = dst_assets / fname
        try:
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
        except Exception as e:
            print(f"[WARN] Impossible de copier {fname}: {e}")
_ensure_vad_assets()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ------------ Fin patch VAD Silero -----------

# -------------- Classe principale -------------
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Transcripteur Whisper")
        self.geometry("700x540")
        self.resizable(False, False)

        # Paramètres modèle/langue/fichiers
        self.files = []
        self.current_file = 0
        self.model = None
        self.loaded_model_name = None

        # ---- Interface (frame du haut) ----
        top_frame = ctk.CTkFrame(self)
        top_frame.pack(pady=12, padx=10, fill="x")

        ctk.CTkLabel(top_frame, text="Modèle Whisper :").pack(side="left")
        self.combo_model = ctk.CTkComboBox(top_frame, values=list(MODELS.keys()), width=140)
        self.combo_model.set(DEFAULT_MODEL)
        self.combo_model.pack(side="left", padx=(5, 20))

        ctk.CTkLabel(top_frame, text="Langue :").pack(side="left")
        self.combo_lang = ctk.CTkComboBox(top_frame, values=list(LANGS.keys()), width=120)
        self.combo_lang.set(DEFAULT_LANG)
        self.combo_lang.pack(side="left", padx=5)

        # ---- Sélection de fichiers ----
        self.btn_select = ctk.CTkButton(self, text="Sélectionner fichiers audio…", command=self.select_files)
        self.btn_select.pack(pady=8)

        # ---- Bouton de lancement ----
        self.btn_lancer = ctk.CTkButton(self, text="Lancer la transcription", command=self.lancer_lot, state="disabled")
        self.btn_lancer.pack(pady=4)

        # ---- Affichage des logs/avancement ----
        self.txt_progress = ctk.CTkTextbox(self, width=670, height=320)
        self.txt_progress.pack(pady=8, padx=10)
        self.txt_progress.insert("end", "Bienvenue !\nSélectionnez plusieurs fichiers audio, puis cliquez sur 'Lancer la transcription'.\n")

        # ---- Affichage barre de progression (pour chaque fichier) ----
        self.progressbar = ctk.CTkProgressBar(self, width=650)
        self.progressbar.pack(pady=2)
        self.progressbar.set(0)

    # ----------- Choix des fichiers (multi-sélection) -----------
    def select_files(self):
        filenames = filedialog.askopenfilenames(
            title="Sélectionnez les fichiers à transcrire",
            filetypes=[("Audio", "*.mp3 *.wav *.m4a *.flac")]
        )
        self.files = list(filenames)
        self.current_file = 0
        self.progressbar.set(0)
        if self.files:
            self.txt_progress.insert("end", f"{len(self.files)} fichiers sélectionnés :\n")
            for f in self.files:
                self.txt_progress.insert("end", f"— {os.path.basename(f)}\n")
            self.btn_lancer.configure(state="normal")
        else:
            self.btn_lancer.configure(state="disabled")
        self.txt_progress.see("end")

    def load_model(self):
        model_name = MODELS[self.combo_model.get()]
        if self.model is not None and self.loaded_model_name == model_name:
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        self.txt_progress.insert("end", f"Chargement du modèle {model_name} sur {device}…\n")
        self.txt_progress.see("end")
        self.model = WhisperModel(model_name, device=device, compute_type=compute)
        self.loaded_model_name = model_name

    # ----------- Lancer la transcription par lot -----------
    def lancer_lot(self):
        self.btn_lancer.configure(state="disabled")
        if not self.files:
            self.txt_progress.insert("end", "Aucun fichier sélectionné.\n")
            return
        self.load_model()
        self.txt_progress.insert("end", "\nDébut du traitement par lot…\n")
        self.progressbar.set(0)
        self.after(100, self.transcrire_prochain)

    def transcrire_prochain(self):
        if self.current_file < len(self.files):
            fichier = self.files[self.current_file]
            self.txt_progress.insert("end", f"\nTranscription de {os.path.basename(fichier)} …\n")
            self.txt_progress.see("end")
            self.progressbar.set(0)
            threading.Thread(target=self.transcribe_thread, args=(fichier,), daemon=True).start()
        else:
            self.txt_progress.insert("end", "\nTous les fichiers ont été traités.\n")
            self.progressbar.set(1)
            self.btn_lancer.configure(state="normal")

    # ----------- Thread de transcription d’un fichier -----------
    def transcribe_thread(self, fichier):
        try:
            lang_code = LANGS[self.combo_lang.get()]
            model = self.model

            segments, info = model.transcribe(
                fichier,
                language=lang_code,
                beam_size=5,
                vad_filter=True
            )
            duration = info.duration or 1
            done, full_text = 0.0, ""

            for seg in segments:
                done += seg.end - seg.start
                pct = min(done / duration, 1.0)
                self.after(0, lambda pct=pct: self.progressbar.set(pct))
                self.after(0, lambda txt=seg.text: self.txt_progress.insert("end", txt))
                full_text += seg.text + "\n"

            # Sauvegarde automatique
            out_dir = "transcriptions"
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, os.path.splitext(os.path.basename(fichier))[0] + ".txt")
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(full_text)
            self.after(0, lambda: self.after_transcription(fichier, out_file))
        except Exception as e:
            self.after(0, lambda: self.txt_progress.insert("end", f"\n[ERREUR] {e}\n"))
            self.after(0, self.transcription_suivante)

    # ----------- Après transcription d’un fichier -----------
    def after_transcription(self, fichier, out_file):
        self.txt_progress.insert("end", f"\nTranscription terminée pour {os.path.basename(fichier)}. "
                                        f"Fichier : {out_file}\n")
        self.txt_progress.see("end")
        self.current_file += 1
        self.after(200, self.transcrire_prochain)

    # ----------- En cas d'erreur, passer au suivant -----------
    def transcription_suivante(self):
        self.current_file += 1
        self.after(100, self.transcrire_prochain)


# ----------- Lancement de l’appli -----------
if __name__ == "__main__":
    app = App()
    app.mainloop()

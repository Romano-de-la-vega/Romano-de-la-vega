"""
Application de transcription audio basée sur faster‑whisper
Interface graphique : CustomTkinter (Tkinter modernisé)

Fonctions :
✔ Choix du modèle Whisper (base → large‑v3)
✔ Choix de la langue (Fr par défaut, 10 langues)
✔ Sélecteur de fichier audio (mp3/wav/m4a/flac)
✔ Progression temps réel + texte qui défile
✔ Enregistrement du .txt où l’utilisateur veut
"""

import os, threading, queue, time
import customtkinter as ctk
from tkinter import filedialog, messagebox
from faster_whisper import WhisperModel

import sys, shutil, pathlib

if hasattr(sys, "_MEIPASS"):
    test_path = pathlib.Path(sys._MEIPASS) / "faster_whisper" / "assets" / "silero_encoder_v5.onnx"
    print("Vérif PyInstaller assets:", test_path, "Existe?", test_path.exists())
else:
    test_path = pathlib.Path(__file__).parent / "faster_whisper" / "assets" / "silero_encoder_v5.onnx"
    print("Vérif DEV assets:", test_path, "Existe?", test_path.exists())


def _ensure_vad_assets():
    """
    Copie les .onnx du dossier 'assets' (packagé avec PyInstaller)
    dans le cache ~/.cache/faster-whisper/assets/
    afin que faster-whisper les trouve quel que soit l'environnement.
    """
    # Chemin source (dans le code ou dans le bundle .exe)
    if hasattr(sys, "_MEIPASS"):
        src_assets = pathlib.Path(sys._MEIPASS, "assets")
    else:
        src_assets = pathlib.Path(__file__).with_suffix("").parent / "assets"

    # Chemin destination (cache officiel)
    dst_assets = pathlib.Path.home() / ".cache" / "faster-whisper" / "assets"
    dst_assets.mkdir(parents=True, exist_ok=True)

    for fname in ("silero_encoder_v5.onnx", "silero_decoder_v5.onnx"):
        src = src_assets / fname
        dst = dst_assets / fname
        try:
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
        except Exception as e:
            # on n'arrête pas l'app pour autant ; whisper retéléchargera si besoin
            print(f"[WARN] Impossible de copier {fname}: {e}")

_ensure_vad_assets()

# Facultatif : éviter le double‑chargement OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# -------------------------------------------------


def resource_path(relative_path):
    """Retourne le chemin absolu vers un fichier de ressource,
    compatible PyInstaller (en .exe) comme en .py."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.abspath(relative_path)

# ---------- Paramètres init ----------
ctk.set_appearance_mode("System")   # "Dark" | "Light" | "System"
ctk.set_default_color_theme("blue") # ou "green", "dark-blue"

# ---------- Constantes ----------
MODELS = {
    "Tiny":   "tiny",
    "Base":   "base",
    "Small":  "small",
    "Medium": "medium",
    "Large v3 (CPU lourd)": "large-v3"
}

LANGS = {
    "Français":  "fr",
    "Anglais":   "en",
    "Espagnol":  "es",
    "Allemand":  "de",
    "Italien":   "it",
    "Portugais": "pt",
    "Néerlandais":"nl",
    "Russe":     "ru",
    "Arabe":     "ar",
    "Chinois":   "zh",
    "Japonais":  "ja"
}
DEFAULT_LANG = "Français"

# ---------- Thread worker ----------
def worker_transcribe(audio_path, model_size, lang_code, q):
    """
    Lance la transcription puis poste dans la queue :
    ('progress', pct)   ou   ('text', segment_text)   ou   ('done', full_text) ou ('error', msg)
    """
    try:
        m = WhisperModel(model_size, device="cpu", compute_type="int8")
    except Exception as e:
        q.put(("error", f"Erreur chargement modèle : {e}"))
        return

    try:
        segments, info = m.transcribe(audio_path, language=lang_code,
                                      beam_size=5, vad_filter=True)
    except Exception as e:
        q.put(("error", f"Erreur transcription : {e}"))
        return

    duration = info.duration or 1
    done, full_text = 0.0, ""
    for seg in segments:
        done += seg.end - seg.start
        q.put(("progress", min(done / duration, 1.0)))
        q.put(("text", seg.text))
        full_text += seg.text + "\n"

    q.put(("done", full_text))

# ---------- Interface ----------
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.iconbitmap(resource_path("icon.ico"))
        self.title("Transcripteur Whisper")
        self.geometry("780x560")
        self.minsize(720, 520)

        # Queue de communication thread → GUI
        self.q = queue.Queue()
        self.thread = None

        # ---- 1ère ligne : modèles & langues ----
        top_frame = ctk.CTkFrame(self)
        top_frame.pack(padx=15, pady=(15, 5), fill="x")

        ctk.CTkLabel(top_frame, text="Modèle :").pack(side="left", padx=(10, 5))
        self.combo_model = ctk.CTkComboBox(top_frame, values=list(MODELS.keys()), width=140)
        self.combo_model.set("Large v3 (CPU lourd)")
        self.combo_model.pack(side="left")

        ctk.CTkLabel(top_frame, text="  Langue :").pack(side="left", padx=(20, 5))
        self.combo_lang = ctk.CTkComboBox(top_frame, values=list(LANGS.keys()), width=120)
        self.combo_lang.set(DEFAULT_LANG)
        self.combo_lang.pack(side="left")

        # ---- 2e ligne : Sélecteur de fichier ----
        mid_frame = ctk.CTkFrame(self)
        mid_frame.pack(padx=15, pady=5, fill="x")

        self.entry_file = ctk.CTkEntry(mid_frame, placeholder_text="Choisir un fichier audio…")
        self.entry_file.pack(side="left", expand=True, fill="x", padx=(10, 5), pady=10)

        ctk.CTkButton(mid_frame, text="Parcourir", command=self.browse_file)\
            .pack(side="left", padx=(0, 10), pady=10)

        # ---- Barre de progression ----
        self.progress = ctk.CTkProgressBar(self, height=18)
        self.progress.set(0)
        self.progress.pack(padx=20, pady=(5, 10), fill="x")

        # ---- Zone de texte ----
        self.txt = ctk.CTkTextbox(self, wrap="word")
        self.txt.configure(state="disabled")
        self.txt.pack(padx=15, pady=5, expand=True, fill="both")

        # ---- Boutons action ----
        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.pack(pady=10)

        self.btn_start = ctk.CTkButton(bottom, text="Lancer", command=self.start_transcription, state="disabled")
        self.btn_start.pack(side="left", padx=10)

        self.btn_save = ctk.CTkButton(bottom, text="Enregistrer", command=self.save_txt, state="disabled")
        self.btn_save.pack(side="left", padx=10)

        ctk.CTkButton(bottom, text="Quitter", command=self.destroy).pack(side="left", padx=10)

        # ---- Boucle de surveillance de la queue ----
        self.after(200, self.poll_queue)

    # ---------- Handlers ----------
    def browse_file(self):
        types = [("Audio", "*.mp3 *.wav *.m4a *.flac"), ("Tous les fichiers", "*.*")]
        path = filedialog.askopenfilename(title="Sélectionner un fichier audio", filetypes=types)
        if path:
            self.entry_file.delete(0, "end")
            self.entry_file.insert(0, path)
            self.btn_start.configure(state="normal")

    def start_transcription(self):
        audio_path = self.entry_file.get()
        if not os.path.isfile(audio_path):
            messagebox.showerror("Erreur", "Fichier invalide.")
            return

        # Reset UI
        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.configure(state="disabled")
        self.progress.set(0)
        self.btn_start.configure(state="disabled")
        self.btn_save.configure(state="disabled")

        # Thread
        model_size = MODELS[self.combo_model.get()]
        lang_code  = LANGS[self.combo_lang.get()]
        self.q.queue.clear()
        self.thread = threading.Thread(target=worker_transcribe,
                                       args=(audio_path, model_size, lang_code, self.q),
                                       daemon=True)
        self.thread.start()

    def poll_queue(self):
        try:
            while True:
                msg_type, payload = self.q.get_nowait()
                if msg_type == "progress":
                    self.progress.set(payload)
                elif msg_type == "text":
                    self.txt.configure(state="normal")
                    self.txt.insert("end", payload)
                    self.txt.see("end")
                    self.txt.configure(state="disabled")
                elif msg_type == "done":
                    self.full_text = payload
                    self.btn_save.configure(state="normal")
                    messagebox.showinfo("Terminé", "Transcription terminée !")
                elif msg_type == "error":
                    messagebox.showerror("Erreur", payload)
                    self.btn_start.configure(state="normal")
        except queue.Empty:
            pass
        self.after(200, self.poll_queue)

    def save_txt(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt",
                                            filetypes=[("Texte", "*.txt")],
                                            title="Enregistrer la transcription")
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.full_text)
                messagebox.showinfo("Enregistré", f"Fichier enregistré :\n{path}")
            except Exception as e:
                messagebox.showerror("Erreur", str(e))

# ---------- Lancement ----------
if __name__ == "__main__":
    app = App()
    app.mainloop()

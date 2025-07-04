import os
import sys
import shutil
import pathlib
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import customtkinter as ctk
from tkinter import filedialog
from faster_whisper import WhisperModel

try:
    import torch  # facultatif ; seulement pour détecter un éventuel GPU
except ImportError:
    torch = None

# -------------------------------------------------------------
# Paramètres disponibles
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# Patch VAD Silero (.onnx) – exécuté une seule fois au premier run
# -------------------------------------------------------------

def _ensure_vad_assets_once():
    """Copie les .onnx nécessaires au VAD dans le cache utilisateur."""
    if hasattr(_ensure_vad_assets_once, "_done"):
        return  # déjà copié
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

    _ensure_vad_assets_once._done = True

_ensure_vad_assets_once()

# KMP_DUPLICATE_LIB_OK évite un crash sur certains environnements Windows/Anaconda
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# -------------------------------------------------------------
# Classe principale
# -------------------------------------------------------------
class App(ctk.CTk):
    UPDATE_INTERVAL = 0.25  # secondes entre deux refresh UI

    def __init__(self):
        super().__init__()
        self.title("Transcripteur Whisper – version optimisée")
        self.geometry("700x560")
        self.resizable(False, False)

        # Paramètres et état
        self.files = []
        self.current_file_idx = 0
        self.model = None  # instance WhisperModel réutilisée
        self.current_model_key = None
        self.executor = ThreadPoolExecutor(max_workers=max(1, os.cpu_count() // 2))

        # -------- Frame du haut (choix modèle/langue) --------
        top = ctk.CTkFrame(self)
        top.pack(pady=12, padx=10, fill="x")

        ctk.CTkLabel(top, text="Modèle Whisper :").pack(side="left")
        self.combo_model = ctk.CTkComboBox(top, values=list(MODELS.keys()), width=150)
        self.combo_model.set(DEFAULT_MODEL)
        self.combo_model.pack(side="left", padx=(5, 20))

        ctk.CTkLabel(top, text="Langue :").pack(side="left")
        self.combo_lang = ctk.CTkComboBox(top, values=list(LANGS.keys()), width=130)
        self.combo_lang.set(DEFAULT_LANG)
        self.combo_lang.pack(side="left", padx=5)

        # -------- Sélection fichiers --------
        self.btn_select = ctk.CTkButton(self, text="Sélectionner fichiers audio…", command=self.select_files)
        self.btn_select.pack(pady=8)

        # -------- Bouton lancer --------
        self.btn_run = ctk.CTkButton(self, text="Lancer la transcription", command=self.run_batch, state="disabled")
        self.btn_run.pack(pady=4)

        # -------- Zone log --------
        self.txt_log = ctk.CTkTextbox(self, width=670, height=330)
        self.txt_log.pack(pady=8, padx=10)
        self._log("Bienvenue ! Sélectionnez un ou plusieurs fichiers audio, puis cliquez sur ‘Lancer la transcription’.\n")

        # -------- Barre progression --------
        self.progress = ctk.CTkProgressBar(self, width=650)
        self.progress.pack(pady=4)
        self.progress.set(0)

    # ---------------------------------------------------------
    # Utils
    # ---------------------------------------------------------
    def _log(self, text: str):
        """Ajoute du texte et scroll vers le bas."""
        self.txt_log.insert("end", text)
        self.txt_log.see("end")

    # ---------------------------------------------------------
    # Sélection fichiers
    # ---------------------------------------------------------
    def select_files(self):
        filenames = filedialog.askopenfilenames(
            title="Sélectionnez les fichiers à transcrire",
            filetypes=[("Audio", "*.mp3 *.wav *.m4a *.flac")],
        )
        self.files = list(filenames)
        self.current_file_idx = 0
        self.progress.set(0)

        self.txt_log.delete("1.0", "end")  # on nettoie les logs précédents
        if self.files:
            self._log(f"{len(self.files)} fichier(s) sélectionné(s) :\n")
            for f in self.files:
                self._log(f" — {os.path.basename(f)}\n")
            self.btn_run.configure(state="normal")
        else:
            self._log("Aucun fichier sélectionné.\n")
            self.btn_run.configure(state="disabled")

    # ---------------------------------------------------------
    # Chargement (ou réutilisation) du modèle
    # ---------------------------------------------------------
    def _get_or_load_model(self):
        key = self.combo_model.get()
        if self.model is not None and key == self.current_model_key:
            return self.model  # déjà prêt

        # (Re)chargement requis
        model_name = MODELS[key]
        device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        self._log(f"\n[INFO] Chargement du modèle {model_name} ({device}, {compute_type})…\n")
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.current_model_key = key
        return self.model

    # ---------------------------------------------------------
    # Lancer le batch
    # ---------------------------------------------------------
    def run_batch(self):
        if not self.files:
            self._log("Aucun fichier à traiter.\n")
            return

        self.btn_run.configure(state="disabled")
        self.progress.set(0)
        self._log("\nDébut du traitement…\n")

        # Assure le modèle prêt avant de lancer le premier thread
        self._get_or_load_model()
        self.after(100, self._process_next_file)

    def _process_next_file(self):
        if self.current_file_idx >= len(self.files):
            self._log("\nTous les fichiers ont été transcrits.\n")
            self.progress.set(1)
            self.btn_run.configure(state="normal")
            return

        filepath = self.files[self.current_file_idx]
        self._log(f"\n——\n[{self.current_file_idx + 1}/{len(self.files)}] {os.path.basename(filepath)}\n")
        self.progress.set(0)

        # Lancement du thread de transcription
        threading.Thread(target=self._transcribe_file, args=(filepath,), daemon=True).start()

    # ---------------------------------------------------------
    # Transcription d’un fichier (thread dédié)
    # ---------------------------------------------------------
    def _transcribe_file(self, filepath: str):
        start_time = time.time()
        try:
            model = self.model  # déjà chargé
            lang_code = LANGS[self.combo_lang.get()]

            segments, info = model.transcribe(
                filepath,
                language=lang_code,
                beam_size=5,
                vad_filter=True,
            )

            duration_audio = info.duration or 1
            done_seconds = 0.0
            last_ui = 0.0  # dernière mise à jour UI

            out_dir = "transcriptions"
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(
                out_dir,
                os.path.splitext(os.path.basename(filepath))[0] + ".txt",
            )

            with open(out_file, "w", encoding="utf-8") as out_f:
                for seg in segments:
                    out_f.write(seg.text + "\n")
                    done_seconds += seg.end - seg.start

                    # Mise à jour UI limitée à UPDATE_INTERVAL
                    now = time.time()
                    if now - last_ui >= self.UPDATE_INTERVAL or done_seconds >= duration_audio:
                        pct = min(done_seconds / duration_audio, 1.0)
                        self.after(0, lambda p=pct: self.progress.set(p))
                        last_ui = now

            elapsed = time.time() - start_time
            self.after(0, lambda: self._on_file_done(filepath, out_file, elapsed))

        except Exception as e:
            self.after(0, lambda: self._on_file_error(filepath, e))

    # ---------------------------------------------------------
    # Callbacks UI post‑transcription
    # ---------------------------------------------------------
    def _on_file_done(self, filepath: str, out_file: str, elapsed: float):
        self._log(
            f"Transcription terminée ({elapsed:.1f}s). Fichier texte : {out_file}\n"
        )
        self.current_file_idx += 1
        self.after(200, self._process_next_file)

    def _on_file_error(self, filepath: str, err: Exception):
        self._log(f"[ERREUR] {os.path.basename(filepath)} : {err}\n")
        self.current_file_idx += 1
        self.after(200, self._process_next_file)


# -------------------------------------------------------------
# Lancement de l’application
# -------------------------------------------------------------
if __name__ == "__main__":
    app = App()
    app.mainloop()

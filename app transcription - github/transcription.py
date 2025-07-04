import sys
import os
from faster_whisper import WhisperModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


audio_path = sys.argv[1]
basename = os.path.splitext(os.path.basename(audio_path))[0]
out_dir = "transcriptions"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, f"{basename}.txt")

# Ici, "medium" pour la qualité supérieure, device="cpu" pour que ça marche partout
model = WhisperModel("large-v3", device="cpu", compute_type="int8")

full_text = ""

# Lance la transcription avec streaming segment par segment
segments, info = model.transcribe(audio_path, language="fr", beam_size=5, vad_filter=True)

for segment in segments:
    print(segment.text, flush=True)
    full_text += segment.text + "\n"

with open(out_file, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"\nTranscription terminée : {out_file}")

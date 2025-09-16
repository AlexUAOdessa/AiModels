# ==============================
# УСТАНОВКА ЗАВИСИМОСТЕЙ (в консоли):
#   pip install transformers datasets accelerate torch soundfile pydub nltk
#
# УСТАНОВИТЬ Git LFS:
#   git lfs install
#
# КЛОНИРОВАНИЕ МОДЕЛЕЙ:
#   git clone https://huggingface.co/parler-tts/parler_tts_large
#   git clone https://huggingface.co/parler-tts/parler_tts_mini
# ==============================

import os
import torch
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from pydub import AudioSegment
import nltk

# Скачать punkt для разбиения текста на предложения
nltk.download("punkt", quiet=True)

# === НАСТРОЙКИ ===
INPUT_FILE = "input.txt"        # входной текстовый файл (до 5000 символов)
OUTPUT_DIR = "output_audio"     # куда сохранять результат
FINAL_FILE = "final_output.wav" # имя итогового файла

USE_LARGE = False   # True = large, False = mini
VOICE = "female"    # "male" или "female"
SPEED = "medium"    # "slow", "medium", "fast"
EMOTION = "neutral" # "neutral", "happy", "sad", "angry", "excited"

# === УСТРОЙСТВО ===
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# === ВЫБОР МОДЕЛИ ===
model_path = "./parler_tts_large" if USE_LARGE else "./parler_tts_mini"

# === ЗАГРУЗКА МОДЕЛИ ===
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_path, torch_dtype=dtype
).to(device)

# === ФУНКЦИЯ: создание голосового промпта ===
def build_prompt(voice="female", speed="medium", emotion="neutral"):
    parts = []
    if voice == "female":
        parts.append("female voice")
    elif voice == "male":
        parts.append("male voice")

    if speed in ["slow", "medium", "fast"]:
        parts.append(f"{speed} speed")

    if emotion != "neutral":
        parts.append(f"{emotion} emotion")

    return "A " + ", ".join(parts) + ", no background noise"

prompt_voice = build_prompt(VOICE, SPEED, EMOTION)

# === ФУНКЦИЯ: генерация аудио ===
def generate_audio(text, filename):
    inputs = processor(text=text, prompt=prompt_voice, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_audio = model.generate(**inputs, max_new_tokens=1000)
    waveform = processor.batch_decode(generated_audio, return_tensors="pt")
    audio = waveform.cpu().numpy().squeeze()
    sf.write(filename, audio, 16000)

# === СОЗДАЕМ ПАПКУ ДЛЯ ВЫХОДА ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === ЧТЕНИЕ ВХОДНОГО ТЕКСТА ===
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    text = f.read().strip()

if len(text) > 5000:
    raise ValueError("⚠️ Текст слишком длинный! Максимум 5000 символов.")

# === РАЗБИЕНИЕ НА ПРЕДЛОЖЕНИЯ ===
sentences = nltk.sent_tokenize(text)
print(f"🔹 Текст разбит на {len(sentences)} сегментов.")

# === ГЕНЕРАЦИЯ ДЛЯ КАЖДОГО СЕГМЕНТА ===
segments_audio = []
for i, sentence in enumerate(sentences, 1):
    filename = os.path.join(OUTPUT_DIR, f"segment_{i}.wav")
    print(f"🎙 Генерация сегмента {i}/{len(sentences)}: {sentence[:60]}...")
    generate_audio(sentence, filename)
    segments_audio.append(AudioSegment.from_wav(filename))

# === ОБЪЕДИНЕНИЕ В ОДИН ФАЙЛ ===
final_audio = AudioSegment.silent(duration=0)
for seg in segments_audio:
    final_audio += seg + AudioSegment.silent(duration=300)  # пауза 0.3с

final_path = os.path.join(OUTPUT_DIR, FINAL_FILE)
final_audio.export(final_path, format="wav")

print(f"✅ Итоговый файл сохранён: {final_path}")
import os
import torch
from diffusers import AutoPipelineForText2Image, KandinskyV22PriorPipeline
from PIL import Image
from huggingface_hub import HfApi  # Добавили для проверки/скачивания с таймаутом

# Для работы этого кода необходимо:
# 1. Установить зависимости: pip install diffusers transformers accelerate torch pillow huggingface_hub
# 2. Клонировать две модели (или скачать через CLI):
#    - Prior: git clone https://huggingface.co/kandinsky-community/kandinsky-2-2-prior
#    - Decoder: git clone https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder
# 3. Убедиться, что Git LFS установлен: git lfs install
# 4. Если таймауты — используйте VPN или huggingface-cli download

# Локальные пути (укажите свои)
model_paths = {
    "kandinsky-community/kandinsky-2-2-decoder": "./kandinsky-2-2-decoder",
    "kandinsky-community/kandinsky-2-2-prior": "./kandinsky-2-2-prior",
}

# Функция для проверки полноты модели (опционально)
def check_model_complete(local_path, model_id):
    required_files = ['config.json', 'model.safetensors.index.json']  # Базовые файлы
    if not os.path.exists(local_path):
        return False
    for file in required_files:
        if not os.path.exists(os.path.join(local_path, file)):
            print(f"⚠️ Missing {file} in {local_path}. Model incomplete — will try to download.")
            return False
    print(f"✅ Model {local_path} seems complete.")
    return True

# Подсказка для генерации
prompt = "A spaceship is flying toward a black hole"
negative_prompt = "low resolution, blurry, bad anatomy"

# Проверяем модели заранее
for model_name, local_path in model_paths.items():
    check_model_complete(local_path, model_name)

for model_name, local_path in model_paths.items():
    print(f"\n=== Loading {model_name} from {local_path} ===")

    # Проверяем, что локальная модель существует и полная
    if not os.path.exists(local_path) or not check_model_complete(local_path, model_name):
        print(f"❌ Local path '{local_path}' incomplete. Trying fallback download...")
        try:
            # Fallback: Скачиваем с увеличенным таймаутом (через HfApi)
            api = HfApi()
            api.hf_hub_download(
                repo_id=model_name,
                filename="model.safetensors",  # Пример; diffusers сам скачает все
                local_dir=local_path,
                timeout=300  # 5 минут таймаут
            )
            print(f"✅ Fallback download attempted for {model_name}")
        except Exception as e:
            print(f"❌ Fallback failed: {e}. Use VPN or CLI download.")
            continue

    # Загружаем prior (если это prior-модель)
    if "prior" in model_name:
        pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
            local_path,
            torch_dtype=torch.float16,
            local_files_only=True  # Принудительно локальные файлы, без онлайн
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe_prior = pipe_prior.to(device)
        print(f"✅ Prior pipeline loaded on {device}")

        # Генерируем embeddings из промпта
        out = pipe_prior(
            prompt,
            negative_prompt=negative_prompt,
            height=512,
            width=512,
            num_inference_steps=25,
            guidance_scale=4.0
        )
        image_embeds = out.image_embeds
        negative_image_embeds = out.negative_image_embeds
        print(f"✅ Embeddings generated: shape {image_embeds.shape}")
        continue

    # Для decoder
    elif "decoder" in model_name:
        pipe = AutoPipelineForText2Image.from_pretrained(
            local_path,
            torch_dtype=torch.float16,
            local_files_only=True  # Принудительно локальные файлы
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        print(f"✅ Decoder pipeline loaded on {device}")

        try:
            # Генерация изображения
            image = pipe(
                image_embeds=image_embeds,
                negative_image_embeds=negative_image_embeds,
                height=512,
                width=512,
                guidance_scale=7.5,
                num_inference_steps=50
            ).images[0]
            print(f"✅ Image generated successfully")

            # Сохранение
            fname = model_name.replace("/", "_") + "_out.png"
            image.save(fname)
            print(f"✅ Saved image to {fname}")
        except NameError:
            print("❌ Error: Run prior first to get embeddings!")
        except Exception as e:
            print(f"❌ Generation error: {e}")

print("\n🎉 Process complete! Check the saved PNG file.")
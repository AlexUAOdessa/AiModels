import os
import torch
from diffusers import AutoPipelineForText2Image, KandinskyV22PriorPipeline
from PIL import Image

# Для работы этого кода необходимо:
# 1. Установить зависимости:  pip install diffusers transformers accelerate torch pillow
# 2. Клонировать две модели:
#    - Prior: git clone https://huggingface.co/kandinsky-community/kandinsky-2-2-prior
#    - Decoder: git clone https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder
# 3. Убедиться, что Git LFS установлен: git lfs install

# Локальные пути (укажите свои)
model_paths = {
    "kandinsky-community/kandinsky-2-2-decoder": "./kandinsky-2-2-decoder",
    "kandinsky-community/kandinsky-2-2-prior": "./kandinsky-2-2-prior",
}

# Подсказка для генерации
prompt = "A spaceship is flying toward a black hole"
negative_prompt = "low resolution, blurry, bad anatomy"

for model_name, local_path in model_paths.items():
    print(f"\n=== Loading {model_name} from {local_path} ===")

    # Проверяем, что локальная модель существует
    if not os.path.exists(local_path):
        print(f"❌ Local path '{local_path}' not found. Please download the model manually.")
        continue

    # Загружаем prior (если это prior-модель)
    if "prior" in model_name:
        pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
            local_path,
            torch_dtype=torch.float16
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
            torch_dtype=torch.float16
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
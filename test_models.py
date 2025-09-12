import os
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

# Список моделей для сравнения (с Hugging Face git-репозиториев)
model_repos = {
    "stabilityai/stable-diffusion-v1-5": "./stable-diffusion-v1-5",
    "kandinsky-community/kandinsky-2-2-decoder": "./kandinsky-2-2-decoder"
}

# Подсказка для генерации
prompt = "A spaceship is flying toward a black hole"

for repo, local_path in model_repos.items():
    print(f"\n=== Using model {repo} ===")

    # если модель ещё не скачана — клонируем через git-lfs
    if not os.path.exists(local_path):
        print(f"Cloning {repo} into {local_path} ...")
        os.system("git lfs install")
        os.system(f"git clone https://huggingface.co/{repo} {local_path}")
    else:
        print(f"Updating {repo} ...")
        os.system(f"cd {local_path} && git pull && git lfs pull")

    # Загружаем локально
    pipe = AutoPipelineForText2Image.from_pretrained(
        local_path,
        torch_dtype=torch.float16  # экономия VRAM
    )

    # проверяем наличие GPU
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")

    # генерируем картинку
    image = pipe(
        prompt,
        height=512,
        width=512,
        guidance_scale=7.5,
        negative_prompt="low resolution, blurry, bad anatomy"
    ).images[0]

    # сохраняем
    fname = repo.replace("/", "_") + "_out.png"
    image.save(fname)
    print(f"✅ Saved image to {fname}")

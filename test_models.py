# Для работы этого кода необходимо:
# 1. Установить зависимости:  pip install diffusers transformers accelerate torch pillow
# 2. Клонировать две модели:
#    - Prior: git clone https://huggingface.co/kandinsky-community/kandinsky-2-2-prior
#    - Decoder: git clone https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder
# 3. Убедиться, что Git LFS установлен: git lfs install

import torch
from diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_paths = {
    "prior": "./kandinsky-2-2-prior",
    "decoder": "./kandinsky-2-2-decoder",
}

# prompt = "A spaceship is flying toward a black hole"
prompt = "A smartphone with ears is lying on the table by itself"
negative_prompt = "low resolution, blurry, bad anatomy"

# === PRIOR ===
pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
    model_paths["prior"], dtype=dtype
).to(device)

out = pipe_prior(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=25,
    guidance_scale=4.0
)
image_embeds = out.image_embeds
negative_image_embeds = out.negative_image_embeds

# === DECODER ===
pipe = KandinskyV22Pipeline.from_pretrained(
    model_paths["decoder"], dtype=dtype
).to(device)

image = pipe(
    image_embeds=image_embeds,
    negative_image_embeds=negative_image_embeds,
    height=1024,
    width=1024,
    guidance_scale=7.5,
    num_inference_steps=50
).images[0]

image.save("kandinsky_out.png")
print("✅ Saved image to kandinsky_out.png")

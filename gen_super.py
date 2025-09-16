import torch
from diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_paths = {
    "prior": "./kandinsky-2-2-prior",
    "decoder": "./kandinsky-2-2-decoder",
}

prompt = "Internal combustion engine: gasoline, diesel, rotary, and Stirling, cross-sectioned and synchronized by phases."
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
pipe_decoder = KandinskyV22Pipeline.from_pretrained(
    model_paths["decoder"], dtype=dtype
).to(device)

# Создаем папку для результатов
os.makedirs("outputs", exist_ok=True)

num_variants = 3
for i in range(1, num_variants + 1):
    image = pipe_decoder(
        image_embeds=image_embeds,
        negative_image_embeds=negative_image_embeds,
        height=1024,
        width=1024,
        guidance_scale=7.5,
        num_inference_steps=50
    ).images[0]
    
    output_path = f"outputs/kandinsky_out_{i}.png"
    image.save(output_path)
    print(f"✅ Saved image to {output_path}")

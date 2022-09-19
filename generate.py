import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, use_auth_token=True
).to("cuda")

image = pipe("a photograph of an astronaut riding a horse").images[0]
image.save("astronaut.png")

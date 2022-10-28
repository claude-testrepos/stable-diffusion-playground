from functools import lru_cache

import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)

MODELS = {
    "SD 1.5": "runwayml/stable-diffusion-v1-5",
    "SD 2.1": "stabilityai/stable-diffusion-2-1",
}
SCHEDULERS = {
    "DPM++": DPMSolverMultistepScheduler,
    "Euler a": EulerAncestralDiscreteScheduler,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32


@lru_cache(maxsize=1)
def txt2img(model: str):
    pipe = StableDiffusionPipeline.from_pretrained(MODELS[model], torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()  # fits in 6GB VRAM
    return pipe


def img2img(model: str):
    base = txt2img(model)
    return StableDiffusionImg2ImgPipeline(**base.components)


def set_scheduler(pipe, name):
    pipe.scheduler = SCHEDULERS[name].from_config(pipe.scheduler.config)


def generate(prompt, negative, model, scheduler, steps, guidance, seed, init_image=None, strength=0.6):
    pipe = img2img(model) if init_image is not None else txt2img(model)
    set_scheduler(pipe, scheduler)
    generator = torch.Generator(device).manual_seed(int(seed))
    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative or None,
        num_inference_steps=int(steps),
        guidance_scale=guidance,
        generator=generator,
    )
    if init_image is not None:
        kwargs.update(image=init_image.convert("RGB").resize((512, 512)), strength=strength)
    return pipe(**kwargs).images[0]

import random

import gradio as gr

from pipeline import MODELS, SCHEDULERS, generate

EXAMPLES = [
    ["a cozy cabin in a snowy forest at night, warm light, digital painting, artstation", "blurry, low quality"],
    ["portrait of a corgi astronaut, studio lighting, 85mm photo", "deformed"],
    ["isometric pixel art of a tiny coffee shop", ""],
]


def run(prompt, negative, model, scheduler, steps, guidance, seed, init_image, strength):
    seed = int(seed)
    if seed < 0:
        seed = random.randint(0, 2**32 - 1)
    img = generate(prompt, negative, model, scheduler, steps, guidance, seed, init_image, strength)
    return img, f"seed: {seed}"


with gr.Blocks(title="Stable Diffusion Playground") as demo:
    gr.Markdown("# 🎨 Stable Diffusion Playground")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=3)
            negative = gr.Textbox(label="Negative prompt")
            with gr.Row():
                model = gr.Dropdown(list(MODELS), value="SD 1.5", label="Model")
                scheduler = gr.Dropdown(list(SCHEDULERS), value="DPM++", label="Sampler")
            steps = gr.Slider(10, 60, value=25, step=1, label="Steps")
            guidance = gr.Slider(1, 15, value=7.5, step=0.5, label="CFG scale")
            seed = gr.Number(value=-1, label="Seed (-1 = random)")
            with gr.Accordion("img2img", open=False):
                init_image = gr.Image(type="pil", label="Init image")
                strength = gr.Slider(0.1, 1.0, value=0.6, label="Denoising strength")
            button = gr.Button("Generate", variant="primary")
        with gr.Column():
            output = gr.Image(label="Result")
            info = gr.Markdown()
    gr.Examples(EXAMPLES, [prompt, negative])
    button.click(run, [prompt, negative, model, scheduler, steps, guidance, seed, init_image, strength], [output, info])

if __name__ == "__main__":
    demo.queue().launch()

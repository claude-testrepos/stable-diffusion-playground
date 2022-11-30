# Stable Diffusion Playground

Gradio web UI around 🤗 diffusers for my ML course project (Fall 2022). Supports SD 1.5 / 2.1, a couple of samplers, negative prompts and img2img.

## Run

```
pip install -r requirements.txt
huggingface-cli login
python app.py
```

## Notes

Runs on a 6GB GPU with attention slicing. CPU works but takes minutes per image.

## TODO

- [ ] refactor into smaller functions
- [ ] clean up the code

# Low Rank Adaptation for fast Text-To-Image Diffusion Fine-tuning

cloned from: https://github.com/cloneofsimo/lora

Adapted for use with GTX 1660 TI, i7-10750H

## Set up & Use

Requires pip, cuda 11.7 and python >= 3.10

`https://github.com/wchesley/lora.git && git submodule update --init && pip install ./lora && pip install accelerate bitsandbytes`

For training: 
- add at least 5 images to `./images` folder, for better results, use 10-20 images of your desired theme/subject.
- Edits I made to `init_train.sh` avoid OOM while training:
  - turned off mixed precision `--mixed-precision="no"`
  - enabled 8bit Adam `--use_8bit_adam`
  - enabled gradient checkpointing `--gradient_checkpointing`
- run `init_train.sh`
  - This will take `runwayml/stable-diffusion-v1-5` from huggingface and fine-tune the model against your images. You can view more models on [huggingface.co](https://huggingface.co/models?pipeline_tag=text-to-image&sort=downloads). 
- Once finished, your fine-tuned `.safetensors` model will be placed in the `./output` directory. 

Image Creation: 
- Open `interface.py` in text editor
- adjust `prompt`, `negative_prompt`, `model_id`, `lora_weights`, `num_inference_setps`, `guidence_scale` and `tune_lora_scale` as you see fit. What's currently set is relatively sane defaults. 
- in the terminal, run `python interface.py`

## Docs and Examples
See the following repos for more information and examples using LoRA: 
- https://github.com/cloneofsimo/lora
- https://github.com/huggingface/diffusers/blob/main/README.md
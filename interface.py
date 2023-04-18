from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
from lora_diffusion import tune_lora_scale, patch_pipe

# Hugging face model: 
model_id = "runwayml/stable-diffusion-v1-5"

# Location of locally stored lora weights either in lora/example_loras or ./output directories 
lora_weights = "./lora/example_loras/openjourneyLora.safetensors"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    safety_checker=None,
    feature_extractor=None,
    ).to(
    "cuda:0"
)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Various prompts I've used: 
prompt = " painting of pair of Eyes of the Ocean with galaxies set within both eyes, fine detail, high detail, contrast good versus evil, cinematic lighting, vibrant colors, terror, hysterical, horrifying, destruction, the void, octane render, very detailed, trending on artstation, intricate details, high definition, 16k, Dark Souls by WLOP "
#prompt = " dungeons and dragons evil dragon full body side profile portrait, dramatic light, dungeon background, 2 0 0 mm focal length, painted by stanley lau, painted by greg rutkowski, painted by stanley artgerm, digital art, trending on artstation "
#prompt = " a wlop 3 d render of very very very highly detailed beautiful mystic portrait of a skeletal undead knight with whirling galaxy around, tattoos by anton pieck, intricate, vray, extremely detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, intimidating lighting, incredible art "
#prompt = "a wlop painting portrait with very highly detailed dramatic mystic dramatic undead skeleton mage with galactic gemstone eyes, tattoos by anton pieck, intricate, extremely detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, intimidating lighting, incredible art, vray, 4k"
#prompt = " hyperdetailed beautiful androgynous undead skeleton lich made of iridescent metals and shiny rubies, bloody, golden necklace, elite skeleton inspired by ross tran, tattoos by anton pieck, vray, wlop painting, intricate details, highly detailed, octane render, 8k, unreal engine, dnd, digital art by artgerm and greg rutkowski "

# Monkey-patch our lora weights into diffusion model: 
patch_pipe(
    pipe,
    "./lora/example_loras/lora_krk.safetensors",
    patch_text=True,
    patch_ti=True,
    patch_unet=True,
)

tune_lora_scale(pipe.unet, 1.0)
tune_lora_scale(pipe.text_encoder, 1.0)

torch.manual_seed(0)
image = pipe(prompt,
             num_inference_steps=80,
              guidance_scale=6,
              height=640,
              width=512, 
              negative_prompt="Nude, NSFW, Porn, Blurry, distorted, mishapen, bad anatomy, blurry, fuzzy, extra arms, extra fingers, poorly drawn hands").images[0]
image.save("./output/eye.jpg")
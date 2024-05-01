import torch
from diffusers import (
    StableDiffusionXLPipeline, 
    EulerAncestralDiscreteScheduler,
    AutoencoderKL
)

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", 
    torch_dtype=torch.float16
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "animagine-xl-3.0",
    vae=vae,
    torch_dtype=torch.float16, 
    use_safetensors=True, 
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to('cuda')

prompt = "girl drowning in a puddle"
negative_prompt = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name"

results = pipe(
    prompt, 
    negative_prompt=negative_prompt, 
    width=832,
    height=1216,
    guidance_scale=7,
    num_inference_steps=28
)

images = results.images
for idx, image in enumerate(images):
    image.save(f'img/test_{idx}.png')


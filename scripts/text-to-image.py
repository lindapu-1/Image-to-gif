##########################################################################################
# This is a text-to-image generation 
# Change the prompt to generate different images
prompt = "a photo of an astronaut riding a horse on mars"
##########################################################################################
import os

from huggingface_hub import snapshot_download
from diffusers import StableDiffusionPipeline
import torch
from diffusers import StableDiffusionPipeline
import torch
import os

print("Downloading Stable Diffusion Model")
snapshot_download(repo_id="CompVis/stable-diffusion-v1-4", local_dir="./models/StableDiffusion/CompVis/stable-diffusion-v1-4")
print("Downloaded Stable Diffusion Model")

model_path = "./models/StableDiffusion/CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

with torch.autocast("cuda"):
    image = pipe(prompt).images[0]

# Save the image
output_dir = "./"
os.makedirs(output_dir, exist_ok=True)
image_path = os.path.join(output_dir, "generated_image.png")
image.save(image_path)
print(f"Image saved to: {image_path} as generated_image.png")


#python scripts/text-to-image.py

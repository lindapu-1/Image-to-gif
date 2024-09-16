import argparse
from types import SimpleNamespace

#self defined:
prompt=["b&w photo of 42 y.o man in black clothes, bald, face, half body, body, high detailed skin, skin pores, coastline, overcast weather, wind, waves, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"]
n_prompt=["semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"]
steps=25
seed = [8964153601421814582]
guidance_scale=8


parameter= [
        {
            "dreambooth_path": "",
            "lora_model_path": "",
            "inference_config": "configs/inference/inference-v2.yaml",
            "motion_module": "models/Motion_Module/mm_sd_v15_v2.ckpt",
            "seed": seed,
            "steps": steps,
            "guidance_scale":  guidance_scale,
            "prompt": prompt,
            "n_prompt": n_prompt
        }
           ]

# setup configuration
config = {
    "pretrained_model_path": "CompVis/stable-diffusion-v1-4",
    "inference_config": "configs/inference/inference-v1.yaml",
    "config": parameter,
    "L": 16,
    "W": 512,
    "H": 512,
    "without_xformers": False
}

# Create a Namespace object
args = SimpleNamespace(**config)


import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf

import torch
import torchvision.transforms as transforms
import imageio
import cv2

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.models.sparse_controlnet import SparseControlNetModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.util import load_weights, auto_download
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import csv, pdb, glob, math
from pathlib import Path
from PIL import Image
from IPython.display import display, Image as IPImage
import numpy as np


@torch.no_grad()
def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/custom-config-{time_str}"
    os.makedirs(savedir)

    config = args.config  # Now this is directly a list of dictionaries
    samples = []

    # create validation pipeline
    tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
    vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").cuda()

    sample_idx = 0
    for model_idx, model_config in enumerate(config):
        model_config["W"] = model_config.get("W", args.W)
        model_config["H"] = model_config.get("H", args.H)
        model_config["L"] = model_config.get("L", args.L)

        inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))
        unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)).cuda()
        
        # load controlnet model
        controlnet = controlnet_images = None
        
        # set xformers
        if is_xformers_available() and (not args.without_xformers):
            unet.enable_xformers_memory_efficient_attention()
            if controlnet is not None: controlnet.enable_xformers_memory_efficient_attention()

        pipeline = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            controlnet=controlnet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        ).to("cuda")

        pipeline = load_weights(
            pipeline,
            # motion module
            motion_module_path         = model_config.get("motion_module", ""),
            motion_module_lora_configs = model_config.get("motion_module_lora_configs", []),
            # domain adapter
            adapter_lora_path          = model_config.get("adapter_lora_path", ""),
            adapter_lora_scale         = model_config.get("adapter_lora_scale", 1.0),
            # image layers
            dreambooth_model_path      = model_config.get("dreambooth_path", ""),
            lora_model_path            = model_config.get("lora_model_path", ""),
            lora_alpha                 = model_config.get("lora_alpha", 0.8),
        ).to("cuda")

        prompts = model_config["prompt"]
        n_prompts = model_config["n_prompt"]
        
        random_seeds = model_config.get("seed", [-1])
        random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
        random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
        
        config[model_idx]['random_seed'] = []
        for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):
            
            # manually set random seed for reproduction
            if random_seed != -1: torch.manual_seed(random_seed)
            else: torch.seed()
            config[model_idx]['random_seed'].append(torch.initial_seed())
            
            print(f"current seed: {torch.initial_seed()}")
            print(f"sampling {prompt} ...")
            sample = pipeline(
                prompt,
                negative_prompt     = n_prompt,
                num_inference_steps = model_config["steps"],
                guidance_scale = model_config["guidance_scale"],
                width = model_config["W"],
                height = model_config["H"],
                video_length = model_config["L"],

                controlnet_images = controlnet_images,
                controlnet_image_index = model_config.get("controlnet_image_indexs", [0]),
            ).videos
            samples.append(sample)

            prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
            save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{prompt}.mp4")
            print(f"save to {savedir}/sample/{prompt}.mp4")
            
            sample_idx += 1

    samples = torch.concat(samples)
    final_gif_path = f"{savedir}/sample.gif"
    save_videos_grid(samples, final_gif_path, n_rows=4)

    print(f"Final GIF saved to: {final_gif_path}")


if __name__ == "__main__":
    main(args)


#python scripts/my.py
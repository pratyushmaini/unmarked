# methods/my_watermarking.py

from watermarked_diffusion_pipeline import BaseWatermarkedDiffusionPipeline
import numpy as np
from scipy.stats import norm
import scipy
import torch
import torch.optim as optim
from ZoDiac.main.wmdiffusion import WMDetectStableDiffusionPipeline
from ZoDiac.main.wmpatch import GTWatermark, GTWatermarkMulti, KeyedGTWatermark
from ZoDiac.main.utils import *
from ZoDiac.loss.loss import LossProvider
from ZoDiac.loss.pytorch_ssim import ssim

import torch
import torchvision.transforms as transforms
from diffusers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor

import os


class MyWatermarkedPipeline(BaseWatermarkedDiffusionPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intermediate_name = "intermediate.tiff"
        self.save_path = "methods/output"
        self.zodiac_cfgs = {
            # general
            "method": 'ZoDiac',
            "save_img": 'methods/output/' ,

            # for stable diffusion
            "model_id": 'stabilityai/stable-diffusion-2-1-base',
            "gen_seed": 0,  # the seed for generating gt image; no use for watermarking existing imgs
            "empty_prompt": True, # whether to use the caption of the image

            # for watermark
            "w_type": 'single', # single or multi
            "w_channel": 3,
            "w_radius": 10,
            "w_seed": 10,

            # for updating
            "start_latents": 'init_w', # 'init', 'init_w', 'rand', 'rand_w'
            "iters": 10,
            "save_iters": [10],
            "loss_weights": [10.0, 0.1, 1.0, 0.0], # L2 loss, watson-vgg loss, SSIM loss, watermark L1 loss

            # for postprocessing and detection
            "ssim_threshold": 0.92,
            "detect_threshold": 0.9,
        }
        self.device_obj = torch.device(self.device)

    def generate(self, prompt, key):
        # Your watermark embedding logic here
        print("Step 1: Generate image from prompt with SDXL")
        self.generate_image(prompt)
        torch.cuda.empty_cache()

        print("Step 2: Embed key in image with ZoDiac method")
        print("Step 2a: Load watermark and diffusion pipeline")
        self.wm_pipe = KeyedGTWatermark(self.device_obj, key=key, w_channel=self.zodiac_cfgs['w_channel'], w_radius=self.zodiac_cfgs['w_radius'], generator=torch.Generator(self.device_obj).manual_seed(self.zodiac_cfgs['w_seed']))
        scheduler = DDIMScheduler.from_pretrained(self.zodiac_cfgs['model_id'], subfolder="scheduler")
        self.wm_sd_pipe = WMDetectStableDiffusionPipeline.from_pretrained(self.zodiac_cfgs['model_id'], scheduler=scheduler).to(torch.device('cuda'))

        print("Step 2b: Load generated image")
        imagename = self.intermediate_name
        gt_img_tensor = get_img_tensor(f'{self.save_path}/{imagename}', self.device_obj)

        print("Step 2c: Get init noise")    
        def get_init_latent(img_tensor, pipe, text_embeddings, guidance_scale=1.0):
            # DDIM inversion from the given image
            img_latents = pipe.get_image_latents(img_tensor, sample=False)
            reversed_latents = pipe.forward_diffusion(
                latents=img_latents,
                text_embeddings=text_embeddings,
                guidance_scale=guidance_scale,
                num_inference_steps=50,
            )
            return reversed_latents

        empty_text_embeddings = self.wm_sd_pipe.get_text_embedding('')
        init_latents_approx = get_init_latent(gt_img_tensor, self.wm_sd_pipe, empty_text_embeddings)

        print("Step 2d: Prepare training")  
        init_latents = init_latents_approx.detach().clone()
        init_latents.requires_grad = True
        optimizer = optim.Adam([init_latents], lr=0.01)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.3) 

        totalLoss = LossProvider(self.zodiac_cfgs['loss_weights'], self.device_obj)
        loss_lst = [] 

        print("Step 2e: Train init latents") 
        for i in range(self.zodiac_cfgs['iters']):
            print(f"iter {i}:")
            init_latents_wm = self.wm_pipe.inject_watermark(init_latents)
            if self.zodiac_cfgs['empty_prompt']:
                pred_img_tensor = self.wm_sd_pipe('', guidance_scale=1.0, num_inference_steps=50, output_type='tensor', use_trainable_latents=True, init_latents=init_latents_wm).images
            else:
                pred_img_tensor = self.wm_sd_pipe(prompt, num_inference_steps=50, output_type='tensor', use_trainable_latents=True, init_latents=init_latents_wm).images
            loss = totalLoss(pred_img_tensor, gt_img_tensor, init_latents_wm, self.wm_pipe)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_lst.append(loss.item())
        torch.cuda.empty_cache()
        
        save_img(f"{self.save_path}/output.tiff", pred_img_tensor, self.wm_sd_pipe)
        pil_img = pipe.numpy_to_pil(self.wm_sd_pipe.img_tensor_to_numpy(pred_img_tensor))[0]

        return pil_img

    def detect(self, image):
        # Your watermark detection logic here
        text_embeddings = self.wm_sd_pipe.get_text_embedding("")
        results = detect_keyed_watermark(
            image,
            self.wm_sd_pipe,
            self.wm_pipe,
            text_embeddings,
            device=self.device
        )
        print(results)
        return results["detected_key"]
    
    def generate_image(self, prompt, **generate_kwargs) -> None:
        if not os.path.exists("./methods/output"):
            os.mkdir("./methods/output")
        image = self.model(
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=0,
            **generate_kwargs
        ).images[0]

        image.save(f"{self.save_path}/{self.intermediate_name}")

if __name__ == "__main__":
    pipe = MyWatermarkedPipeline()
    pipe.generate("a pokemon", 9)
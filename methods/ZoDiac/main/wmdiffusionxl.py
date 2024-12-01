from typing import Callable, List, Optional, Union, Any, Dict
from functools import partial
import numpy as np
import copy
from dataclasses import dataclass
import PIL
import torch
from torch.utils.checkpoint import checkpoint
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import BaseOutput

@dataclass
class ModifiedStableDiffusionXLPipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]
    init_latents: Optional[torch.FloatTensor]

class WatermarkStableDiffusionXLPipeline(StableDiffusionXLPipeline):
    def __init__(
        self,
        vae,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        unet,
        scheduler,
        # requires_safety_checker: bool = True
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            # requires_safety_checker=requires_safety_checker
        )
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        components = StableDiffusionXLPipeline.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        ).components
        
        # Remove unwanted components
        components.pop("image_encoder", None)
        components.pop("feature_extractor", None)
        components.pop("force_zeros_for_empty_prompt", None)
        
        return cls(**components)

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
    ):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # Encode text
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_embeds = self.text_encoder(text_input_ids)[0]

        # Encode text 2
        text_inputs_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs_2.input_ids.to(device)
        prompt_embeds_2 = self.text_encoder_2(text_input_ids_2)[0]
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)

        # Duplicate for classifier-free guidance
        if do_classifier_free_guidance:
            neg_text_inputs = self.tokenizer(
                negative_prompt or "",
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            neg_input_ids = neg_text_inputs.input_ids.to(device)
            negative_prompt_embeds = self.text_encoder(neg_input_ids)[0]

            neg_text_inputs_2 = self.tokenizer_2(
                negative_prompt or "",
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            neg_input_ids_2 = neg_text_inputs_2.input_ids.to(device)
            negative_prompt_embeds_2 = self.text_encoder_2(neg_input_ids_2)[0]
            negative_prompt_embeds = torch.cat(
                [negative_prompt_embeds, negative_prompt_embeds_2], dim=-1
            )

            # Duplicate for batch size
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds.repeat(num_images_per_prompt, 1, 1)

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        num_inference_steps: int = 4,
        guidance_scale: float = 0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        use_trainable_latents: bool = False,
        init_latents: Optional[torch.FloatTensor] = None,
    ):
        # 1. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Set device
        device = self._execution_device

        # 3. Encode input prompt
        prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            guidance_scale > 1.0,
            negative_prompt,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        latents_shape = (
            batch_size * num_images_per_prompt,
            self.unet.config.in_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if not use_trainable_latents:
            if latents is None:
                latents = torch.randn(
                    latents_shape,
                    generator=generator,
                    device=device,
                    dtype=prompt_embeds.dtype
                )
            init_latents = copy.deepcopy(latents)
        else:
            if init_latents is None:
                raise ValueError("Initial trainable latents must be provided when use_trainable_latents is True")
            latents = init_latents

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Scale input if needed
                latent_model_input = self.scheduler.scale_model_input(latents, t)

                # Predict noise
                if not use_trainable_latents:
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
                else:
                    noise_pred = checkpoint(
                        self.unet_custom_forward,
                        latent_model_input,
                        t,
                        prompt_embeds,
                        cross_attention_kwargs
                    ).sample

                # Compute previous sample
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                progress_bar.update()

        # 8. Post-processing
        if output_type == "latent":
            return ModifiedStableDiffusionXLPipelineOutput(
                images=latents,
                nsfw_content_detected=None,
                init_latents=init_latents
            )

        # Decode latents
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)

        # Convert to output format
        if output_type == "pil":
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = self.numpy_to_pil(image)
        
        return ModifiedStableDiffusionXLPipelineOutput(
            images=image,
            nsfw_content_detected=None,
            init_latents=init_latents
        )

class WMDetectStableDiffusionXLPipeline(WatermarkStableDiffusionXLPipeline):
    def __init__(
        self,
        vae,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        unet,
        scheduler,
        requires_safety_checker: bool = False
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            # requires_safety_checker=requires_safety_checker
        )
        self.forward_diffusion = partial(self.backward_diffusion, reverse_process=True)

    @torch.inference_mode()
    def get_text_embedding(self, prompt):
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        prompt_embeds = self.text_encoder(text_input_ids, output_hidden_states=True).hidden_states[-2]

        text_inputs_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs_2.input_ids.to(self.device)
        prompt_embeds_2 = self.text_encoder_2(text_input_ids_2, output_hidden_states=True).hidden_states[-2]
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
        return prompt_embeds

    @torch.inference_mode()
    def get_image_latents(self, image: torch.Tensor, sample=True, rng_generator=None):
        image = 2.0 * image - 1.0
        encoding_dist = self.vae.encode(image).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = encoding * self.vae.config.scaling_factor
        return latents

    def backward_ddim(self, x_t, alpha_t, alpha_tm1, eps_xt):
        return (
            alpha_tm1**0.5
            * (
                (alpha_t**-0.5 - alpha_tm1**-0.5) * x_t
                + ((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt
            )
            + x_t
        )
    
    def prepare_time_ids(self, original_size, crops_coords_top_left, target_size):
        # SDXL expects these additional inputs
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(self.device, dtype=self.text_encoder_2.dtype)
        return add_time_ids

    @torch.inference_mode()
    def backward_diffusion(
        self,
        use_old_emb_i=25,
        text_embeddings=None,
        old_text_embeddings=None,
        new_text_embeddings=None,
        latents: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 4,
        guidance_scale: float = 0,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        reverse_process: bool = False,
        **kwargs,
    ):
        # original_size = (1024, 1024)
        # target_size = (1024, 1024)
        # crops_coords_top_left = (0, 0)
        # time_ids = self.prepare_time_ids(original_size, target_size, crops_coords_top_left)
        # time_ids = time_ids.to(self.device)
        
        # # Get text embeddings for pooled output
        # text_input_ids_2 = self.tokenizer_2(
        #     "",  # empty string for unconditional
        #     padding="max_length",
        #     max_length=self.tokenizer_2.model_max_length,
        #     truncation=True,
        #     return_tensors="pt",
        # ).input_ids.to(self.device)
        # pooled_text_embeddings = self.text_encoder_2(text_input_ids_2)[1]
        
        # added_cond_kwargs = {
        #     "text_embeds": pooled_text_embeddings,
        #     "time_ids": time_ids
        # }
        do_classifier_free_guidance = guidance_scale > 0
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        latents = latents * self.scheduler.init_noise_sigma

        prompt_to_prompt = old_text_embeddings is not None and new_text_embeddings is not None

        for i, t in enumerate(self.progress_bar(timesteps_tensor if not reverse_process else reversed(timesteps_tensor))):
            if prompt_to_prompt:
                text_embeddings = old_text_embeddings if i < use_old_emb_i else new_text_embeddings

            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # NOTE: fixing unet error
            self.unet.config.addition_embed_type = None 
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=kwargs.get("cross_attention_kwargs"),
            ).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
            if reverse_process:
                alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
            latents = self.backward_ddim(
                x_t=latents,
                alpha_t=alpha_prod_t,
                alpha_tm1=alpha_prod_t_prev,
                eps_xt=noise_pred,
            )
        # return latents
            # modify the process for Euler step
            # if reverse_process:
            #     # For reverse process, modify the step calculation
            #     step_output = self.scheduler.step(
            #         noise_pred,
            #         t,
            #         latents,
            #         reverse=True
            #     )
            # else:
            #     step_output = self.scheduler.step(
            #         noise_pred,
            #         t,
            #         latents
            #     )
            # latents = step_output.prev_sample
        return latents
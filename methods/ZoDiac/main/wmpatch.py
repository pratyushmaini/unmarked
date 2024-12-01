import numpy as np
from scipy.stats import norm
import scipy
import torch
from diffusers.utils.torch_utils import randn_tensor

class GTWatermark():
    def __init__(self, device, shape=(1,4,64,64), dtype=torch.float32, w_channel=3, w_radius=10, generator=None):
        self.device = device
        # from latent tensor
        self.shape = shape
        self.dtype = dtype
        # from hyperparameters
        self.w_channel = w_channel
        self.w_radius = w_radius

        self.gt_patch, self.watermarking_mask = self._gen_gt(generator=generator)
        self.mu, self.sigma = self.watermark_stat()

    def _circle_mask(self, size=64, r=10, x_offset=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
        x0 = y0 = size // 2
        x0 += x_offset
        y0 += y_offset
        y, x = np.ogrid[:size, :size]
        y = y[::-1]
        return ((x - x0)**2 + (y-y0)**2)<= r**2

    def _get_watermarking_pattern(self, gt_init): # in fft space
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        for i in range(self.w_radius, 0, -1): # from outer circle to inner circle
            tmp_mask = torch.tensor(self._circle_mask(gt_init.shape[-1], r=i)).to(self.device) # circle mask in bool value
            gt_patch[:, self.w_channel, tmp_mask] = gt_patch[0, self.w_channel, 0, i].item() # set the value inside the circle to be a value from Gaussian Distribution
        return gt_patch

    def _get_watermarking_mask(self, gt_patch):
        watermarking_mask = torch.zeros(gt_patch.shape, dtype=torch.bool).to(self.device)
        watermarking_mask[:,self.w_channel] = torch.tensor(self._circle_mask(gt_patch.shape[-1], r=self.w_radius)).to(self.device)
        return watermarking_mask

    def _gen_gt(self, generator=None):
        gt_init = randn_tensor(self.shape, generator=generator, device=self.device, dtype=self.dtype)
        gt_patch = self._get_watermarking_pattern(gt_init)
        watermarking_mask = self._get_watermarking_mask(gt_patch)
        return gt_patch, watermarking_mask

    def inject_watermark(self, latents): 
        latents_fft = torch.fft.fftshift(torch.fft.fft2(latents), dim=(-1, -2))
        # latents_fft[self.watermarking_mask] = self.gt_patch[self.watermarking_mask].clone()
        latents_fft = latents_fft * ~(self.watermarking_mask) + self.gt_patch * self.watermarking_mask
        latents_w = torch.fft.ifft2(torch.fft.ifftshift(latents_fft, dim=(-1, -2))).real
        return latents_w

    def eval_watermark(self, latents_w):
        latents_w_fft = torch.fft.fftshift(torch.fft.fft2(latents_w), dim=(-1, -2))
        l1_metric = torch.abs(latents_w_fft[self.watermarking_mask] - self.gt_patch[self.watermarking_mask]).mean().item()
        return l1_metric

    def watermark_stat(self):
        dis_all = []
        for i in range(1000):
            rand_latents = randn_tensor(self.shape, device=self.device, dtype=self.dtype)
            dis = self.eval_watermark(rand_latents)
            dis_all.append(dis)
        dis_all = np.array(dis_all)
        return dis_all.mean(), dis_all.var()

    # the probability of being watermarked
    def one_minus_p_value(self, latents):
        l1_metric = self.eval_watermark(latents)
        return abs(0.5 - norm.cdf(l1_metric, self.mu, self.sigma))*2
    
    def tree_ring_p_value(self, latents):
        target_patch = self.gt_patch[self.watermarking_mask].flatten()
        target_patch = torch.concatenate([target_patch.real, target_patch.imag])

        reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(latents), dim=(-1, -2))[self.watermarking_mask].flatten()
        reversed_latents_w_fft = torch.concatenate([reversed_latents_w_fft.real, reversed_latents_w_fft.imag])
        
        sigma_w = reversed_latents_w_fft.std()
        lambda_w = (target_patch ** 2 / sigma_w ** 2).sum().item()
        x_w = (((reversed_latents_w_fft - target_patch) / sigma_w) ** 2).sum().item()
        p_w = scipy.stats.ncx2.cdf(x=x_w, df=len(target_patch), nc=lambda_w)
        return p_w
        

class GTWatermarkMulti(GTWatermark):
    def __init__(self, device, shape=(1,4,64,64), dtype=torch.float32, w_settings={0:[1,5,9], 1:[2,6,10], 2:[3,7], 3:[4,8]}, generator=None):
        self.device = device
        # from latent tensor
        self.shape = shape
        self.dtype = dtype
        # from hyperparameters
        self.w_settings = w_settings

        self.gt_patch, self.watermarking_mask = self._gen_gt(generator=generator)
        self.mu, self.sigma = self.watermark_stat()

    def _get_watermarking_pattern(self, gt_init): # in fft space
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        watermarking_mask = torch.zeros(gt_init.shape, dtype=torch.bool).to(self.device)
        for w_channel in self.w_settings:
            for w_radius in self.w_settings[w_channel]:
                tmp_mask_alter, tmp_mask_inner = self._circle_mask(gt_init.shape[-1], r=w_radius), self._circle_mask(gt_init.shape[-1], r=w_radius-1) 
                tmp_mask = torch.tensor(np.logical_xor(tmp_mask_alter,tmp_mask_inner)).to(self.device) 
                gt_patch[:, w_channel, tmp_mask] = gt_patch[0, w_channel, 0, w_radius].item()
                watermarking_mask[:, w_channel, tmp_mask] = True
        return gt_patch, watermarking_mask

    def _gen_gt(self, generator=None):
        gt_init = randn_tensor(self.shape, generator=generator, device=self.device, dtype=self.dtype)
        gt_patch, watermarking_mask = self._get_watermarking_pattern(gt_init)
        return gt_patch, watermarking_mask

    # def eval_watermark(self, latents_w):
    #     latents_w_fft = torch.fft.fftshift(torch.fft.fft2(latents_w), dim=(-1, -2))
    #     l1_tensor = torch.abs(latents_w_fft[self.watermarking_mask] - self.gt_patch[self.watermarking_mask])
        
    #     # num_samples = len(l1_tensor) // 2 
    #     num_samples = 400
    #     indices = torch.randint(0, len(l1_tensor), (num_samples,), generator=torch.Generator().manual_seed(0))
    #     sampled_elements = l1_tensor[indices]
    #     l1_metric = sampled_elements.mean().item()
    #     return l1_metric

class KeyedGTWatermark(GTWatermark):
    def __init__(self, device, key, shape=(1,4,64,64), dtype=torch.float32, w_channel=3, w_radius=10, generator=None):
        super().__init__(device, shape, dtype, w_channel, w_radius, generator)
        self.key = key
        self.ring_stats = self._compute_ring_statistics()

    def inject_watermark(self, latents):
        """Override parent's inject_watermark to include key encoding"""
        latents_fft = torch.fft.fftshift(torch.fft.fft2(latents), dim=(-1, -2))
        
        # Convert key to binary and pad to w_radius length
        key_bits = format(self.key, 'b').zfill(self.w_radius)
        
        # Apply key-based watermark
        for i, bit in enumerate(key_bits):
            radius = i + 1
            ring_mask = self._get_ring_mask(latents_fft.shape[-1], radius)
            ring_mask = ring_mask.unsqueeze(0).unsqueeze(0)
            ring_mask = ring_mask.expand(latents_fft.shape[0], 1, -1, -1)
            
            # Select and modify the watermark channel
            modifier = 2.0 if bit == '1' else 0.5
            channel_mask = torch.zeros_like(latents_fft, dtype=torch.bool)
            channel_mask[:, self.w_channel:self.w_channel+1] = ring_mask
            print(channel_mask)
            
            # Apply modification
            latents_fft = torch.where(channel_mask, 
                                    latents_fft * modifier, 
                                    latents_fft)
        
        # Convert back to spatial domain
        latents_w = torch.fft.ifft2(torch.fft.ifftshift(latents_fft, dim=(-1, -2))).real
        return latents_w

    def _compute_ring_statistics(self):
        """Compute statistical thresholds for each ring"""
        ring_stats = []
        num_samples = 1000
        
        for i in range(self.w_radius):
            ring_amplitudes = []
            for _ in range(num_samples):
                rand_latents = randn_tensor(self.shape, device=self.device, dtype=self.dtype)
                rand_fft = torch.fft.fftshift(torch.fft.fft2(rand_latents), dim=(-1, -2))
                
                radius = i + 1
                ring_mask = self._get_ring_mask(rand_fft.shape[-1], radius)
                ring_amp = torch.abs(rand_fft[:, self.w_channel, ring_mask]).mean().item()
                ring_amplitudes.append(ring_amp)
            
            stats = {
                'mean': np.mean(ring_amplitudes),
                'std': np.std(ring_amplitudes),
                'threshold': np.mean(ring_amplitudes) * 1.5
            }
            ring_stats.append(stats)
        print(ring_stats)
        return ring_stats

    def _get_ring_mask(self, size, radius):
        """Get mask for a specific ring"""
        outer = self._circle_mask(size, r=radius)
        inner = self._circle_mask(size, r=radius-1)
        return torch.tensor(np.logical_xor(outer, inner)).to(self.device)

    def _detect_key(self, latents):
        latents_fft = torch.fft.fftshift(torch.fft.fft2(latents), dim=(-1, -2))
        detected_bits = []
        confidence_scores = []

        for i in range(self.w_radius):
            radius = i + 1
            ring_mask = self._get_ring_mask(latents_fft.shape[-1], radius)
            
            # Expand mask to match tensor dimensions
            ring_mask = ring_mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            ring_mask = ring_mask.expand(latents_fft.shape[0], 1, -1, -1)
            
            # Select the watermark channel and apply mask
            channel_data = latents_fft[:, self.w_channel:self.w_channel+1]
            masked_data = channel_data[ring_mask]

            print(masked_data)
            
            # Calculate amplitude
            ring_amplitude = torch.abs(masked_data).mean().item()
            normalized_amp = (ring_amplitude - self.ring_stats[i]['mean']) / self.ring_stats[i]['std']
            print(ring_amplitude, normalized_amp)
            
            bit = '1' if normalized_amp > 1.5 else '0'
            confidence = abs(normalized_amp)
            
            detected_bits.append(bit)
            confidence_scores.append(confidence)

        # Apply confidence threshold
        reliable_bits = [b if c > 2.0 else '0' for b, c in zip(detected_bits, confidence_scores)]
        return int(''.join(reliable_bits), 2)

    def evaluate_key_detection(self, latents):
        """Evaluate key detection with confidence metrics"""
        detected_key = self._detect_key(latents)
        watermark_prob = 1 - self.tree_ring_p_value(latents)
        
        # Compare with original key
        original_bits = format(self.key, 'b').zfill(self.w_radius)
        detected_bits = format(detected_key, 'b').zfill(self.w_radius)
        bit_accuracy = sum(a == b for a, b in zip(original_bits, detected_bits)) / len(original_bits)
        
        return {
            'detected_key': detected_key,
            'watermark_probability': watermark_prob,
            'bit_accuracy': bit_accuracy,
            'confidence': bit_accuracy * watermark_prob
        }
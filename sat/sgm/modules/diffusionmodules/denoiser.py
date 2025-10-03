from typing import Dict, Union

import torch
import torch.nn as nn

from ...util import append_dims, instantiate_from_config


class Denoiser(nn.Module):
    def __init__(self, weighting_config, scaling_config):
        super().__init__()

        self.weighting = instantiate_from_config(weighting_config)
        self.scaling = instantiate_from_config(scaling_config)

    def possibly_quantize_sigma(self, sigma):
        return sigma

    def possibly_quantize_c_noise(self, c_noise):
        return c_noise

    def w(self, sigma):
        return self.weighting(sigma)

    def forward(
        self,
        network: nn.Module, #查看是什么
        input: torch.Tensor,
        input_reference: torch.Tensor,
        sigma: torch.Tensor,
        cond: Dict,
        **additional_model_inputs, #看看里面的concat_images
    ) -> torch.Tensor:
        # denoised = denoiser( #Tora/modules/SwissArmyTransformer/sat/model/transformer.py  P571
        #        *self.guider.prepare_inputs(x, alpha_cumprod_sqrt, cond, uc), **additional_model_inputs
        #    ).to(torch.float32)
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma, **additional_model_inputs) 
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))

        #sigma_reference = self.possibly_quantize_sigma(sigma_reference)
        #sigma_shape_reference = sigma_reference.shape
        #sigma_reference = append_dims(sigma_reference, input_reference.ndim)
        #c_skip_reference, c_out_reference, c_in_reference, c_noise_reference = self.scaling(sigma_reference, **additional_model_inputs)
        #c_noise_reference = self.possibly_quantize_c_noise(c_noise_reference.reshape(sigma_shape_reference))
        tem = network(input * c_in, input_reference * c_in, c_noise, cond, **additional_model_inputs)
        #return network(input * c_in, input_reference * c_in, c_noise, cond, **additional_model_inputs) * c_out + input * c_skip #以及input_reference
        return tem[0] * c_out + input * c_skip, tem[1] * c_out + input_reference * c_skip
        #/sat/sgm/modules/diffusionmodules/wrappers.py P35

class DiscreteDenoiser(Denoiser):
    def __init__(
        self,
        weighting_config,
        scaling_config,
        num_idx,
        discretization_config,
        do_append_zero=False,
        quantize_c_noise=True,
        flip=True,
    ):
        super().__init__(weighting_config, scaling_config)
        sigmas = instantiate_from_config(discretization_config)(num_idx, do_append_zero=do_append_zero, flip=flip)
        self.sigmas = sigmas
        # self.register_buffer("sigmas", sigmas)
        self.quantize_c_noise = quantize_c_noise

    def sigma_to_idx(self, sigma):
        dists = sigma - self.sigmas.to(sigma.device)[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx):
        return self.sigmas.to(idx.device)[idx]

    def possibly_quantize_sigma(self, sigma):
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise):
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise

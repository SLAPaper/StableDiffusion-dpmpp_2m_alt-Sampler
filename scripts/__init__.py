# Copyright 2024 SLAPaper
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import k_diffusion.sampling
import torch


@torch.no_grad()
def sample_dpmpp_2m_alt(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M) alternative sampler
    Source: https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/8457
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in k_diffusion.sampling.trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        sigma_progress = i / len(sigmas)
        adjustment_factor = 1 + (0.15 * (sigma_progress * sigma_progress))
        old_denoised = denoised * adjustment_factor
    return x


def add_sample_dpmpp_2m_alt_webui() -> None:
    """Adds DPM-Solver++(2M) alternative sampler to the list of available samplers."""
    try:
        from modules import (  # type: ignore
            sd_samplers,
            sd_samplers_common,
            sd_samplers_kdiffusion,
        )
    except ImportError:
        return

    samplers_dpmpp_2m_alt = [
        (
            "DPM++ 2M alt",
            sample_dpmpp_2m_alt,
            ["k_dpmpp_2m_alt"],
            {"scheduler": "karras"},
        )
    ]
    samplers_data_dpmpp_2m_alt = [
        sd_samplers_common.SamplerData(
            label,
            lambda model, funcname=funcname: sd_samplers_kdiffusion.KDiffusionSampler(
                funcname, model
            ),
            aliases,
            options,
        )
        for label, funcname, aliases, options in samplers_dpmpp_2m_alt
    ]

    sd_samplers.all_samplers.extend(samplers_data_dpmpp_2m_alt)
    for x in samplers_data_dpmpp_2m_alt:
        sd_samplers.all_samplers_map[x.name] = x

    sd_samplers.set_samplers()


add_sample_dpmpp_2m_alt_webui()

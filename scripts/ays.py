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


import logging

import numpy as np
import torch


def loglinear_interp(t_steps: list[float], num_steps: int) -> np.ndarray:
    """
    Performs log-linear interpolation of a given array of decreasing numbers.
    """
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])

    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)

    interped_ys = np.exp(new_ys)[::-1].copy()
    return interped_ys


def align_your_step_scheduler_v15(
    n: int, sigma_min: float, sigma_max: float, device
) -> torch.Tensor:
    """SD15 AYS scheduler from https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html"""
    NOISE_LEVELS = [
        14.615,
        6.475,
        3.861,
        2.697,
        1.886,
        1.396,
        0.963,
        0.652,
        0.399,
        0.152,
        0.029,
    ]
    TIMESTEP_INDICES = [999, 850, 736, 645, 545, 455, 343, 233, 124, 24, 0]
    sigs = [sigma for sigma in loglinear_interp(NOISE_LEVELS, n)]
    sigs.append(0.0)

    logging.info(f"AYS scheduler: {sigs=}")
    return torch.FloatTensor(sigs).to(device)


def align_your_step_scheduler_xl(
    n: int, sigma_min: float, sigma_max: float, device
) -> torch.Tensor:
    """SDXL AYS scheduler from https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html"""
    NOISE_LEVELS = [
        14.615,
        6.315,
        3.771,
        2.181,
        1.342,
        0.862,
        0.555,
        0.380,
        0.234,
        0.113,
        0.029,
    ]
    TIMESTEP_INDICES = [999, 845, 730, 587, 443, 310, 193, 116, 53, 13, 0]
    sigs = [sigma for sigma in loglinear_interp(NOISE_LEVELS, n)]
    sigs.append(0.0)

    logging.info(f"AYS scheduler: {sigs=}")
    return torch.FloatTensor(sigs).to(device)


def add_align_your_step_scheduler() -> None:
    """Add AYS scheduler to the list of schedulers"""
    try:
        from modules import sd_schedulers  # type: ignore
    except ImportError:
        return

    # latest dev webui have already added these scheduler: https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/15751
    if 'align_your_steps' in sd_schedulers.schedulers_map:
        return

    scheduler_v15 = sd_schedulers.Scheduler(
        "ays_v15", "Align Your Step SD15", align_your_step_scheduler_v15
    )
    scheduler_xl = sd_schedulers.Scheduler(
        "ays_xl", "Align Your Step SDXL", align_your_step_scheduler_xl
    )

    sd_schedulers.schedulers.append(scheduler_v15)
    sd_schedulers.schedulers.append(scheduler_xl)

    sd_schedulers.schedulers_map[scheduler_v15.name] = scheduler_v15
    sd_schedulers.schedulers_map[scheduler_v15.label] = scheduler_v15

    sd_schedulers.schedulers_map[scheduler_xl.name] = scheduler_xl
    sd_schedulers.schedulers_map[scheduler_xl.label] = scheduler_xl


add_align_your_step_scheduler()

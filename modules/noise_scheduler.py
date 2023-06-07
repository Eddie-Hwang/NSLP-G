import math


def linear_noise_scheduler(initial_noise_std, final_noise_std, total_steps, current_step, **kwargs):
    progress = (current_step + 1) / (total_steps + 1)
    current_noise_std = initial_noise_std - progress * (initial_noise_std - final_noise_std)
    return current_noise_std


def exponential_noise_scheduler(initial_noise_std, final_noise_std, total_steps, current_step, sharpness_factor=1., **kwargs):
    decay_rate = (final_noise_std / initial_noise_std) ** (1.0 / total_steps)
    current_noise_std = initial_noise_std * (decay_rate ** (current_step * sharpness_factor))
    return current_noise_std


def cosine_annealing_noise_scheduler(initial_noise_std, final_noise_std, total_steps, current_step, sharpness_factor=1., **kwargs):
    progress = (current_step + 1) / (total_steps + 1)
    cosine_term = (1 + math.cos(sharpness_factor * math.pi * progress)) / 2
    current_noise_std = final_noise_std + (initial_noise_std - final_noise_std) * cosine_term
    return current_noise_std


def ConstantNoiseScheduler(initial_noise_std, **kwargs):
    return initial_noise_std


noise_scheduler_dict = {
    "linear": linear_noise_scheduler,
    "exp": exponential_noise_scheduler,
    "cosine_annealing": cosine_annealing_noise_scheduler,
    "constant": ConstantNoiseScheduler,
}



class LinearNoiseScheduler:
    def __init__(self, initial_noise_std, final_noise_std, total_steps, **kwargs):
        self.initial_noise_std = initial_noise_std
        self.final_noise_std = final_noise_std
        self.total_steps = total_steps + 1

    def step(self, current_step):
        progress = (current_step + 1) / self.total_steps
        current_noise_std = self.initial_noise_std - progress * (self.initial_noise_std - self.final_noise_std)
        return current_noise_std



class ExponentialNoiseScheduler:
    def __init__(self, initial_noise_std, final_noise_std, total_steps, sharpness_factor=1., **kwargs):
        self.initial_noise_std = initial_noise_std
        self.final_noise_std = final_noise_std
        self.total_steps = total_steps + 1
        self.sharpness_factor = sharpness_factor
        # self.decay_rate = (final_noise_std / initial_noise_std) ** (1.0 / (total_steps * sharpness_factor))
        self.decay_rate = (final_noise_std / initial_noise_std) ** (1.0 / total_steps)

    def step(self, current_step):
        current_noise_std = self.initial_noise_std * (self.decay_rate ** (current_step * self.sharpness_factor))
        return current_noise_std


class CosineAnnealingNoiseScheduler:
    def __init__(self, initial_noise_std, final_noise_std, total_steps, sharpness_factor=1., **kwargs):
        self.initial_noise_std = initial_noise_std
        self.final_noise_std = final_noise_std
        self.total_steps = total_steps + 1
        self.sharpness_factor = sharpness_factor

    def step(self, current_step):
        progress = (current_step + 1) / self.total_steps
        cosine_term = (1 + math.cos(self.sharpness_factor * math.pi * progress)) / 2
        current_noise_std = self.final_noise_std + (self.initial_noise_std - self.final_noise_std) * cosine_term
        return current_noise_std
    

class ConstantNoiseScheduler:
    def __init__(self, initial_noise_std, **kwargs):
        self.noise_std = initial_noise_std

    def step(self, **kwargs):
        return self.noise_std
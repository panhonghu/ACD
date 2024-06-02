import os
import logging
import time
import glob
import numpy as np
import tqdm
import torch
from .unet import Model
from torch.autograd import Variable


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)   # 连乘
    return a


def generalized_steps(x, con_x, modality, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())             # alpha_t in Eq.12
            at_next = compute_alpha(b, next_t.long())   # alpha_{t-1} in Eq.12
            xt = xs[-1].to('cuda')
            et = model(torch.cat([con_x, xt], dim=1), t, modality)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (kwargs.get("eta",0)*((1-at/at_next)*(1-at_next)/(1-at)).sqrt())  # delta_t=0
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next)
    return xs, x0_preds


class Diffusion(object):
    def __init__(self, args, device=None):
        self.args = args
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        self.model_var_type = self.args.var_type
        betas = get_beta_schedule(
            beta_schedule=self.args.beta_schedule,
            beta_start=self.args.beta_start,
            beta_end=self.args.beta_end,
            num_diffusion_timesteps=self.args.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0)
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()
        self.model = Model(self.args)
        self.model = self.model.to(self.device)

    def train(self, x, con_x, modality):   # bs*3*144*72
        n = x.size(0)
        x = x.to(self.device)
        x = 2 * x - 1.0
        e = torch.randn_like(x)
        b = self.betas
        # antithetic sampling
        t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.num_timesteps-t-1], dim=0)[:n]
        return self.noise_estimation_loss(x, con_x, modality, t, e, b)

    def noise_estimation_loss(self, x0, con_x, modality, t, e, b, keepdim=False):
        a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        # print('a -> ', a)
        x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
        output = self.model(torch.cat([con_x, x], dim=1), t.float(), modality)
        if keepdim:
            return (e - output).square().sum(dim=(1, 2, 3))
        else:
            return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

    def sample(self, x0, con_x, modality, last=True):
        # print((1-self.betas).cumprod(dim=0).sqrt()[-1])
        e = torch.randn_like(x0)
        if self.args.skip_type == "uniform":
            skip = self.num_timesteps // self.args.timesteps
            seq = range(0, self.num_timesteps, skip)
        elif self.args.skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps)**2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        xs = generalized_steps(e, con_x, modality, seq, self.model, self.betas, eta=self.args.eta)
        x = xs
        if last:
            x = x[0][-1]
        x = (x + 1.0) / 2.0
        return torch.clamp(x, 0.0, 1.0)

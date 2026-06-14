# Copyright 2025 The RLinf Authors.
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

from typing import Callable, Optional

import torch

from rlinf.algorithms.registry import register_policy_loss
from rlinf.algorithms.utils import huber_loss
from rlinf.utils.utils import masked_mean, masked_mean_ratio
import numpy as np

def compute_decoupled_ppo_actor_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    advantages: torch.Tensor,
    proximal_logprobs: Optional[torch.Tensor] = None,
    versions: Optional[torch.Tensor] = None,
    current_version: Optional[float] = None,
    loss_mask: Optional[torch.Tensor] = None,
    clip_ratio_c: Optional[float] = None,
    loss_agg_func: Optional[Callable[..., torch.Tensor]] = masked_mean,
    max_episode_steps: Optional[int] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    critic_warmup: Optional[bool] = False,
    behave_weight_threshold: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """Compute actor loss for decoupled PPO with optional proximal policy anchor."""
    assert logprobs.dtype == torch.float32, (
        "logprobs must be float32 to keep numerical stability"
    )
    assert old_logprobs.dtype == torch.float32, (
        "old_logprobs must be float32 to keep numerical stability"
    )
    assert advantages.dtype == torch.float32, (
        "advantages must be float32 to keep numerical stability"
    )

    if loss_mask is None:
        loss_mask = torch.ones_like(logprobs).bool()

    loss_mask_ratio = None
    if (
        max_episode_steps is not None
        and loss_mask_sum is not None
        and loss_mask is not None
    ):
        loss_mask_ratio = (loss_mask_sum * 1.0) / max_episode_steps
        loss_agg_func = masked_mean_ratio

    if proximal_logprobs is None:
        if versions is None or current_version is None:
            proximal_logprobs = old_logprobs.detach()
        else:
            v_behav = versions.float()
            v_theta = float(current_version)
            v_prox = v_theta - 1.0

            version_diff = v_theta - v_behav
            version_gap = v_prox - v_behav
            generated_tokens_mask = versions >= 0
            alpha = torch.where(
                (version_diff > 0) & generated_tokens_mask,
                version_gap / version_diff,
                torch.zeros_like(v_behav),
            )
            while alpha.dim() < logprobs.dim():
                alpha = alpha.unsqueeze(-1)
            alpha = torch.clamp(alpha, 0.0, 1.0)
            proximal_logprobs = (
                old_logprobs + alpha * (logprobs - old_logprobs)
            ).detach()

    assert proximal_logprobs.dtype == torch.float32, (
        "proximal_logprobs must be float32 to keep numerical stability"
    )

    loss_mask_count = loss_mask.count_nonzero() or 1
    proximal_ratio = torch.where(
        loss_mask, torch.exp(logprobs - proximal_logprobs), 0.0
    )
    clipped_proximal_ratio = torch.clamp(
        proximal_ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high
    )

    pg_loss1 = -advantages * proximal_ratio
    pg_loss2 = -advantages * clipped_proximal_ratio
    pg_loss = torch.max(pg_loss1, pg_loss2)

    if clip_ratio_c is not None:
        assert clip_ratio_c > 1.0, clip_ratio_c
        pg_loss3 = torch.sign(advantages) * clip_ratio_c * advantages
        dual_clip_mask = pg_loss3.detach() < pg_loss.detach()
        pg_loss = torch.min(pg_loss, pg_loss3)
    else:
        dual_clip_mask = torch.zeros_like(pg_loss, dtype=torch.bool)

    behav_weight = torch.exp(proximal_logprobs - old_logprobs)
    behav_mask = (
        (behav_weight <= behave_weight_threshold).logical_and(loss_mask)
        if behave_weight_threshold is not None
        else loss_mask
    )
    behav_mask_count = behav_mask.count_nonzero() or 1

    pg_loss = loss_agg_func(pg_loss * behav_weight, behav_mask, loss_mask_ratio)
    if critic_warmup:
        pg_loss = torch.tensor(0.0, device=pg_loss.device)

    with torch.no_grad():
        clip_fraction = (pg_loss1 < pg_loss2).logical_and(
            loss_mask
        ).count_nonzero() / loss_mask_count
        dual_clip_fraction = (
            dual_clip_mask.logical_and(loss_mask).count_nonzero() / loss_mask_count
        )
        proximal_approx_kl = (
            -torch.where(loss_mask, logprobs - proximal_logprobs, 0.0).sum()
            / loss_mask_count
        )
        behav_approx_kl = (
            -torch.where(behav_mask, proximal_logprobs - old_logprobs, 0.0).sum()
            / behav_mask_count
        )
        behav_clip_fraction = 1.0 - (behav_mask_count / loss_mask_count)

    metrics_data = {
        "actor/policy_loss": pg_loss.detach(),
        "actor/proximal_ratio": masked_mean(proximal_ratio.detach(), loss_mask),
        "actor/clipped_proximal_ratio": masked_mean(
            clipped_proximal_ratio.detach(), loss_mask
        ),
        "actor/clip_fraction": clip_fraction,
        "actor/dual_clip_fraction": dual_clip_fraction,
        "actor/behav_clip_fraction": behav_clip_fraction,
        "actor/proximal_approx_kl": proximal_approx_kl,
        "actor/behav_approx_kl": behav_approx_kl,
    }
    if (
        versions is not None
        and current_version is not None
        and versions.shape == loss_mask.shape
        and loss_mask.any()
    ):
        metrics_data["actor/average_version"] = versions[loss_mask].float().mean()
        metrics_data["actor/current_version"] = torch.tensor(
            float(current_version), device=logprobs.device
        )

    return pg_loss, metrics_data


def compute_loglik_surrogate_actor_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    load_noise: torch.Tensor,   # [B, K, D]
    load_v_t: torch.Tensor,     # [B, K, D]
    new_v_t: torch.Tensor,  # [B,K,D]
    advantages: torch.Tensor,   # [B]
    prev_x_t_std: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    loss_agg_func: Optional[Callable[..., torch.Tensor]] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    loss_mask_ratio: Optional[torch.Tensor] = None,
    norm_scale: Optional[torch.Tensor] = None,
    use_gradient_reweight: Optional[bool] = False,
    denoise_inds=None,
    denoise_num_steps=None,
    noise_level:float =None,
    detach_v_norm:bool =True,
    gamma=None,
    eps=1e-8,
    delta=1e-6,
    beta_norm:float =0.1,
    lambda_perp:float =1.0,
    use_grad_clip: bool = True,
    max_grad_proj: float = 5.0,
    clip_ratio_low:float = 0.8,
    clip_ratio_high:float = 1.2,
    signal_scale:float =5,
    lambda_align = 0.0004,
    lambda_kl = 0.05,

    **kwargs,
):
    """
    Flow surrogate loss:
        L = - E[ C_t * epsilon^T v_theta * A ]
    """

    assert load_noise.dtype == torch.float32
    assert load_v_t.dtype == torch.float32
    assert advantages.dtype == torch.float32
    assert new_v_t.dtype == torch.float32
    B, K, D = load_noise.shape
    # print("load_noise_in_loss",load_noise.shape)  
    # print("log_probs_in_loss",logprobs.shape)   #[24]
    # print("denoise inds",denoise_inds.shape)   #[24,4]
    # print("load_noise:",load_noise.shape)
    # print("load_v_t",load_v_t.shape)
    if loss_mask is None:
        loss_mask = torch.ones((B, K), device=load_noise.device, dtype=torch.bool)
        
    if loss_agg_func is None:
        def loss_agg_func(x, mask):
            x = torch.where(mask, x, torch.zeros_like(x))
            return x.sum() / (mask.sum() + 1e-8)
    # expand advantage: [B] -> [B, 1]
    adv = advantages[:, None]
    t_values = torch.tensor([0.75, 0.75, 0.5, 0.25], 
                          device=denoise_inds.device, dtype=torch.float32)
    
    t_norm = t_values[denoise_inds]
    t_norm = t_norm[:,:1].repeat(1,8)
    t_norm = t_norm.unsqueeze(-1)   #[24,8,1]

    if noise_level is not None:
        alpha_term = 1 + noise_level**2/2
    else:
        alpha_term = 1
    
    delta_v = new_v_t - load_v_t.detach()
    delta_t = 1/4
    
    scaling_term_1 = (alpha_term**2)*(1.0 - t_norm) * delta_t / (2*t_norm*(noise_level**2))
    scaling_term_2 = alpha_term*torch.sqrt(delta_t*(1-t_norm))/(noise_level*torch.sqrt(t_norm))
    
    align_target = load_noise / scaling_term_2
    align_loss = lambda_align * torch.sum((delta_v + align_target)**2, dim=-1)
    align_loss = align_loss * torch.clamp(adv,min=0)    
    
    # print("scaling_term_1",scaling_term_1.shape)
    # print("scaling_term_2",scaling_term_2.shape)
    time_weight = torch.sqrt(t_norm/(1-t_norm))
    
    logr1 = -scaling_term_1*(delta_v**2).sum(dim=-1, keepdim=True)
    logr2 = -scaling_term_2*(load_noise*delta_v).sum(dim=-1, keepdim=True)
    logr = logr1 + logr2
    logr = torch.clamp(logr, -5, 2)
    ratio = torch.exp(logr)
    ratio = ratio.squeeze(-1)   
    # print("ratio",ratio.shape)    # [24, 8]
    
    ratio_clipped = torch.clamp(ratio, 1-clip_ratio_low, 1+clip_ratio_high)

    surr1 = -ratio * adv
    surr2 = -ratio_clipped * adv


    clip_mask = surr1.detach() < surr2.detach()
    clip_fraction = (clip_mask * loss_mask).sum() / float(loss_mask.count_nonzero())

    policy_loss = torch.max(surr1, surr2)
    
    # tempflow 
    weight = torch.sqrt(t_norm.squeeze(-1))/torch.sqrt(1-t_norm.squeeze(-1))
    # policy_loss *= weight    
    
    kl_loss = 5 * (lambda_kl / 2.0) * (scaling_term_2.squeeze(-1) * torch.norm(delta_v, dim=-1))**2
    total_loss = 10*policy_loss + align_loss + kl_loss
    total_loss = loss_agg_func(total_loss, loss_mask)

    # ---------------------------
    # metrics
    # ---------------------------
    with torch.no_grad():
        policy_loss_mean = torch.where(
            loss_mask,
            policy_loss,
            torch.zeros_like(policy_loss),
        ).sum() / (loss_mask.sum() + eps)
        align_loss_mean = torch.where(
            loss_mask,
            align_loss,
            torch.zeros_like(align_loss),
        ).sum() / (loss_mask.sum() + eps)
        kl_loss_mean = torch.where(
            loss_mask,
            kl_loss,
            torch.zeros_like(kl_loss),
        ).sum() / (loss_mask.sum() + eps)
        
    metrics_data = {
        "actor/total_loss": total_loss.detach(),
        "actor/policy_loss": policy_loss_mean.detach(),
        # "actor/L2_scale":scaling_term_1.mean().detach(),
        # "actor/proj_scale":scaling_term_2.mean().detach(),
        # "actor/L2_loss":torch.exp(logr1).mean().detach(),
        # "actor/proj_loss":torch.exp(logr2).mean().detach(),
        "actor/adv_mean": advantages.mean().detach(),
        "actor/align_loss":align_loss_mean.detach(),
        "actor/kl_loss":kl_loss_mean.detach(),
        "actor/ratio_max": torch.max(ratio).detach(),
        "actor/ratio_min": torch.min(ratio).detach(),
        "actor/ratio_std": ratio.std().detach(),
        "actor/clip_fraction":clip_fraction.detach(),
        "actor/delta_v_norm": (delta_v**2).sum(dim=-1).mean().detach(),
    }

    return total_loss, metrics_data




def compute_orth_loglik_surrogate_actor_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    load_noise: torch.Tensor,   # [B, K, D]
    load_v_t: torch.Tensor,     # [B, K, D]
    new_v_t: torch.Tensor,  # [B,K,D]
    advantages: torch.Tensor,   # [B]
    prev_x_t_std: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    loss_agg_func: Optional[Callable[..., torch.Tensor]] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    loss_mask_ratio: Optional[torch.Tensor] = None,
    norm_scale: Optional[torch.Tensor] = None,
    use_gradient_reweight: Optional[bool] = False,
    denoise_inds=None,
    denoise_num_steps=None,
    noise_level:float =None,
    detach_v_norm:bool =True,
    gamma=None,
    eps=1e-8,
    delta=1e-6,
    beta_norm:float =0.1,
    lambda_perp:float =1.0,
    use_grad_clip: bool = True,
    max_grad_proj: float = 5.0,
    clip_ratio_low:float = 0.8,
    clip_ratio_high:float = 1.2,
    signal_scale:float =5,
    lambda_align = 0.0004,
    lambda_kl = 0.05,

    **kwargs,
):
    """
    Flow surrogate loss:
        L = - E[ C_t * epsilon^T v_theta * A ]
    """

    assert load_noise.dtype == torch.float32
    assert load_v_t.dtype == torch.float32
    assert advantages.dtype == torch.float32
    assert new_v_t.dtype == torch.float32
    B, K, D = load_noise.shape
    # print("load_noise_in_loss",load_noise.shape)  
    # print("log_probs_in_loss",logprobs.shape)   #[24]
    # print("denoise inds",denoise_inds.shape)   #[24,4]
    # print("load_noise:",load_noise.shape)
    # print("load_v_t",load_v_t.shape)
    if loss_mask is None:
        loss_mask = torch.ones((B, K), device=load_noise.device, dtype=torch.bool)
        
    if loss_agg_func is None:
        def loss_agg_func(x, mask):
            x = torch.where(mask, x, torch.zeros_like(x))
            return x.sum() / (mask.sum() + 1e-8)
    # expand advantage: [B] -> [B, 1]
    adv = advantages[:, None]
    t_values = torch.tensor([0.75, 0.75, 0.5, 0.25], 
                          device=denoise_inds.device, dtype=torch.float32)
    
    t_norm = t_values[denoise_inds]
    t_norm = t_norm[:,:1].repeat(1,8)
    t_norm = t_norm.unsqueeze(-1)   #[24,8,1]

    if noise_level is not None:
        alpha_term = 1 + noise_level**2/2
    else:
        alpha_term = 1
    
    delta_v = new_v_t - load_v_t.detach()
    delta_t = 1/4
    
    u = load_v_t / (load_v_t.norm(dim=-1, keepdim=True) + 1e-8)
    I = torch.eye(load_v_t.shape[-1], device=load_v_t.device)
    I = I.expand(u.shape[0],u.shape[1],u.shape[-1],u.shape[-1])
    P_parallel = torch.einsum('...i,...j->...ij', u, u)
    Sigma = 0.05 * P_parallel + 1.0 * (I - P_parallel)   # 单位比例 Sigma
    Sigma_inv = torch.linalg.inv(Sigma + 1e-6 * I)
    
    term1 = torch.einsum('...i,...ij,...j->...', delta_v, Sigma_inv, delta_v)
    term2 = torch.einsum('...i,...ij,...j->...', delta_v, Sigma_inv, load_noise)
    
    scaling_term_1 = (alpha_term**2)*(1.0 - t_norm) * delta_t / (2*t_norm*(noise_level**2))
    scaling_term_2 = alpha_term*torch.sqrt(delta_t*(1-t_norm))/(noise_level*torch.sqrt(t_norm))
    
    align_target = load_noise / scaling_term_2
    align_loss = lambda_align * torch.sum((delta_v + align_target)**2, dim=-1)
    align_loss = align_loss * torch.clamp(adv,min=0)    
    
    # print("scaling_term_1",scaling_term_1.shape)
    # print("scaling_term_2",scaling_term_2.shape)
    time_weight = torch.sqrt(t_norm/(1-t_norm))
    
    logr1 = -scaling_term_1 * term1 
    logr2 = -scaling_term_2 * term2 
    logr = logr1 + logr2
    ratio = torch.exp(logr)
    ratio = ratio.squeeze(-1)   
    # print("ratio",ratio.shape)    # [24, 8]
    
    ratio_clipped = torch.clamp(ratio, 1-clip_ratio_low, 1+clip_ratio_high)

    surr1 = -ratio * adv
    surr2 = -ratio_clipped * adv


    clip_mask = surr1.detach() < surr2.detach()
    clip_fraction = (clip_mask * loss_mask).sum() / float(loss_mask.count_nonzero())

    policy_loss = torch.max(surr1, surr2)
    
    # tempflow 
    weight = torch.sqrt(t_norm.squeeze(-1))/torch.sqrt(1-t_norm.squeeze(-1))
    # policy_loss *= weight    
    
    kl_loss = 5 * (lambda_kl / 2.0) * (scaling_term_2.squeeze(-1) * torch.norm(delta_v, dim=-1))**2
    total_loss = 10*policy_loss + align_loss + kl_loss
    total_loss = loss_agg_func(total_loss, loss_mask)

    # ---------------------------
    # metrics
    # ---------------------------
    with torch.no_grad():
        policy_loss_mean = torch.where(
            loss_mask,
            policy_loss,
            torch.zeros_like(policy_loss),
        ).sum() / (loss_mask.sum() + eps)
        align_loss_mean = torch.where(
            loss_mask,
            align_loss,
            torch.zeros_like(align_loss),
        ).sum() / (loss_mask.sum() + eps)
        kl_loss_mean = torch.where(
            loss_mask,
            kl_loss,
            torch.zeros_like(kl_loss),
        ).sum() / (loss_mask.sum() + eps)
        
    metrics_data = {
        "actor/total_loss": total_loss.detach(),
        "actor/policy_loss": policy_loss_mean.detach(),
        # "actor/L2_scale":scaling_term_1.mean().detach(),
        # "actor/proj_scale":scaling_term_2.mean().detach(),
        # "actor/L2_loss":torch.exp(logr1).mean().detach(),
        # "actor/proj_loss":torch.exp(logr2).mean().detach(),
        "actor/adv_mean": advantages.mean().detach(),
        "actor/align_loss":align_loss_mean.detach(),
        "actor/kl_loss":kl_loss_mean.detach(),
        "actor/ratio_max": torch.max(ratio).detach(),
        "actor/ratio_min": torch.min(ratio).detach(),
        "actor/ratio_std": ratio.std().detach(),
        "actor/clip_fraction":clip_fraction.detach(),
        "actor/delta_v_norm": (delta_v**2).sum(dim=-1).mean().detach(),
    }

    return total_loss, metrics_data







def compute_flow_surrogate_actor_loss(
    load_noise: torch.Tensor,   # [B, K, D]
    load_v_t: torch.Tensor,     # [B, K, D]
    new_v_t: torch.Tensor,  # [B,K,D]
    advantages: torch.Tensor,   # [B]
    prev_x_t_std: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    loss_agg_func: Optional[Callable[..., torch.Tensor]] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    loss_mask_ratio: Optional[torch.Tensor] = None,
    norm_scale: Optional[torch.Tensor] = None,
    use_gradient_reweight: Optional[bool] = False,
    denoise_inds=None,
    denoise_num_steps=None,
    detach_v_norm:bool =True,
    gamma=None,
    eps=1e-8,
    delta=1e-6,
    beta_norm:float =0.1,
    lambda_perp:float =1.0,
    _beta: float = 6,
    use_grad_clip: bool = True,
    max_grad_proj: float = 5.0,
    clip_ratio_low:float = 0.8,
    clip_ratio_high:float = 1.2,
    signal_scale:float =5,
    **kwargs,
):
    """
    Flow surrogate loss:
        L = - E[ C_t * epsilon^T v_theta * A ]
    """

    assert load_noise.dtype == torch.float32
    assert load_v_t.dtype == torch.float32
    assert advantages.dtype == torch.float32
    assert new_v_t.dtype == torch.float32
    B, K, D = load_noise.shape

    # print("denoise inds",denoise_inds)
    # print("load_noise:",load_noise.shape)
    # print("load_v_t",load_v_t.shape)
    
    t_values = torch.tensor([0.75, 0.75, 0.5, 0.25], 
                          device=denoise_inds.device, dtype=torch.float32)
    
    t_norm = t_values[denoise_inds]
    t_norm = t_norm[:,:1].unsqueeze(-1)
    # print(t_norm)
    scaling_term = torch.sqrt((1.0 - t_norm) / t_norm)
    if loss_mask is None:
        loss_mask = torch.ones((B, K), device=load_noise.device, dtype=torch.bool)
        
    if loss_agg_func is None:
        def loss_agg_func(x, mask):
            x = torch.where(mask, x, torch.zeros_like(x))
            return x.sum() / (mask.sum() + 1e-8)
    # expand advantage: [B] -> [B, 1]
    adv = advantages[:, None]
    score_new = (load_noise * new_v_t).sum(dim=-1) 
    score_old = (load_noise * load_v_t).sum(dim=-1)
    delta_s = score_new - score_old.detach()

    ratio = torch.exp(_beta * delta_s)
    # print("ratio min/max/mean:", ratio.min().item(), ratio.max().item(), ratio.mean().item())
    # print("clamped_ratio:", torch.clamp(ratio, 1-clip_ratio_low, 1+clip_ratio_high))
    ratio_clipped = torch.clamp(ratio, 1-clip_ratio_low, 1+clip_ratio_high)

    surr1 = -ratio * adv
    surr2 = -ratio_clipped * adv
    # print("surr1",surr1)
    # print("surr2",surr2)
    clip_mask= surr1.detach()!=surr2.detach()
    clip_fraction = clip_mask.count_nonzero()/loss_mask.count_nonzero()
    policy_loss = torch.max(surr1, surr2)
    
    delta_v = new_v_t - load_v_t
    perp_proj = delta_v - (torch.sum(delta_v * load_noise, dim=-1, keepdim=True) * load_noise)
    perp_loss = lambda_perp * torch.sum(perp_proj ** 2, dim=-1)
    
    norm_diff = (torch.norm(new_v_t, dim=-1) - torch.norm(load_v_t + prev_x_t_std * load_noise, dim=-1)) ** 2
    weight = torch.sigmoid(3*adv)
    norm_loss = beta_norm * weight * norm_diff    #[B , action horizon]
    
    total_loss = 10*policy_loss + 5*perp_loss + norm_loss
    
   

    if use_gradient_reweight:
        K_steps = denoise_num_steps[0, 0].item()

        def build_t(K, device):
            return torch.linspace(1, 1 / K, K, device=device)

        def compute_weight(K, device, a=1.0):
            t = build_t(K, device)
            delta_t = 1.0 / K
            sigma = a * torch.sqrt(t / (1 - torch.where(t == 1, t[1], t)))
            w = sigma * torch.sqrt(torch.tensor(delta_t, device=device))
            return w

        def normalize_weight(w, eps=1e-8):
            return w / (w.mean() + eps)

        w = compute_weight(K_steps, loss.device)
        w = normalize_weight(w)

        # select timestep weight
        # denoise_inds: [B, K]
        idx = denoise_inds[:, 1]  # [B]
        scale = w[idx]            # [B]
        scale = scale[:, None, None]  # [B,1,1]

        loss = loss * scale

    total_loss = loss_agg_func(total_loss, loss_mask)

    # ---------------------------
    # metrics
    # ---------------------------
    with torch.no_grad():
        perp_loss_mean = torch.where(
            loss_mask,
            perp_loss,
            torch.zeros_like(perp_loss),
        ).sum() / (loss_mask.sum() + eps)

        norm_loss_mean = torch.where(
            loss_mask,
            norm_loss,
            torch.zeros_like(norm_loss),
        ).sum() / (loss_mask.sum() + eps)

        policy_loss_mean = torch.where(
            loss_mask,
            policy_loss,
            torch.zeros_like(policy_loss),
        ).sum() / (loss_mask.sum() + eps)

    # print("load_v_t",load_v_t.requires_grad)
    # print("load_noise",load_noise.requires_grad)
    # print("new_v_t",new_v_t.requires_grad)
    metrics_data = {
        "actor/total_loss": total_loss.detach(),
        "actor/policy_loss": policy_loss_mean.detach(),
        "actor/perp_loss": perp_loss_mean.detach(),
        "actor/norm_loss": norm_loss_mean.detach(),
        "actor/adv_mean": advantages.mean().detach(),
        "actor/adv_mean": advantages.mean().detach(),
        # "actor/v_norm_mean": v_norm.mean().detach(),
        # "actor/cos_abs_mean": cosine.abs().mean().detach(),
        "actor/ratio_max": torch.max(ratio).detach(),
        "actor/ratio_min": torch.min(ratio).detach(),
        "actor/clip_fraction":clip_fraction.detach(),
    }

    return total_loss, metrics_data


def compute_ppo_actor_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    advantages: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    clip_ratio_c: Optional[float] = None,
    loss_agg_func: Optional[Callable[..., torch.Tensor]] = masked_mean,
    max_episode_steps: Optional[int] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    critic_warmup: Optional[bool] = False,
    clip_log_ratio_min: Optional[float] = None,
    clip_log_ratio_max: Optional[float] = None,
    fast_path_zero_loss_mask: Optional[bool] = False,
    denoise_inds=None,
    denoise_num_steps=None,
    kl_beta=None,
    norm_scale=None,
    gamma=None,
    use_gradient_reweight: Optional[bool]=False,
    load_noise=None,
    load_v_t=None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO actor loss function.

    Args:
        logprobs (torch.FloatTensor): Log probabilities of actions.
        old_logprobs (torch.FloatTensor): Old log probabilities of actions.
        clip_ratio_low (float): Lower bound of clipping ratio.
        clip_ratio_high (float): Upper bound of clipping ratio.
        advantages (torch.FloatTensor): GAE (normalized) advantages.
        loss_mask (Optional[torch.BoolTensor], optional): Mask for valid entries. Defaults to None.
        clip_ratio_c (Optional[float], optional): Optional clipping coefficient. Defaults to None.
        loss_agg_func (callable, optional): Aggregation function (e.g., masked_mean). Defaults to None.
        max_episode_steps (Optional[int], optional): Max episode length for normalization. Defaults to None.

    Returns:
        Tuple[torch.Tensor, Dict]: (actor_loss, metrics_dict)
    """
    if fast_path_zero_loss_mask and (
        loss_mask is not None and loss_mask[0].sum() == 0.0
    ):
        return torch.tensor(0.0, device=logprobs.device), {
            "actor/token_num": torch.tensor(0.0, device=logprobs.device),
            "actor/policy_loss": torch.tensor(0.0, device=logprobs.device),
            "actor/policy_loss_mbs_mean": torch.tensor(0.0, device=logprobs.device),
            "actor/policy_loss_abs": torch.tensor(0.0, device=logprobs.device),
            "actor/ratio": torch.tensor(0.0, device=logprobs.device),
            "actor/clipped_ratio": torch.tensor(0.0, device=logprobs.device),
            "actor/dual_cliped_ratio": torch.tensor(0.0, device=logprobs.device),
            "actor/approx_kl": torch.tensor(0.0, device=logprobs.device),
            "actor/clip_fraction": torch.tensor(0.0, device=logprobs.device),
        }

    loss_mask_ratio = None

    if (
        max_episode_steps is not None
        and loss_mask_sum is not None
        and loss_mask is not None
    ):
        loss_mask_ratio = (loss_mask_sum * 1.0) / max_episode_steps
        loss_agg_func = masked_mean_ratio

    if loss_mask is None:
        loss_mask = torch.ones_like(logprobs).bool()

    assert logprobs.dtype == torch.float32, (
        "logprobs must be float32 to keep numerical stability"
    )
    assert old_logprobs.dtype == torch.float32, (
        "old_logprobs must be float32 to keep numerical stability"
    )
    assert advantages.dtype == torch.float32, (
        "advantages must be float32 to keep numerical stability"
    )

    loss_mask_count = loss_mask.count_nonzero() or 1
    # For numerical stability.  
    log_ratio = logprobs - old_logprobs    # [12, 5, 7]
    if clip_log_ratio_min is not None:
        log_ratio = torch.clamp(log_ratio, min=clip_log_ratio_min)
    if clip_log_ratio_max is not None:
        log_ratio = torch.clamp(log_ratio, max=clip_log_ratio_max)
    ratio = torch.where(loss_mask, torch.exp(log_ratio), 0)
    # approx_kl = torch.where(loss_mask, log_ratio.detach(), 0.0)
    #----wyh--
    
    
    print("load_noise:",load_noise.shape)
    print("load_v_t:",load_v_t.shape)
    print("advantage:",advantages.shape)
    
    with torch.no_grad():
        approx_kl = (ratio - 1) - log_ratio
        mean_kl = loss_agg_func(approx_kl, loss_mask, loss_mask_ratio)
    
    #--------
  
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    policy_loss1 = -advantages * ratio
    policy_loss2 = -advantages * clipped_ratio    
    
    clip_mask = policy_loss1.detach() < policy_loss2.detach()
    policy_loss = torch.max(policy_loss1, policy_loss2)
    # -------wyh-------  tempflow grpo
    if use_gradient_reweight:
        denoise_num_step=denoise_num_steps[0,0].item()
        def build_t(K, device):
            return torch.linspace(1, 1/K, K, device=device)
        def compute_weight(K, device, a=1.0):
            t = build_t(K, device)                 # [K]
            delta_t = 1.0 / K
            sigma = a * torch.sqrt(t / (1 - torch.where(t==1,t[1],t)))    # [K]
            w = sigma * torch.sqrt(torch.tensor(delta_t))        # [K]
            return w
        def normalize_weight(w, eps=1e-8):
            return w / (w.mean() + eps)
        w = compute_weight(denoise_num_step, policy_loss.device)   # [K]
        w = normalize_weight(w)   # [K]
        # print("norm_scale",norm_scale.shape)  [24,8,32]
        # print("policy loss",policy_loss.shape)  [24,5,7]
        # deniose_inds  [bzs,num_steps]
        denoise_inds=denoise_inds[:,1]
        scale=w[denoise_inds]
        scale=scale.unsqueeze(1).unsqueeze(1)
        
        # norm_scale=norm_scale[:, :policy_loss.shape[1], :policy_loss.shape[2]]
        # policy_loss=policy_loss*(norm_scale/norm_scale.mean())
        policy_loss=policy_loss*scale
    #---------------------------------
    target_mean_kl=0.1
    kl_beta=kl_beta*(mean_kl/target_mean_kl)
    kl_loss =  kl_beta * mean_kl
    actor_loss = policy_loss + kl_loss
    
    actor_loss = loss_agg_func(actor_loss, loss_mask, loss_mask_ratio)
    kl_loss = loss_agg_func(kl_loss, loss_mask, loss_mask_ratio)
    if clip_ratio_c is not None:
        assert clip_ratio_c > 1.0, "clip_ratio_c must be greater than 1.0"
        policy_loss3 = torch.sign(advantages) * clip_ratio_c * advantages
        dual_clip_mask = policy_loss3.detach() < policy_loss.detach()
        policy_loss = torch.min(policy_loss, policy_loss3)
    else:
        dual_clip_mask = torch.zeros_like(clip_mask)

    metric_policy_loss_abs = loss_agg_func(
        policy_loss.abs(), loss_mask, loss_mask_ratio
    )
    policy_loss = loss_agg_func(
        policy_loss, loss_mask, loss_mask_ratio
    )  # default max_episode_steps is None

    clip_mask = policy_loss1.detach() < policy_loss2.detach()
    dual_clip_mask = (dual_clip_mask * loss_mask).bool()

    clip_fraction = (clip_mask * loss_mask).sum() / float(loss_mask_count)
    approx_kl = -torch.sum(approx_kl) / float(loss_mask_count)

    dual_cliped_ratio = torch.where(dual_clip_mask, ratio, 0)

    if critic_warmup:
        policy_loss = torch.tensor(0.0, device=policy_loss.device)

    # Compile metrics for logging
    loss_mask_for_metrics = loss_mask
    ratio_for_metrics = ratio.detach()
    ratio_abs_for_metrics = (ratio - 1).abs().detach()
    clipped_ratio_for_metrics = clipped_ratio.detach()
    dual_cliped_ratio_for_metrics = dual_cliped_ratio.detach()

    # Only broadcast when ratio has action_dim dimension and loss_mask's last dim is 1
    # This handles token_level mode: ratio [bsz, num_chunks, action_dim], loss_mask [bsz, num_chunks, 1]
    if len(ratio.shape) > 2 and loss_mask.shape[-1] == 1 and ratio.shape[-1] > 1:
        # Broadcast loss_mask to match ratio's shape for metrics computation
        loss_mask_for_metrics = loss_mask.expand_as(ratio)

    metrics_data = {
        "actor/policy_loss": policy_loss.detach(),
        "actor/kl_loss": kl_loss.detach(),  #wyh
        "actor/policy_loss_abs": metric_policy_loss_abs.detach(),
        "actor/ratio": masked_mean(ratio_for_metrics, loss_mask_for_metrics),
        "actor/ratio_abs": masked_mean(ratio_abs_for_metrics, loss_mask_for_metrics),
        "actor/clipped_ratio": masked_mean(
            clipped_ratio_for_metrics, loss_mask_for_metrics
        ),
        "actor/dual_cliped_ratio": masked_mean(
            dual_cliped_ratio_for_metrics, loss_mask_for_metrics
        ),
        "actor/approx_kl": mean_kl.detach(),  #wyh
        "actor/clip_fraction": clip_fraction.detach(),
        "actor/gamma_mean":gamma.mean().detach(),
    }
    return actor_loss, metrics_data


def compute_ppo_critic_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    prev_values: torch.Tensor,
    value_clip: float,
    huber_delta: float,
    loss_mask: Optional[torch.Tensor] = None,
    max_episode_steps: Optional[int] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO critic loss function.

    Args:
        values (torch.Tensor): Current value predictions.
        returns (torch.Tensor): Return values.
        prev_values (torch.Tensor): Previous value predictions.
        value_clip (float): Value clipping threshold.
        huber_delta (float): Huber loss delta parameter.

    Returns:
        Tuple[torch.Tensor, Dict]: (critic_loss, metrics_dict)
    """
    loss_mask_ratio = None
    loss_agg_func = masked_mean

    if (
        max_episode_steps is not None
        and loss_mask_sum is not None
        and loss_mask is not None
    ):
        loss_mask_ratio = (loss_mask_sum * 1.0) / max_episode_steps
        loss_agg_func = masked_mean_ratio

    value_pred_clipped = prev_values + (values - prev_values).clamp(
        -value_clip, value_clip
    )  # [bsz, ] | [bsz, chunk-step]

    value_loss_original = huber_loss(
        returns - values, huber_delta
    )  # [bsz, ] | [bsz, chunk-step]
    value_loss_clipped = huber_loss(
        returns - value_pred_clipped, huber_delta
    )  # [bsz, ] | [bsz, chunk-step]
    value_loss = torch.max(value_loss_original, value_loss_clipped)
    value_loss = loss_agg_func(value_loss, loss_mask, loss_mask_ratio)

    value_clip_indicator = (value_pred_clipped - prev_values).abs() > value_clip
    value_clip_ratio = value_clip_indicator.float().mean()

    # explained variance
    if loss_mask is not None:
        masked_returns = returns[loss_mask]
        masked_values = values[loss_mask]
    else:
        masked_returns = returns
        masked_values = values

    var_returns = torch.var(masked_returns)
    if torch.isnan(var_returns) or var_returns == 0:
        explained_variance = torch.tensor(float("nan"), device=returns.device)
    else:
        var_diff = torch.var(masked_returns - masked_values)
        if torch.isnan(var_diff):
            explained_variance = torch.tensor(float("nan"), device=returns.device)
        else:
            explained_variance = 1 - var_diff / var_returns

    # Compile metrics for logging
    metrics_data = {
        "critic/value_loss": value_loss.detach(),
        "critic/value_clip_ratio": value_clip_ratio.detach(),
        "critic/explained_variance": explained_variance.detach(),
    }
    return value_loss, metrics_data


@register_policy_loss("decoupled_actor_critic")
def compute_decoupled_ppo_actor_critic_loss(**kwargs) -> tuple[torch.Tensor, dict]:
    """Compute decoupled PPO actor+critic loss."""
    metrics_data = {}
    actor_loss, actor_metrics_data = compute_decoupled_ppo_actor_loss(**kwargs)
    critic_loss, critic_metrics_data = compute_ppo_critic_loss(**kwargs)

    loss = actor_loss + critic_loss
    metrics_data.update(actor_metrics_data)
    metrics_data.update(critic_metrics_data)
    return loss, metrics_data


@register_policy_loss("actor_critic")
def compute_ppo_actor_critic_loss(**kwargs) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO actor loss function.

    Args:
        logprobs (torch.Tensor): Log probabilities of actions
        values (torch.Tensor): Current value predictions
        old_log_prob (torch.Tensor): Previous log probabilities
        advantages (torch.Tensor): Advantage values
        returns (torch.Tensor): Return values
        prev_values (torch.Tensor): Previous value predictions
        clip_ratio_low (float): Lower clipping ratio for PPO
        clip_ratio_high (float): Upper clipping ratio for PPO
        value_clip (float): Value clipping threshold
        huber_delta (float): Huber loss delta parameter

    Returns:
        Tuple[torch.Tensor, Dict]: Loss and metrics dictionary
    """
    metrics_data = {}
    # actor_loss, actor_metrics_data = compute_ppo_actor_loss(**kwargs)
    # actor_loss, actor_metrics_data = compute_flow_surrogate_actor_loss(**kwargs)
    actor_loss, actor_metrics_data = compute_loglik_surrogate_actor_loss(**kwargs)
    critic_loss, critic_metrics_data = compute_ppo_critic_loss(**kwargs)

    loss = actor_loss + critic_loss
    metrics_data.update(actor_metrics_data)
    metrics_data.update(critic_metrics_data)

    return loss, metrics_data


@register_policy_loss("actor")
def compute_grpo_actor_loss_fn(**kwargs) -> tuple[torch.Tensor, dict]:
    """
    Compute actor loss for Group Relative Policy Optimization (GRPO).

    This function implements the PPO-style actor loss with clipping for GRPO.
    Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppotrainer.py#L1122

    Args:
        log_prob (torch.Tensor): Current log probabilities
        old_log_prob (torch.Tensor): Previous log probabilities
        advantages (torch.Tensor): Advantage values of shape
        clip_ratio_high (float): Upper clipping ratio for PPO
        clip_ratio_low (float): Lower clipping ratio for PPO
        loss_mask (Optional[torch.Tensor]): Mask tensor of shape to apply to the loss

    Returns:
        Tuple[torch.Tensor, Dict]: Policy gradient loss and metrics dictionary containing:
            - actor/loss: Total actor loss
            - actor/policy_loss: Policy gradient loss
            - actor/clip_fraction: Fraction of clipped policy gradient loss
            - actor/ppo_kl: Approximate KL divergence
    """
    metrics_data = {}
    actor_loss, actor_metrics_data = compute_ppo_actor_loss(**kwargs)
    metrics_data.update(actor_metrics_data)

    return actor_loss, metrics_data

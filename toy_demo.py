# import os
# import time
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import gymnasium as gym
# import numpy as np
# from torch.distributions import Categorical
# from torch.utils.tensorboard import SummaryWriter
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")


# # ========================
# # Actor-Critic Network
# # ========================
# class ActorCritic(nn.Module):
#     def __init__(self, obs_dim, act_dim):
#         super().__init__()
#         self.shared = nn.Sequential(
#             nn.Linear(obs_dim, 64),
#             nn.Tanh(),
#             nn.Linear(64, 64),
#             nn.Tanh(),
#         )
#         self.actor = nn.Linear(64, act_dim)
#         self.critic = nn.Linear(64, 1)

#     def forward(self, x):
#         x = self.shared(x)
#         logits = self.actor(x)
#         value = self.critic(x)
#         return logits, value

#     def get_action_and_value(self, obs):
#         logits, value = self.forward(obs)
#         dist = Categorical(logits=logits)
#         action = dist.sample()
#         log_prob = dist.log_prob(action)
#         return action, log_prob, value.squeeze(-1), dist


# # ========================
# # GAE (VectorEnv 版本)
# # ========================
# @torch.no_grad()
# def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
#     """rewards, values, dones: shape (steps, num_envs)"""
#     steps, num_envs = rewards.shape
#     advantages = torch.zeros_like(rewards, device=device)
#     last_gae = torch.zeros(num_envs, device=device)
#     for t in reversed(range(steps)):
#         if t == steps - 1:
#             next_value = last_value
#         else:
#             next_value = values[t + 1]

#         delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
#         last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
#         advantages[t] = last_gae

#     returns = advantages + values
#     return advantages, returns
#     # for t in reversed(range(steps)):
#     #     next_value = torch.zeros(num_envs, device=device) if t == steps - 1 else values[t + 1]
#     #     delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
#     #     last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
#     #     advantages[t] = last_gae

#     # return advantages, advantages + values


# # ========================
# # 安全版 Hessian Trace Estimator（曲率估计）
# # ========================
# def hessian_trace_estimate(loss, model):
#     params = [p for p in model.parameters() if p.requires_grad]
#     grads = torch.autograd.grad(
#     loss, params,
#     create_graph=True,
#     allow_unused=True
#     )
#     grads = [g for g in grads if g is not None]

#     # 2. 采样和梯度一样结构的随机噪声 z
#     z = [torch.randn_like(g) for g in grads]

#     # 3. 计算 H * z （核心！不会有形状错误）
#     H_z = torch.autograd.grad(
#         grads, params,
#         grad_outputs=z,
#         retain_graph=True,
#         allow_unused=True
#     )
#     H_z = [g for g in H_z if g is not None]

#     # 4. 计算 z^T H z （迹的无偏估计）
#     trace = 0.0
#     for zi, hzi in zip(z, H_z):
#         trace += (zi * hzi).sum()
    
#     return trace
#     # """Hutchinson 方法估计 Hessian trace（已处理 grad_fn 问题）"""
#     # model.zero_grad()
#     # params = [p for p in model.parameters() if p.requires_grad]

#     # # 第一步：计算一阶梯度（保留计算图）
#     # grads = torch.autograd.grad(
#     #     loss, params,
#     #     create_graph=True,
#     #     retain_graph=True,
#     #     allow_unused=True
#     # )

#     # g_vec = torch.cat([g.reshape(-1) for g in grads if g is not None])

#     # # 随机向量 v
#     # v = [torch.randn_like(p) for p in params]
#     # # v = v / (v.norm() + 1e-8)
#     # print("g_vec",g_vec.shape)
#     # for p in params:
#     #     print("p",p.shape)

#     # # 计算 Hv = ∇²loss · v
#     # Hv = torch.autograd.grad(
#     #     g_vec,
#     #     params,
#     #     grad_outputs=v,
#     #     retain_graph=False,
#     #     allow_unused=True
#     # )

#     # Hv_vec = torch.cat([h.reshape(-1) for h in Hv if h is not None])
#     # print("Hv_vec",Hv_vec.shape)
#     # # trace = torch.dot(v, Hv_vec).item()
#     # trace = (torch.cat([vv.flatten() for vv in v]) * Hv_vec).sum()
#     # return max(trace, 1e-6)  # 避免除零


# # ========================
# # 动态正交梯度噪声（根据曲率自适应）
# # ========================
# # @torch.no_grad()
# def apply_orthogonal_gradient_noise(model, loss, base_scale=5e-4, compute_trace_every=5):
#     """正交噪声 + 曲率自适应 scale"""
#     # 计算一阶梯度（保留图给后面的 backward）
#     grads = torch.autograd.grad(
#         loss, model.parameters(),
#         create_graph=False,
#         retain_graph=True,
#         allow_unused=True
#     )

#     grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, model.parameters())]
#     grads = [g.unsqueeze(1) if g.dim()==1 else g for g in grads]
#     noise = [torch.randn_like(g) for g in grads]

#     g_dot_v = []
#     g_norm_sq = []

#     for i, (g,v) in enumerate(zip(grads, noise)):
#         g_dot_v.append((g * v).sum(dim=1,keepdim=True))
#         g_norm_sq.append((g **2).sum(dim=1,keepdim=True))

#     for i in range(len(noise)):
#         # print(noise[i].shape,grads[i].shape,(g_dot_v[i]/g_norm_sq[i]).shape)
#         noise[i] = noise[i] -(g_dot_v[i]/g_norm_sq[i]) * grads[i]
        

#     trace = hessian_trace_estimate(loss, model)
#     noise_scale = base_scale / (np.sqrt(abs(trace.cpu())) + 1e-6)
#     # for p,g,z in zip(model.parameters(),grads,noise):
#         # print(p,noise_scale * z)
#     #     print("p mean",g.mean(),"z mean",(noise_scale * z).mean())
#     # breakpoint()
#     # noisy_params = [p + noise_scale * z for p, z in zip(model.parameters(), noise)]
#     # noisy_params = [noisy_param.squeeze(1) if noisy_param.shape[1]==1 else noisy_param for noisy_param in noisy_params ]
#     noisy_grads = [g + noise_scale * z for g, z in zip(grads, noise)]
#     noisy_grads = [noisy_grad.squeeze(1) if noisy_grad.shape[1]==1 else noisy_grad for noisy_grad in noisy_grads ]
#     return noisy_grads, float(noise_scale)


# def assign_grads(model, grads):
#     for p, g in zip(model.parameters(), grads):
#         if g is None:
#             continue
#         if p.grad is None:
#             p.grad = g.detach().clone()
#         else:
#             p.grad.copy_(g.detach())

# def update_params(model,params):
#     print("update params")
#     for p, v in zip(model.parameters(), params):
#         p = v.detach()
    
# # ========================
# # PPO Loss
# # ========================
# def ppo_loss(new_logp, old_logp, adv, entropy, clip=0.2, ent_coef=0.01):
#     ratio = torch.exp(new_logp - old_logp)
#     clipped_ratio = torch.clamp(ratio, 1 - clip, 1 + clip)
#     surrogate = torch.min(ratio * adv, clipped_ratio * adv)
#     return -torch.mean(surrogate)-ent_coef * entropy.mean()


# def approx_kl(new_logp, old_logp):
#     return 0.5 * torch.mean((new_logp - old_logp) ** 2)
#     # return torch.mean(old_logp - new_logp)


# # ========================
# # Collect Rollout（正确 episode reward）
# # ========================
# def collect_rollout(envs, policy, steps=2048):
#     obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
#     episode_returns = [0.0] * envs.num_envs
#     total_episode_rewards = []

#     obs, _ = envs.reset()
#     obs = torch.as_tensor(obs, dtype=torch.float32, device=device)

#     for _ in range(steps):
#         with torch.no_grad():
#             act, logp, val, _ = policy.get_action_and_value(obs)

#         next_obs, reward, terminated, truncated, _ = envs.step(act.cpu().numpy())
#         done = np.logical_or(terminated, truncated)

#         # 累加 episode reward
#         for i in range(envs.num_envs):
#             episode_returns[i] += float(reward[i])
#             if done[i]:
#                 total_episode_rewards.append(episode_returns[i])
#                 episode_returns[i] = 0.0
                

#         obs_buf.append(obs)
#         act_buf.append(act)
#         logp_buf.append(logp)
#         rew_buf.append(torch.as_tensor(reward, dtype=torch.float32, device=device))
#         val_buf.append(val)
#         done_buf.append(torch.as_tensor(done, dtype=torch.float32, device=device))

#         obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)

#     # 最后一个 value
#     with torch.no_grad():
#         _, last_val = policy.forward(obs)
#         last_val = last_val.squeeze(-1).detach()

#     values = torch.stack(val_buf)
#     rewards = torch.stack(rew_buf)
#     dones = torch.stack(done_buf)

#     advantages, returns = compute_gae(rewards, values, dones, last_val)

#     # 展平并归一化
#     advantages = advantages.reshape(-1)
#     returns = returns.reshape(-1)
#     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

#     obs_tensor = torch.cat(obs_buf)
#     act_tensor = torch.cat(act_buf)
#     old_logp_tensor = torch.cat(logp_buf)

#     avg_reward = np.mean(total_episode_rewards) if total_episode_rewards else 0.0

#     return obs_tensor, act_tensor, old_logp_tensor, advantages, returns, avg_reward


# # ========================
# # Train Step
# # ========================
# def train_step(policy, optimizer, obs, act, old_logp, adv, returns, use_noise=False, base_scale=5e-3):
#     optimizer.zero_grad()

#     logits, values = policy(obs)
#     dist = Categorical(logits=logits)
#     new_logp = dist.log_prob(act)
#     entropy = dist.entropy()
#     policy_loss = ppo_loss(new_logp, old_logp, adv, entropy, ent_coef=0.01)
#     value_loss = 0.5 * nn.functional.mse_loss(values.squeeze(-1), returns)
#     kl = approx_kl(new_logp, old_logp)

#     loss = policy_loss + 0.5 * value_loss + 0.01 * kl

#     if use_noise:
#         noisy_grads, noise_scale = apply_orthogonal_gradient_noise(
#             policy, loss, base_scale=base_scale
#         )
#         assign_grads(policy,noisy_grads)
#         # update_params(policy,noisy_params)
#     else:
#         loss.backward()
#         noise_scale = 0.0

#     torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
#     optimizer.step()

#     return policy_loss.item(), kl.item(), value_loss.item(), noise_scale, entropy.cpu().detach()


# # ========================
# # Main Training Loop
# # ========================
# def run(use_noise=False, base_scale=5e-3, epochs=60,run_time=None):
#     num_envs = 20
#     log_root = "runs" 
#     if use_noise:
#         run_name = run_time+"_Noise"
#     else:
#         run_name = run_time+"_Baseline"
#     log_dir = os.path.join(log_root, run_name)
#     writer = SummaryWriter(log_dir=log_dir)
    
#     envs = gym.vector.SyncVectorEnv([lambda: gym.make("MountainCar-v0") for _ in range(num_envs)])
#     obs_dim = envs.single_observation_space.shape[0]
#     act_dim = envs.single_action_space.n
    
#     policy = ActorCritic(obs_dim=obs_dim, act_dim=act_dim).to(device)
#     optimizer = optim.Adam(policy.parameters(), lr=3e-4)

#     best_reward = 0.0
#     for epoch in range(epochs):
#         obs, act, old_logp, adv, returns, avg_reward = collect_rollout(envs, policy)

#         losses, kls, vlosses, noise_scales,entropys = [], [], [], [], []
#         for _ in range(4):
#             p_loss, kl, v_loss, n_scale,entropy = train_step(
#                 policy, optimizer, obs, act, old_logp, adv, returns,
#                 use_noise=use_noise, base_scale=base_scale
#             )
#             losses.append(p_loss)
#             kls.append(kl)
#             vlosses.append(v_loss)
#             noise_scales.append(n_scale)
#             entropys.append(entropy)
#         mean_noise = np.mean([s for s in noise_scales if s > 0]) if use_noise else 0.0

#         if avg_reward > best_reward:
#             best_reward = avg_reward
#         writer.add_scalar("Reward/avg", avg_reward, epoch)
#         writer.add_scalar("Loss/policy", np.mean(losses), epoch)
#         writer.add_scalar("Loss/value", np.mean(vlosses), epoch)
#         writer.add_scalar("KL", np.mean(kls), epoch)
#         writer.add_scalar("Entropy", np.mean(entropys),epoch)
#         if use_noise:
#             writer.add_scalar("Noise/scale", mean_noise, epoch)
#         print(f"[{'NOISE' if use_noise else 'BASELINE'}] "
#               f"Epoch {epoch:3d} | "
#               f"Policy Loss {np.mean(losses):.4f} | "
#               f"KL {np.mean(kls):.4f} | "
#               f"Value Loss {np.mean(vlosses):.4f} | "
#               f"Avg Episode Reward {avg_reward:.1f} (best {best_reward:.1f}) | "
#               f"Entropy {np.mean(entropys):.4f} |"
#               f"Noise Scale {mean_noise:.6f}")

#     envs.close()
#     writer.close()
#     return best_reward


# # ========================
# # Run
# # ========================
# if __name__ == "__main__":
#     torch.manual_seed(42)
#     np.random.seed(42)
#     run_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    
    
#     print("\n=== 1. Baseline PPO ===")
#     baseline_reward = run(use_noise=False,run_time=run_time)
    


#     print("\n=== 2. PPO + 曲率自适应正交梯度噪声 ===")
#     noise_reward = run(use_noise=True, base_scale=5e-4 ,run_time=run_time)
    

    
#     print("\n=== 训练结束 ===")
#     print(f"Baseline 最终平均奖励: {baseline_reward:.1f}")
#     print(f"带噪声版本最终平均奖励: {noise_reward:.1f}")
    



import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Actor-Critic (Gaussian)
# =========================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )

        self.actor_mean = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)

        # state-independent log std（经典 PPO 写法）
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        x = self.shared(x)
        mean = self.actor_mean(x)
        value = self.critic(x)
        return mean, value


# =========================
# PPO Buffer
# =========================
class Buffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.__init__()


# =========================
# PPO Agent
# =========================
class PPO:
    def __init__(self, state_dim, action_dim):

        self.gamma = 0.99
        self.lam = 0.95
        self.clip_eps = 0.2
        self.k_epochs = 10
        self.lr = 3e-4

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.buffer = Buffer()

    # -------- tanh gaussian policy --------
    def get_dist(self, state):
        mean, value = self.policy(state)
        std = self.policy.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        return dist, value

    def select_action(self, state,writer):
        state = torch.FloatTensor(state).to(device)

        dist, value = self.get_dist(state)
        raw_action = dist.sample()

        action = torch.tanh(raw_action)  # squash to [-1, 1]

        logprob = dist.log_prob(raw_action).sum(dim=-1)

        self.buffer.states.append(state)
        self.buffer.actions.append(raw_action.detach())
        self.buffer.logprobs.append(logprob.detach())
        self.buffer.values.append(value.squeeze().detach())
        # writer.add_scalar("action/norm", raw_action.norm().item(), global_step)
        return action.cpu().numpy()

    # -------- GAE --------
    def compute_gae(self, next_value=0):
        rewards = []
        gae = 0

        values = self.buffer.values + [torch.tensor(next_value, device=device)]

        for i in reversed(range(len(self.buffer.rewards))):
            delta = self.buffer.rewards[i] + self.gamma * values[i+1] * (1 - self.buffer.dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - self.buffer.dones[i]) * gae
            rewards.insert(0, gae + values[i])

        returns = torch.stack(rewards)
        values = torch.stack(self.buffer.values)
        advantages = returns - values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns.detach(), advantages.detach()

    # -------- PPO update --------
    def update(self,writer,global_step):
        states = torch.stack(self.buffer.states).to(device)
        actions = torch.stack(self.buffer.actions).to(device)
        old_logprobs = torch.stack(self.buffer.logprobs).to(device)

        returns, advantages = self.compute_gae()

        for _ in range(self.k_epochs):

            mean, values = self.policy(states)

            std = self.policy.log_std.exp().unsqueeze(0).expand_as(mean)

            new_dist = Normal(mean, std)

            new_logprobs = new_dist.log_prob(actions).sum(dim=-1)
            entropy = new_dist.entropy().sum(dim=-1)

            ratios = torch.exp(new_logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), returns)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()
            writer.add_scalar("loss/actor", actor_loss.item(), global_step)
            writer.add_scalar("loss/critic", critic_loss.item(), global_step)
            writer.add_scalar("loss/entropy", entropy.mean().item(), global_step)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.buffer.clear()


# =========================
# Training Loop
# =========================
def train():
    env = gym.make("HalfCheetah-v4")
    writer = SummaryWriter(log_dir=f"runs/ppo_halfcheetah_{int(time.time())}")
    global_step = 0
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPO(state_dim, action_dim)

    max_episodes = 1000
    step_count = 0
    batch_size = 2048

    for ep in range(max_episodes):

        state, _ = env.reset()
        ep_reward = 0

        while True:

            action = agent.select_action(state,writer)

            next_state, reward, done, truncated, _ = env.step(action)

            agent.buffer.rewards.append(reward)
            agent.buffer.dones.append(done or truncated)

            state = next_state
            ep_reward += reward
            step_count += 1

            if step_count % batch_size == 0:
                agent.update(writer,step_count)

            if done or truncated:
                break

        print(f"Episode {ep} | Reward: {ep_reward:.2f}")


if __name__ == "__main__":
    train()
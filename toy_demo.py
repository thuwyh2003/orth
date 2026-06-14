# # import gymnasium as gym
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # import numpy as np
# # import random
# # from collections import deque

# # # ============================================================
# # # Config
# # # ============================================================

# # ENV_NAME = "LunarLanderContinuous-v2"

# # GAMMA = 0.99
# # GAE_LAMBDA = 0.95

# # LR = 3e-4

# # CLIP_EPS = 0.2

# # ENTROPY_COEF = 0.01
# # VALUE_COEF = 0.5

# # MAX_GRAD_NORM = 0.5

# # ROLLOUT_STEPS = 2048
# # UPDATE_EPOCHS = 10
# # MINIBATCH_SIZE = 256

# # TOTAL_UPDATES = 250

# # SEED = 42

# # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # # ============================================================
# # # Seed
# # # ============================================================

# # def set_seed(seed):
# #     torch.manual_seed(seed)
# #     np.random.seed(seed)
# #     random.seed(seed)


# # # ============================================================
# # # GAE
# # # ============================================================

# # def compute_gae(
# #     rewards,
# #     values,
# #     dones,
# #     next_value,
# #     gamma=0.99,
# #     lam=0.95,
# # ):
# #     advantages = []

# #     gae = 0.0

# #     values = values + [next_value]

# #     for t in reversed(range(len(rewards))):

# #         mask = 1.0 - dones[t]

# #         delta = (
# #             rewards[t]
# #             + gamma * values[t + 1] * mask
# #             - values[t]
# #         )

# #         gae = delta + gamma * lam * mask * gae

# #         advantages.insert(0, gae)

# #     advantages = torch.FloatTensor(advantages)

# #     returns = advantages + torch.FloatTensor(values[:-1])

# #     return advantages, returns


# # # ============================================================
# # # Actor Critic
# # # ============================================================

# # class ActorCritic(nn.Module):

# #     def __init__(self, state_dim, action_dim):
# #         super().__init__()

# #         self.shared = nn.Sequential(
# #             nn.Linear(state_dim, 128),
# #             nn.Tanh(),
# #             nn.Linear(128, 128),
# #             nn.Tanh(),
# #         )

# #         self.actor_mean = nn.Linear(128, action_dim)

# #         self.log_std = nn.Parameter(torch.zeros(action_dim))

# #         self.critic = nn.Linear(128, 1)

# #     def forward(self, x):

# #         feat = self.shared(x)

# #         mean = self.actor_mean(feat)

# #         std = self.log_std.exp().expand_as(mean)

# #         value = self.critic(feat).squeeze(-1)

# #         return mean, std, value

# #     def get_dist(self, x):

# #         mean, std, value = self.forward(x)

# #         dist = torch.distributions.Normal(mean, std)

# #         return dist, mean, value

# #     def sample_action(self, x):

# #         dist, mean, value = self.get_dist(x)

# #         action = dist.rsample()

# #         logp = dist.log_prob(action).sum(dim=-1)

# #         entropy = dist.entropy().sum(dim=-1)

# #         return action, logp, entropy, value, mean, dist


# # # ============================================================
# # # Rollout Buffer
# # # ============================================================

# # class RolloutBuffer:

# #     def __init__(self):

# #         self.states = []
# #         self.actions = []

# #         self.logps = []

# #         self.rewards = []
# #         self.dones = []

# #         self.values = []

# #         self.epsilons = []

# #     def clear(self):
# #         self.__init__()


# # # ============================================================
# # # Standard PPO Loss
# # # ============================================================

# # def standard_ppo_loss(
# #     logp_new,
# #     logp_old,
# #     advantages,
# #     clip_eps=0.2,
# # ):

# #     ratio = torch.exp(logp_new - logp_old)

# #     surr1 = ratio * advantages

# #     surr2 = (
# #         torch.clamp(
# #             ratio,
# #             1 - clip_eps,
# #             1 + clip_eps,
# #         )
# #         * advantages
# #     )

# #     policy_loss = -torch.min(surr1, surr2).mean()

# #     approx_kl = (logp_old - logp_new).mean().item()

# #     clip_frac = (
# #         ((ratio - 1.0).abs() > clip_eps)
# #         .float()
# #         .mean()
# #         .item()
# #     )

# #     return policy_loss, approx_kl, clip_frac


# # # ============================================================
# # # Velocity Weighted PPO Loss
# # # ============================================================

# # def velocity_weighted_ppo_loss(
# #     logp_new,
# #     logp_old,
# #     advantages,
# #     epsilon,
# #     v_theta,
# #     clip_eps=0.2,

# #     # 推荐默认配置
# #     use_direction_only=True,
# #     detach_alignment=True,
# #     squared_alignment=True,
# #     positive_only=False,
# # ):

# #     ratio = torch.exp(logp_new - logp_old)

# #     # --------------------------------------------------------
# #     # velocity normalization
# #     # --------------------------------------------------------

# #     if use_direction_only:

# #         v_norm = (
# #             torch.norm(
# #                 v_theta,
# #                 p=2,
# #                 dim=-1,
# #                 keepdim=True,
# #             )
# #             + 1e-8
# #         )

# #         v_used = v_theta / v_norm

# #     else:

# #         v_used = v_theta

# #     # --------------------------------------------------------
# #     # epsilon projection
# #     # --------------------------------------------------------

# #     alignment = (epsilon * v_used).sum(dim=-1)

# #     # --------------------------------------------------------
# #     # optional positive-only
# #     # --------------------------------------------------------

# #     if positive_only:
# #         alignment = torch.relu(alignment)

# #     # --------------------------------------------------------
# #     # squared alignment
# #     # --------------------------------------------------------

# #     if squared_alignment:
# #         alignment = alignment.pow(2)

# #     # --------------------------------------------------------
# #     # stop-gradient (推荐)
# #     # --------------------------------------------------------

# #     if detach_alignment:
# #         alignment = alignment.detach()

# #     # --------------------------------------------------------
# #     # PPO surrogate
# #     # --------------------------------------------------------

# #     surr1 = ratio * alignment * advantages

# #     surr2 = (
# #         torch.clamp(
# #             ratio,
# #             1 - clip_eps,
# #             1 + clip_eps,
# #         )
# #         * alignment
# #         * advantages
# #     )

# #     policy_loss = -torch.min(surr1, surr2).mean()

# #     approx_kl = (logp_old - logp_new).mean().item()

# #     clip_frac = (
# #         ((ratio - 1.0).abs() > clip_eps)
# #         .float()
# #         .mean()
# #         .item()
# #     )

# #     alignment_mean = alignment.mean().item()

# #     return (
# #         policy_loss,
# #         approx_kl,
# #         clip_frac,
# #         alignment_mean,
# #     )


# # # ============================================================
# # # Train
# # # ============================================================

# # def train(
# #     use_velocity_weighted=True,
# #     seed=42,
# # ):

# #     set_seed(seed)

# #     env = gym.make(ENV_NAME)

# #     env.reset(seed=seed)

# #     state_dim = env.observation_space.shape[0]

# #     action_dim = env.action_space.shape[0]

# #     model = ActorCritic(
# #         state_dim,
# #         action_dim,
# #     ).to(DEVICE)

# #     optimizer = optim.Adam(
# #         model.parameters(),
# #         lr=LR,
# #     )

# #     buffer = RolloutBuffer()

# #     # --------------------------------------------------------
# #     # logging
# #     # --------------------------------------------------------

# #     episode_rewards = deque(maxlen=50)

# #     total_env_steps = 0

# #     total_episodes = 0

# #     best_avg_reward = -1e9

# #     # --------------------------------------------------------
# #     # initial reset
# #     # --------------------------------------------------------

# #     state, _ = env.reset()

# #     current_episode_reward = 0.0

# #     # ========================================================
# #     # UPDATE LOOP
# #     # ========================================================

# #     for update in range(1, TOTAL_UPDATES + 1):

# #         buffer.clear()

# #         # ====================================================
# #         # COLLECT ROLLOUT
# #         # ====================================================

# #         for step in range(ROLLOUT_STEPS):

# #             state_t = (
# #                 torch.FloatTensor(state)
# #                 .unsqueeze(0)
# #                 .to(DEVICE)
# #             )

# #             with torch.no_grad():

# #                 (
# #                     action,
# #                     logp,
# #                     entropy,
# #                     value,
# #                     mean,
# #                     dist,
# #                 ) = model.sample_action(state_t)

# #             action_np = action.squeeze(0).cpu().numpy()

# #             next_state, reward, terminated, truncated, _ = env.step(
# #                 action_np
# #             )

# #             done = terminated or truncated

# #             # =================================================
# #             # epsilon
# #             # =================================================

# #             epsilon = (
# #                 (action - mean)
# #                 / (dist.scale + 1e-8)
# #             ).squeeze(0)

# #             # =================================================
# #             # store rollout
# #             # =================================================

# #             buffer.states.append(state)

# #             buffer.actions.append(
# #                 action.squeeze(0).cpu()
# #             )

# #             buffer.logps.append(
# #                 logp.squeeze(0).cpu()
# #             )

# #             buffer.rewards.append(reward)

# #             buffer.dones.append(float(done))

# #             buffer.values.append(value.item())

# #             buffer.epsilons.append(
# #                 epsilon.cpu()
# #             )

# #             # =================================================
# #             # update env state
# #             # =================================================

# #             state = next_state

# #             current_episode_reward += reward

# #             total_env_steps += 1

# #             # =================================================
# #             # episode done
# #             # =================================================

# #             if done:

# #                 episode_rewards.append(
# #                     current_episode_reward
# #                 )

# #                 total_episodes += 1

# #                 current_episode_reward = 0.0

# #                 state, _ = env.reset()

# #         # ====================================================
# #         # Bootstrap value
# #         # ====================================================

# #         with torch.no_grad():

# #             next_state_t = (
# #                 torch.FloatTensor(state)
# #                 .unsqueeze(0)
# #                 .to(DEVICE)
# #             )

# #             _, _, next_value = model.forward(next_state_t)

# #             next_value = next_value.item()

# #         # ====================================================
# #         # GAE
# #         # ====================================================

# #         advantages, returns = compute_gae(
# #             rewards=buffer.rewards,
# #             values=buffer.values,
# #             dones=buffer.dones,
# #             next_value=next_value,
# #             gamma=GAMMA,
# #             lam=GAE_LAMBDA,
# #         )

# #         advantages = (
# #             advantages - advantages.mean()
# #         ) / (advantages.std() + 1e-8)

# #         # ====================================================
# #         # Tensorize
# #         # ====================================================

# #         states_t = torch.FloatTensor(
# #             np.array(buffer.states)
# #         ).to(DEVICE)

# #         actions_t = torch.stack(
# #             buffer.actions
# #         ).to(DEVICE)

# #         old_logps_t = torch.stack(
# #             buffer.logps
# #         ).to(DEVICE)

# #         epsilons_t = torch.stack(
# #             buffer.epsilons
# #         ).to(DEVICE)

# #         advantages_t = advantages.to(DEVICE)

# #         returns_t = returns.to(DEVICE)

# #         dataset_size = states_t.shape[0]

# #         # ====================================================
# #         # PPO UPDATE
# #         # ====================================================

# #         policy_loss_value = 0.0
# #         value_loss_value = 0.0
# #         entropy_value = 0.0

# #         approx_kl_value = 0.0
# #         clip_frac_value = 0.0

# #         alignment_value = 0.0

# #         for epoch in range(UPDATE_EPOCHS):

# #             indices = np.random.permutation(dataset_size)

# #             for start in range(
# #                 0,
# #                 dataset_size,
# #                 MINIBATCH_SIZE,
# #             ):

# #                 end = start + MINIBATCH_SIZE

# #                 mb_idx = indices[start:end]

# #                 mb_states = states_t[mb_idx]

# #                 mb_actions = actions_t[mb_idx]

# #                 mb_old_logps = old_logps_t[mb_idx]

# #                 mb_advantages = advantages_t[mb_idx]

# #                 mb_returns = returns_t[mb_idx]

# #                 mb_eps = epsilons_t[mb_idx]

# #                 # =============================================
# #                 # current policy
# #                 # =============================================

# #                 dist, mean_new, value_new = (
# #                     model.get_dist(mb_states)
# #                 )

# #                 logp_new = dist.log_prob(
# #                     mb_actions
# #                 ).sum(dim=-1)

# #                 entropy = (
# #                     dist.entropy()
# #                     .sum(dim=-1)
# #                     .mean()
# #                 )

# #                 # =============================================
# #                 # policy loss
# #                 # =============================================

# #                 if use_velocity_weighted:

# #                     (
# #                         policy_loss,
# #                         approx_kl,
# #                         clip_frac,
# #                         alignment_mean,
# #                     ) = velocity_weighted_ppo_loss(
# #                         logp_new=logp_new,
# #                         logp_old=mb_old_logps,
# #                         advantages=mb_advantages,
# #                         epsilon=mb_eps,
# #                         v_theta=mean_new,
# #                         clip_eps=CLIP_EPS,
# #                     )

# #                 else:

# #                     (
# #                         policy_loss,
# #                         approx_kl,
# #                         clip_frac,
# #                     ) = standard_ppo_loss(
# #                         logp_new=logp_new,
# #                         logp_old=mb_old_logps,
# #                         advantages=mb_advantages,
# #                         clip_eps=CLIP_EPS,
# #                     )

# #                     alignment_mean = 0.0

# #                 # =============================================
# #                 # value loss
# #                 # =============================================

# #                 value_loss = (
# #                     (value_new - mb_returns)
# #                     .pow(2)
# #                     .mean()
# #                 )

# #                 # =============================================
# #                 # total loss
# #                 # =============================================

# #                 loss = (
# #                     policy_loss
# #                     + VALUE_COEF * value_loss
# #                     - ENTROPY_COEF * entropy
# #                 )

# #                 optimizer.zero_grad()

# #                 loss.backward()

# #                 grad_norm = torch.nn.utils.clip_grad_norm_(
# #                     model.parameters(),
# #                     MAX_GRAD_NORM,
# #                 )

# #                 optimizer.step()

# #                 # =============================================
# #                 # logging
# #                 # =============================================

# #                 policy_loss_value = policy_loss.item()

# #                 value_loss_value = value_loss.item()

# #                 entropy_value = entropy.item()

# #                 approx_kl_value = approx_kl

# #                 clip_frac_value = clip_frac

# #                 alignment_value = alignment_mean

# #         # ====================================================
# #         # Update logging
# #         # ====================================================

# #         avg_reward = (
# #             np.mean(episode_rewards)
# #             if len(episode_rewards) > 0
# #             else 0.0
# #         )

# #         best_avg_reward = max(
# #             best_avg_reward,
# #             avg_reward,
# #         )

# #         print(
# #             f"\n"
# #             f"========================================================\n"
# #             f"Update            : {update:4d}/{TOTAL_UPDATES}\n"
# #             f"Method            : "
# #             f"{'VelocityWeightedPPO' if use_velocity_weighted else 'StandardPPO'}\n"
# #             f"Env Steps         : {total_env_steps:8d}\n"
# #             f"Episodes          : {total_episodes:6d}\n"
# #             f"Avg Reward (50ep) : {avg_reward:8.2f}\n"
# #             f"Best Avg Reward   : {best_avg_reward:8.2f}\n"
# #             f"\n"
# #             f"Policy Loss       : {policy_loss_value:8.4f}\n"
# #             f"Value Loss        : {value_loss_value:8.4f}\n"
# #             f"Entropy           : {entropy_value:8.4f}\n"
# #             f"Approx KL         : {approx_kl_value:8.6f}\n"
# #             f"Clip Fraction     : {clip_frac_value:8.4f}\n"
# #             f"Alignment Mean    : {alignment_value:8.4f}\n"
# #             f"Grad Norm         : {grad_norm:8.4f}\n"
# #             f"========================================================"
# #         )

# #     env.close()

# #     return best_avg_reward


# # # ============================================================
# # # Main
# # # ============================================================

# # if __name__ == "__main__":

# #     print("\n==============================")
# #     print(" Standard PPO")
# #     print("==============================")

# #     std_reward = train(
# #         use_velocity_weighted=False,
# #         seed=SEED,
# #     )

# #     print("\n==============================")
# #     print(" Velocity Weighted PPO")
# #     print("==============================")

# #     vel_reward = train(
# #         use_velocity_weighted=True,
# #         seed=SEED,
# #     )

# #     print("\n==============================")
# #     print(" Final Result")
# #     print("==============================")

# #     print(f"Standard PPO Best Avg Reward : {std_reward:.2f}")

# #     print(f"Velocity Weighted PPO Best Avg Reward : {vel_reward:.2f}")
    

# import gymnasium as gym
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import random
# from collections import deque
# from torch.utils.tensorboard import SummaryWriter   # ⭐ NEW


# # ============================================================
# # Config
# # ============================================================

# # ENV_NAME = "LunarLanderContinuous-v2"
# ENV_NAME = "Walker2d-v4"

# GAMMA = 0.99
# GAE_LAMBDA = 0.95

# LR = 3e-4

# CLIP_EPS = 0.2

# ENTROPY_COEF = 0.01
# VALUE_COEF = 0.5

# MAX_GRAD_NORM = 0.5

# ROLLOUT_STEPS = 2048
# UPDATE_EPOCHS = 10
# MINIBATCH_SIZE = 256

# TOTAL_UPDATES = 250

# SEED = 42

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # ============================================================
# # TensorBoard helper ⭐ NEW
# # ============================================================

# def make_writer(use_velocity):
#     return SummaryWriter(
#         log_dir=f"runs/{'velocity_ppo' if use_velocity else 'ppo'}_seed{SEED}_env{ENV_NAME}"
#     )


# # ============================================================
# # Seed
# # ============================================================

# def set_seed(seed):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)


# # ============================================================
# # GAE
# # ============================================================

# def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):

#     advantages = []
#     gae = 0.0
#     values = values + [next_value]

#     for t in reversed(range(len(rewards))):

#         mask = 1.0 - dones[t]

#         delta = rewards[t] + gamma * values[t + 1] * mask - values[t]

#         gae = delta + gamma * lam * mask * gae

#         advantages.insert(0, gae)

#     advantages = torch.FloatTensor(advantages)
#     returns = advantages + torch.FloatTensor(values[:-1])

#     return advantages, returns


# # ============================================================
# # Actor Critic
# # ============================================================

# class ActorCritic(nn.Module):

#     def __init__(self, state_dim, action_dim):
#         super().__init__()

#         self.shared = nn.Sequential(
#             nn.Linear(state_dim, 128),
#             nn.Tanh(),
#             nn.Linear(128, 128),
#             nn.Tanh(),
#         )

#         self.actor_mean = nn.Linear(128, action_dim)
#         self.log_std = nn.Parameter(torch.zeros(action_dim))
#         self.critic = nn.Linear(128, 1)

#     def forward(self, x):

#         feat = self.shared(x)
#         mean = self.actor_mean(feat)
#         std = self.log_std.exp().expand_as(mean)
#         value = self.critic(feat).squeeze(-1)

#         return mean, std, value

#     def get_dist(self, x):
#         mean, std, value = self.forward(x)
#         dist = torch.distributions.Normal(mean, std)
#         return dist, mean, value

#     def sample_action(self, x):
#         dist, mean, value = self.get_dist(x)
#         # action = dist.rsample()
#         action = torch.tanh(dist.rsample())
#         logp = dist.log_prob(action).sum(dim=-1)
#         entropy = dist.entropy().sum(dim=-1)
#         return action, logp, entropy, value, mean, dist


# # ============================================================
# # PPO Loss
# # ============================================================

# def standard_ppo_loss(logp_new, logp_old, advantages, clip_eps=0.2):

#     ratio = torch.exp(logp_new - logp_old)

#     surr1 = ratio * advantages
#     surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages

#     policy_loss = -torch.min(surr1, surr2).mean()

#     approx_kl = (logp_old - logp_new).mean().item()

#     clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()

#     return policy_loss, approx_kl, clip_frac


# # ============================================================
# # Velocity PPO Loss
# # ============================================================

# def velocity_weighted_ppo_loss(
#     logp_new,
#     logp_old,
#     advantages,
#     epsilon,
#     v_theta,
#     clip_eps=0.2,
# ):

#     ratio = torch.exp(logp_new - logp_old)

#     v_used = v_theta / (v_theta.norm(dim=-1, keepdim=True) + 1e-8)

#     alignment = (epsilon * v_used).sum(dim=-1)
#     # alignment = alignment.pow(2)
#     alignment = alignment.detach()

#     # surr1 = ratio * alignment * advantages
#     surr1 = alignment * advantages
#     # surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * alignment * advantages

#     # policy_loss = -torch.min(surr1, surr2).mean()
#     policy_loss = -surr1.mean()
#     approx_kl = (logp_old - logp_new).mean().item()
#     clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()

#     alignment_mean = alignment.mean().item()

#     return policy_loss, approx_kl, clip_frac, alignment_mean


# # ============================================================
# # Train
# # ============================================================

# def train(use_velocity_weighted=True, seed=42):

#     set_seed(seed)

#     env = gym.make(ENV_NAME)
#     env.reset(seed=seed)

#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.shape[0]

#     model = ActorCritic(state_dim, action_dim).to(DEVICE)
#     optimizer = optim.Adam(model.parameters(), lr=LR)

#     # ⭐ TensorBoard writer
#     writer = make_writer(use_velocity_weighted)

#     episode_rewards = deque(maxlen=50)

#     total_env_steps = 0
#     total_episodes = 0
#     best_avg_reward = -1e9

#     state, _ = env.reset()
#     current_episode_reward = 0.0

#     for update in range(1, TOTAL_UPDATES + 1):

#         buffer = {
#             "states": [],
#             "actions": [],
#             "logps": [],
#             "rewards": [],
#             "dones": [],
#             "values": [],
#             "eps": [],
#         }

#         # ================= rollout =================

#         for step in range(ROLLOUT_STEPS):

#             s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

#             with torch.no_grad():
#                 action, logp, entropy, value, mean, dist = model.sample_action(s)

#             a = action.squeeze(0).cpu().numpy()

#             next_state, reward, terminated, truncated, _ = env.step(a)
#             done = terminated or truncated

#             epsilon = ((action - mean) / (dist.scale + 1e-8)).squeeze(0)

#             buffer["states"].append(state)
#             buffer["actions"].append(action.squeeze(0).cpu())
#             buffer["logps"].append(logp.squeeze(0).cpu())
#             buffer["rewards"].append(reward)
#             buffer["dones"].append(float(done))
#             buffer["values"].append(value.item())
#             buffer["eps"].append(epsilon.cpu())

#             state = next_state
#             current_episode_reward += reward
#             total_env_steps += 1

#             if done:
#                 episode_rewards.append(current_episode_reward)
#                 total_episodes += 1
#                 current_episode_reward = 0.0
#                 state, _ = env.reset()

#         # ================= bootstrap =================

#         with torch.no_grad():
#             s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
#             _, _, next_value = model.forward(s)
#             next_value = next_value.item()

#         # ================= GAE =================

#         advantages, returns = compute_gae(
#             buffer["rewards"],
#             buffer["values"],
#             buffer["dones"],
#             next_value,
#             GAMMA,
#             GAE_LAMBDA,
#         )

#         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

#         # ================= tensor =================

#         states = torch.FloatTensor(np.array(buffer["states"])).to(DEVICE)
#         actions = torch.stack(buffer["actions"]).to(DEVICE)
#         old_logps = torch.stack(buffer["logps"]).to(DEVICE)
#         eps = torch.stack(buffer["eps"]).to(DEVICE)

#         advantages = advantages.to(DEVICE)
#         returns = returns.to(DEVICE)

#         # ================= update =================

#         policy_loss_val = 0
#         value_loss_val = 0
#         entropy_val = 0
#         alignment_val = 0
#         kl_val = 0
#         clip_val = 0

#         for epoch in range(UPDATE_EPOCHS):

#             idx = np.random.permutation(len(states))

#             for start in range(0, len(states), MINIBATCH_SIZE):

#                 mb = idx[start:start+MINIBATCH_SIZE]

#                 mb_s = states[mb]
#                 mb_a = actions[mb]
#                 mb_lp = old_logps[mb]
#                 mb_adv = advantages[mb]
#                 mb_ret = returns[mb]
#                 mb_eps = eps[mb]

#                 dist, mean, value = model.get_dist(mb_s)

#                 logp = dist.log_prob(mb_a).sum(dim=-1)
#                 entropy = dist.entropy().sum(dim=-1).mean()

#                 if use_velocity_weighted:

#                     policy_loss, kl, clip, align = velocity_weighted_ppo_loss(
#                         logp, mb_lp, mb_adv, mb_eps, mean
#                     )
#                     alignment_val = align

#                 else:

#                     policy_loss, kl, clip = standard_ppo_loss(
#                         logp, mb_lp, mb_adv
#                     )
#                     alignment_val = 0.0

#                 value_loss = (value - mb_ret).pow(2).mean()

#                 loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

#                 optimizer.zero_grad()
#                 loss.backward()

#                 grad_norm = torch.nn.utils.clip_grad_norm_(
#                     model.parameters(),
#                     MAX_GRAD_NORM,
#                 )

#                 optimizer.step()

#                 policy_loss_val = policy_loss.item()
#                 value_loss_val = value_loss.item()
#                 entropy_val = entropy.item()
#                 kl_val = kl
#                 clip_val = clip

#         # ================= logging =================

#         avg_reward = np.mean(episode_rewards) if len(episode_rewards) > 0 else 0.0

#         best_avg_reward = max(best_avg_reward, avg_reward)

#         step = update

#         writer.add_scalar("reward/avg50", avg_reward, step)
#         writer.add_scalar("reward/best", best_avg_reward, step)

#         writer.add_scalar("loss/policy", policy_loss_val, step)
#         writer.add_scalar("loss/value", value_loss_val, step)

#         writer.add_scalar("stats/entropy", entropy_val, step)
#         writer.add_scalar("stats/kl", kl_val, step)
#         writer.add_scalar("stats/clip", clip_val, step)
#         writer.add_scalar("stats/grad_norm", grad_norm.item(), step)

#         writer.add_scalar("geometry/alignment", alignment_val, step)

#         print(
#             f"[{update}] reward={avg_reward:.1f} "
#             f"align={alignment_val:.3f} "
#             f"kl={kl_val:.5f}"
#         )

#     writer.close()
#     env.close()

#     return best_avg_reward


# # ============================================================
# # Run
# # ============================================================

# if __name__ == "__main__":

    

#     print("Velocity PPO")
#     vel = train(True)
    
#     print("Standard PPO")
#     std = train(False)

#     print("Final:", std, vel)




import torch
# load_ckpt = torch.load("/home/wyh/RLinf/logs/20260526-21:03:57-maniskill_ppo_openpi_pi05_baseline/maniskill_ppo_openpi_pi05_baseline/checkpoints/global_step_150/actor/model_state_dict/full_weights.pt",map_location="cpu")
# print(load_ckpt.keys())
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import Metadata

reader = FileSystemReader(
    "/home/wyh/RLinf/logs/20260526-21:03:57-maniskill_ppo_openpi_pi05_baseline/maniskill_ppo_openpi_pi05_baseline/checkpoints/global_step_150/actor/dcp_checkpoint"
)
# "/home/wyh/RLinf/logs/20260527-03:10:00-maniskill_ppo_openpi_pi05_baseline/maniskill_ppo_openpi_pi05_baseline/checkpoints/global_step_250/actor/dcp_checkpoint"
metadata = reader.read_metadata()
for name in metadata.state_dict_metadata.keys():
    if "optimizers.param_groups.0.lr" in name:
        print(metadata.state_dict_metadata[name])

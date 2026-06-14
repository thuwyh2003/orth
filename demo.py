from email import generator
import torch
import random

# denoise_inds=torch.tensor([random.randint(0, 10)] * 11)
# denoise_inds = denoise_inds[None].repeat(5, 1)
# print(denoise_inds)
# num_steps=4
# bsize=5
# num_steps=torch.tensor([num_steps]*num_steps).repeat(bsize,1)
# print(num_steps)

# denoise_inds=torch.tensor([[3, 3, 3, 3],
#                             [1, 1, 1, 1],
#                             [3, 3, 3, 3],
#                             [3, 3, 3, 3],
#                             [2, 2, 2, 2],
#                             [1, 1, 1, 1],
#                             [1, 1, 1, 1],
#                             [2, 2, 2, 2],
#                             [3, 3, 3, 3],
#                             [3, 3, 3, 3],
#                             [3, 3, 3, 3],
#                             [3, 3, 3, 3]])

# # print(denoise_inds/4)
# loss=torch.rand((12,5,7))
# scale=torch.sqrt((torch.ones_like(denoise_inds)-denoise_inds/4)/(denoise_inds/4))*torch.sqrt(torch.tensor([1/4]))
# new_scale=scale[:,:1].unsqueeze(2).repeat(1,loss.shape[1],loss.shape[2])
# print(torch.sqrt(torch.tensor(1.0 / 4, device=denoise_inds.device)))


# denoise_steps=4
# timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps)
# timesteps = torch.cat([timesteps, torch.tensor([0.0])])
# print(timesteps)

# sigmas=torch.sqrt(timesteps/(1 - torch.where(timesteps == 1, timesteps[1], timesteps)))[:-1]
# print(sigmas)


# a=[1,2,3,4,5]

# print(denoise_inds[0,0].item())


# noise=torch.rand((6,8,32))
# v=torch.ones_like(noise)
# def project_orthogonal_batch(g,v,eps=1e-8):
#     """
#     this function is used to project vector g to the orthogonal direction of vector v
#     """
#     dot = (g*v).sum(dim=-1,keepdim=True)
#     norm=(v*v).sum(dim=-1,keepdim=True).clamp(min=eps)
#     proj=g-(dot/norm)*v
#     return proj
# proj=project_orthogonal_batch(noise,v)
# print(proj)
# check = (proj * v).sum(dim=-1)
# print(check)


# def build_t(K, device):
#     return torch.linspace(1, 1/K, K, device=device)

# def compute_weight(K, device, a=1.0):
#     t = build_t(K, device)                 # [K]
#     delta_t = 1.0 / K

#     sigma = a * torch.sqrt(t / (1 - torch.where(t==1,t[1],t)))    # [K]
#     w = sigma * torch.sqrt(torch.tensor(delta_t))        # [K]

#     return w

# def normalize_weight(w, eps=1e-8):
#     return w / (w.mean() + eps)
# def apply_denoise_boost(w, denoise_inds, boost=1.0):
#     """
#     w: [K]
#     denoise_inds: [B]
#     """
#     B = denoise_inds.shape[0]
#     K = w.shape[0]
#     return w[denoise_inds]
# policy_loss=torch.rand((12,5,7))
# B, K, D = policy_loss.shape

# w = compute_weight(K, policy_loss.device)   # [K]
# print("w",w)
# w = normalize_weight(w)                     # [K]
# print("w norm",w)
# # reshape成 [1, K, 1]，广播到 [B, K, D]


# denoise_inds=torch.randint(0,4,(12,4))
# print(denoise_inds)
# denoise_inds=denoise_inds[:,0]
# print(denoise_inds)

# w=apply_denoise_boost(w,denoise_inds)
# print("w apply",w)
# ploicy_liss=policy_loss*(w.unsqueeze(1).unsqueeze(1))

# g=torch.Generator()
# g.manual_seed(1234)
# shuffle_id=torch.randperm(12,generator=g)
# # print(shuffle_id)
# # print(policy_loss)
# # print(policy_loss[shuffle_id])
# x=torch.randint(1,4,(3,1,1))
# x=x.repeat(1,4,10)
# print(x)
# print(x/x.float().mean())
# print(x/x.float().flatten().mean())
# print(x.shape)
# x=x.unsqueeze(1)
# print(x.shape)


# import os
# import re
# import numpy as np
# import matplotlib.pyplot as plt

# # ========= 你的多个实验目录 =========
# exp_dirs = {
#     # "orth_2": "/home/wyh/RLinf/logs/eval/maniskill_ppo_openpi_pi05_orth_2",
#     # "baseline": "/home/wyh/RLinf/logs/eval/maniskill_ppo_openpi_pi05_baseline",
#     "Pi_rl_fullenv": "/home/wyh/RLinf/logs/eval/maniskill_ppo_openpi_pi05_baseline_fullenv",
    
#     # "orth": "/home/wyh/RLinf/logs/eval/maniskill_ppo_openpi_pi05_orth",
#     # "reweight": "/home/wyh/RLinf/logs/eval/maniskill_ppo_openpi_pi05_reweight",
#     # "orth_3": "/home/wyh/RLinf/logs/eval/maniskill_ppo_openpi_pi05_orth_3",
#     # "reweight_2":"/home/wyh/RLinf/logs/eval/maniskill_ppo_openpi_pi05_reweight2",
#     # "orth_4": "/home/wyh/RLinf/logs/eval/maniskill_ppo_openpi_pi05_orth4",
#     # "orth_5": "/home/wyh/RLinf/logs/eval/maniskill_ppo_openpi_pi05_orth_5",
#     # "orth_3_fullenv": "/home/wyh/RLinf/logs/eval/maniskill_ppo_openpi_pi05_orth_3_fullenv",
#     # "orth_6_fullenv":"/home/wyh/RLinf/logs/eval/maniskill_ppo_openpi_pi05_orth_6_fullenv",
#     "Ours_fullenv":"/home/wyh/RLinf/logs/eval/maniskill_ppo_openpi_pi05_orth_7_fullenv",
#     # "orth_8_fullenv":"/home/wyh/RLinf/logs/eval/maniskill_ppo_openpi_pi05_orth_8_fullenv",
    
    
#     # 可以继续加
# }

# success_pattern = re.compile(r"'eval/success_once': array\(([\d\.]+)")

# def parse_one_exp(root_dir):
#     task_success = {}

#     for subdir in os.listdir(root_dir):
#         subdir_path = os.path.join(root_dir, subdir)
#         if not os.path.isdir(subdir_path):
#             continue

#         log_path = os.path.join(subdir_path, "run_ppo.log")
#         if not os.path.exists(log_path):
#             continue

#         # ===== 提取任务名 =====
#         try:
#             parts = subdir.split("-")
#             task_full = parts[2]+parts[4]
#             task_name = re.sub(r"PutOnPlateInScene\d+", "", task_full)
#         except:
#             continue

#         # ===== 提取 success（取最后一个）=====
#         success = None
#         with open(log_path, "r") as f:
#             for line in f:
#                 match = success_pattern.search(line)
#                 if match:
#                     success = float(match.group(1))

#         if success is not None:
#             task_success[task_name] = success

#     return task_success


# # ========= 解析所有实验 =========
# all_results = {}
# all_tasks = set()

# for name, path in exp_dirs.items():
#     res = parse_one_exp(path)
#     all_results[name] = res
#     all_tasks.update(res.keys())

# # ========= 统一任务顺序 =========
# tasks = sorted(list(all_tasks))
# num_vars = len(tasks)

# angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
# angles += angles[:1]

# # ========= 画图 =========
# plt.figure(figsize=(8, 8))
# ax = plt.subplot(111, polar=True)

# for exp_name, res in all_results.items():
#     values = []
#     for t in tasks:
#         values.append(res.get(t, 0.0))  # 没有的填0（或 np.nan）

#     values += values[:1]

#     ax.plot(angles, values, linewidth=2, label=exp_name)
#     ax.fill(angles, values, alpha=0.1)

# # ========= 美化 =========
# ax.set_xticks(angles[:-1])
# ax.set_xticklabels(tasks, fontsize=9)

# ax.set_ylim(0, 1)

# plt.title("OOD-Experiment Success Rate Radar", size=15)
# plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

# plt.tight_layout()
# plt.savefig("multi_radar.png", dpi=200)
# plt.show()

# print("Done. Saved to multi_radar.png")




import os
import re
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 配置区 ======================
exp_dirs = {
    "algorithm_1": "/home/wyh/RLinf/logs/eval/maniskill_ppo_openpi_pi05_baseline",
    "algorithm_2": "/home/wyh/RLinf/logs/eval/maniskill_ppo_openpi_pi05_orth_7_fullenv",
    "algorithm_3": "/home/wyh/RLinf/logs/eval/maniskill_ppo_openpi_pi05_reweight2"
    # 可继续添加你的其他实验
}

# ================== 手动输入 Baseline ==================
baseline = {
    "Instruct-test": 0.48,
    "VisionImage-test": 0.42,
    "VisionWhole03-test": 0.4,
    "VisionWhole05-test": 0.3,
    "VisionTexture03-test": 0.38,
    "VisionTexture05-test": 0.35,
    "MultiPlate-train":0.09,
    "MultiPlate-test":0.12,
    "MultiCarrot-train":0.3,
    "MultiCarrot-test":0.18,
    "EEPose-test":0.36,
    "PositionChangeTo-test":0.17,
    "Position-test":0.30,
    "Carrot-train":0.5,
    "Carrot-test":0.4,
    "Plate-test":0.32,
    # 根据你的实际任务继续添加或修改
}

# ====================== 解析函数 ======================
success_pattern = re.compile(r"'eval/success_once': array\(([\d\.]+)")

def parse_one_exp(root_dir):
    task_success = {}
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        log_path = os.path.join(subdir_path, "run_ppo.log")
        if not os.path.exists(log_path):
            continue

        try:
            parts = subdir.split("-")
            
            task_full = parts[2].split("25")[-1] + "-" + parts[4]
            task_name = re.sub(r"PutOnPlateInScene\d+", "PutOnPlate", task_full)
            # task_name = re.sub(r"\d+", "", task_name)   # 清理数字
        except:
            continue

        success = None
        with open(log_path, "r") as f:
            for line in f:
                match = success_pattern.search(line)
                if match:
                    success = float(match.group(1))
        if success is not None:
            task_success[task_name] = success
    return task_success


# ====================== 解析数据 ======================
all_results = {}
all_tasks = set()

for name, path in exp_dirs.items():
    res = parse_one_exp(path)
    all_results[name] = res
    all_tasks.update(res.keys())

# 加入 Baseline
all_results["ODE"] = baseline
all_tasks.update(baseline.keys())

tasks = sorted(list(all_tasks))
num_vars = len(tasks)

# ====================== 雷达图角度 ======================
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# ====================== 绘图 ======================
fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)

# 漂亮的颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
styles = ['-', '-', '-', '--']   # Baseline 用虚线

for i, (exp_name, res) in enumerate(all_results.items()):
    values = [res.get(t, 0.0) for t in tasks]
    values += values[:1]
    
    color = colors[i % len(colors)]
    linestyle = '--' if exp_name == "Baseline" else '-'
    linewidth = 3.5 if exp_name == "Baseline" else 2.5
    
    ax.plot(angles, values, linewidth=linewidth, linestyle=linestyle,
            color=color, label=exp_name, marker='o', markersize=5)
    
    alpha = 0.25 if exp_name != "Baseline" else 0.12
    ax.fill(angles, values, color=color, alpha=alpha)

# ====================== 美化设置 ======================
ax.set_xticks(angles[:-1])
ax.set_xticklabels(tasks, fontsize=11, fontweight='bold')

ax.set_ylim(0, 1.0)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10, fontweight='bold')

# 网格
ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.7)

# plt.title("OOD Success Rate Comparison (Radar Chart)", 
#           fontsize=18, fontweight='bold', pad=25)

# 图例优化
plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05), 
           fontsize=12, frameon=True, shadow=True)

plt.tight_layout()
plt.savefig("OOD_Success_Rate_Radar_Comparison.pdf", dpi=400, bbox_inches='tight')
plt.savefig("OOD_Success_Rate_Radar_Comparison.png", dpi=400, bbox_inches='tight')

plt.show()

print("✅ 精致雷达图绘制完成！")
print("已保存为：OOD_Success_Rate_Radar_Comparison.pdf 和 .png")
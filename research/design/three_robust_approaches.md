# 三种鲁棒RL方案详细技术报告

## 问题背景

风电场控制中存在"仿真器阶梯"问题：
- FLORIS (稳态, ~10ms) → WFSim (2D NS) → FAST.Farm (气弹) → LES/CFD → 真实风场
- FLORIS训练的策略部署到FAST.Farm，性能下降15-21%
- 需要策略对仿真器模型误差具有鲁棒性

---

## 方案A：对抗域随机化 + MAPPO（RARL框架）

### A.1 核心思想

将策略训练建模为**两人零和博弈**：
- **主体(Protagonist)**：风电场控制策略 μ_θ，目标是最大化发电量
- **对抗者(Adversary)**：扰动策略 ν_φ，目标是最小化发电量

训练过程中，对抗者施加物理上合理的扰动（尾流参数偏差、风速噪声等），迫使主体策略学到对模型不确定性鲁棒的控制。

### A.2 数学建模

#### 极小极大博弈

```
max_θ  min_φ  E_{τ~(μ_θ, ν_φ)} [ Σ_t γ^t · r(s_t, a_t^μ, a_t^ν) ]
```

其中：
- a_t^μ ∈ A_μ：主体动作（偏航角变化）
- a_t^ν ∈ A_ν：对抗者动作（环境扰动）
- 环境转移：s_{t+1} = f(s_t, a_t^μ, a_t^ν)

#### 对抗者的动作空间

对抗者不是随意扰动，而是在**物理合理范围内**施加最坏情况扰动：

```
对抗者动作 a_t^ν = [Δk_wake, ΔTI, Δv_wind, Δθ_wind, Δk_yaw]

约束（物理合理性）：
  尾流衰减率扰动：    Δk_wake  ∈ [-0.02, +0.02]  (nominal k ≈ 0.04)
  湍流强度扰动：      ΔTI      ∈ [-0.03, +0.03]  (nominal TI ≈ 0.06)
  风速扰动：          Δv_wind  ∈ [-1.0, +1.0] m/s
  风向扰动：          Δθ_wind  ∈ [-5°, +5°]
  偏航有效性扰动：    Δk_yaw   ∈ [-0.15, 0]       (只退化，不增强)

变化率约束（防止非物理的瞬变）：
  |d(Δk_wake)/dt| ≤ 0.005/step
  |d(Δv_wind)/dt| ≤ 0.3 m/s/step
```

#### 交替训练算法

```
for iteration = 1, 2, ..., N_iter:
    
    # Phase 1: 固定对抗者，训练主体（PPO更新）
    for n_μ steps:
        收集 rollout: τ ~ (μ_θ, ν_φ_fixed)
        计算优势: Â_t = r_t + γV(s_{t+1}) - V(s_t)
        PPO-Clip更新:
            L^CLIP(θ) = E[min(ρ_t·Â_t, clip(ρ_t, 1-ε, 1+ε)·Â_t)]
            其中 ρ_t = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)
    
    # Phase 2: 固定主体，训练对抗者（PPO更新，reward取反）
    for n_ν steps:
        收集 rollout: τ ~ (μ_θ_fixed, ν_φ)
        计算优势: Â_t = -r_t + γV^ν(s_{t+1}) - V^ν(s_t)  # reward取反
        PPO-Clip更新对抗者参数 φ
```

#### 域随机化层（叠加在RARL之上）

除了可学习的对抗者，还在每个episode开始时随机化FLORIS环境参数：

```
每个episode开始时：
  k_wake     ~ Uniform[0.03, 0.07]        # 尾流衰减率
  TI_ambient ~ Uniform[0.04, 0.12]        # 环境湍流
  C_t_scale  ~ Uniform[0.85, 1.05]        # 推力系数缩放
  yaw_eff    ~ Uniform[0.80, 1.00]        # 偏航有效性
  v_wind     ~ Uniform[5.0, 15.0]         # 风速
  θ_wind     ~ Uniform[250°, 290°]        # 风向
```

### A.3 MAPPO集成（多智能体版本）

```
架构: Centralized Training, Decentralized Execution (CTDE)

主体：
  - 每台风机 i 有独立的策略网络 π_θ(a_i | o_i)（参数共享）
  - 集中式价值网络 V_ψ(s_global) 用于训练

对抗者（两种选择）：
  选择1 - 集中式对抗者：单一策略 ν_φ(a^ν | s_global)，协调扰动所有风机
  选择2 - 分散式对抗者：每台风机一个对抗者 ν_φ_i(a^ν_i | o_i)（推荐，更现实）
```

### A.4 优缺点

```
✅ 优点：
  - 实现简单（在现有PPO/MAPPO代码上加一个对抗者网络）
  - 不需要额外的数学理论
  - 对模型参数不确定性天然鲁棒
  - 域随机化在风电场领域有经验验证（FALCON论文）
  - 训练速度与标准RL相当（每步多一次对抗者forward pass）

❌ 缺点：
  - 没有形式化的鲁棒性保证（不知道"对什么程度的偏差鲁棒"）
  - 对抗者强度(ε_max)需要手动调参
  - 可能过于保守（对抗者过强）或不够鲁棒（对抗者过弱）
  - 2023 AAAI论文指出：对抗训练不总是优于标准训练
  - 方法论贡献有限（域随机化+RARL不算新方法）
```

### A.5 关键参考文献

- Pinto et al., "Robust Adversarial RL" (ICML 2017) — RARL奠基
- Mehta et al., "Active Domain Randomization" (CoRL 2019) — ADR
- CleanRL RPO — 简化版鲁棒PPO实现

---

## 方案B：KL散度分布鲁棒SAC + CMDP寿命约束（DR-SAC框架）

### B.1 核心思想

不是对抗训练，而是**在MDP转移概率上施加KL散度模糊集**，求解最坏情况分布下的最优策略。同时叠加CMDP的疲劳约束。

核心优势：KL散度模糊集 + 最大熵RL（SAC）天然兼容——因为**KL正则化在数学上等价于熵正则化**。

### B.2 数学建模

#### 分布鲁棒Bellman方程

标准SAC的Bellman方程：
```
Q(s,a) = r(s,a) + γ E_{s'~P₀} [V(s')]
V(s) = E_{a~π} [Q(s,a) - α log π(a|s)]
```

DR-SAC的鲁棒Bellman方程：
```
Q^{DR}(s,a) = r(s,a) + γ min_{P' : D_KL(P'||P₀) ≤ ρ} E_{s'~P'} [V(s')]
```

其中：
- P₀ 是nominal转移核（FLORIS模型给出的）
- ρ 是模糊集半径（控制鲁棒性程度）
- D_KL(P'||P₀) 是KL散度

#### KL约束的对偶重整化

利用强对偶性，内层min问题有闭式解：

```
min_{P': D_KL(P'||P₀) ≤ ρ}  E_{P'}[V(s')]

= max_{λ≥0} { -λ·ρ - λ·log E_{P₀}[exp(-V(s')/λ)] }
```

其中 λ 是对偶变量。代入Bellman方程：

```
Q^{DR}(s,a) = r(s,a) - γ·λ*·ρ - γ·λ*·log E_{P₀}[exp(-V(s')/λ*)]
```

**关键洞察：** `log E[exp(-V/λ)]` 就是V的**cumulant generating function**。当ρ→0时退化为标准Bellman；ρ增大时，越来越关注V的高分位数（最坏情况）。

#### 与SAC熵正则化的统一

SAC的目标：`max E[r] + α·H(π)`

DR-SAC的目标：`max min_{P'∈U_KL} E_{P'}[r] + α·H(π)`

两者可以统一为：
```
V^{DR-SAC}(s) = max_π { min_{P'} E_{P'}[Q(s,a)] + α·H(π(·|s)) }
             = max_π { E_{P₀}[Q(s,a)] - ρ·λ - λ·log E_{P₀}[exp(-Q/λ)] + α·H(π) }
```

**双重正则化效果：**
- α (熵系数) → 鼓励探索，防止策略坍缩
- ρ (KL半径) → 防止策略依赖特定转移模型
- 两者协同：策略既不会过度利用模型细节，也不会停止探索

#### 叠加CMDP疲劳约束

完整的DR-SAC-CMDP目标：

```
max_π  min_{P': D_KL(P'||P₀) ≤ ρ}  E_{P'}[ Σ_t γ^t · P_farm(t) ] + α·H(π)

s.t.  E_{P'}[ Σ_t γ^t · DEL_i(t) ] ≤ D_max,   ∀ i = 1,...,9
```

Lagrangian形式：
```
L(π, λ_KL, {μ_i}) = E_{P₀}[Σ γ^t · P_farm(t)]
                    + α·H(π)
                    - λ_KL·(D_KL - ρ)                    # KL鲁棒性约束
                    - Σ_i μ_i·(E[Σ γ^t·DEL_i(t)] - D_max)  # 疲劳约束
```

#### 对偶变量更新

```
三个可学习的对偶变量，通过梯度上升更新：

λ_KL ← max(0, λ_KL + β₁·(D̂_KL - ρ))                 # KL约束
μ_i  ← max(0, μ_i  + β₂·(D̂EL_i - D_max))    ∀i       # 疲劳约束
α    ← α + β₃·(Ĥ(π) - H_target)                       # 熵约束(自动调温)
```

### B.3 Actor-Critic网络更新

#### Critic更新（Twin Q-networks）

```
Loss_Q(θ) = E_{(s,a,r,s')~Buffer} [(Q_θ(s,a) - y)²]

其中 target:
y = r + γ(1-done)·(min_{j=1,2} Q_{θ'_j}(s',ã') - α·log π_φ(ã'|s') - λ_KL·ρ̂)
ã' ~ π_φ(·|s')
ρ̂ = KL散度惩罚项的近似
```

#### Actor更新

```
Loss_π(φ) = E_{s~Buffer} [ α·log π_φ(ã|s) - min_{j=1,2} Q_θ_j(s,ã) + Σ_i μ_i·DEL_i(s,ã) ]
ã ~ π_φ(·|s)  (reparameterization trick)
```

#### 疲劳代理在Critic中的角色

疲劳约束通过μ_i进入有效reward：
```
r_effective = P_farm(s,a) - Σ_i μ_i · DEL_i(s,a)
```

当某台风机的疲劳接近上限 → μ_i增大 → 有效reward中该风机的疲劳权重增大 → 策略自动变保守。

### B.4 超参数

```
SAC基础参数：
  学习率 (actor/critic):   3e-4
  折扣因子 γ:              0.99
  软更新系数 τ:            0.005
  Replay buffer大小:       1,000,000
  Batch size:              256
  网络结构:                MLP [256, 256] (actor和critic各两层)

DR-SAC特有参数：
  KL模糊集半径 ρ:          0.05  (建议从0.01开始sweep到0.2)
  λ_KL 学习率:             0.01
  λ_KL 初始值:             1.0

CMDP参数（与之前方案一致）：
  μ_i 学习率:              0.01
  D_max:                   通过sweep产生Pareto前沿
  
计算开销：比标准SAC增加约5-10%
```

### B.5 优缺点

```
✅ 优点：
  - 有形式化的鲁棒性保证：策略对D_KL(P'||P₀) ≤ ρ 内所有转移模型最优
  - KL + 熵天然兼容，理论优雅
  - ρ参数有明确物理含义（模型不确定性程度）
  - 已有DR-SAC开源实现（GitHub: Lemutisme/DR-SAC）
  - 在MuJoCo连续控制上验证过有效
  - 计算开销低（5-10%）
  - 可与CMDP自然组合（都是Lagrangian框架）

❌ 缺点：
  - KL散度要求P'和P₀有相同支撑集（sim和real分布不同时有理论缺陷）
  - 不是多智能体原生方法（需要适配到MAPPO框架）
  - ρ的选择仍需交叉验证
  - SAC是off-policy（样本效率高但稳定性不如PPO）
  - 对FLORIS→FAST.Farm的迁移效果未经验证（需要实验）
```

### B.6 关键参考文献

- DR-SAC论文 (arxiv 2506.12622) — KL-DR + SAC
- GitHub: Lemutisme/DR-SAC — 开源实现
- Robust-Safe-RL (GitHub: jqueeney/robust-safe-rl) — 鲁棒安全RL

---

## 方案C：Wasserstein罚项 + 多精度验证（WGAN思路）

### C.1 核心思想

不把Wasserstein距离作为模糊集约束（NP-Hard），而是作为**正则化罚项**：

> 在FLORIS上训练策略，同时惩罚策略在FLORIS和FAST.Farm上产生的行为分布差异

本质上是一个**域适应(Domain Adaptation)问题**：让策略学到在两个仿真器上都表现良好的控制行为。

### C.2 数学建模

#### 总目标函数

```
max_π  J(π) = E_{P_FLORIS}[ Σ_t γ^t · P_farm(t) ] - λ_W · W_1(D^π_FLORIS, D^π_FAST)

其中：
  D^π_FLORIS = 策略π在FLORIS上产生的轨迹分布
  D^π_FAST   = 策略π在FAST.Farm上产生的轨迹分布
  W_1(·,·)   = 1-Wasserstein距离
  λ_W        = Wasserstein罚项权重
```

**直觉：** 如果策略在两个仿真器上的行为几乎一样（W₁小），那么它更可能对模型差异鲁棒——因为它没有利用某个仿真器特有的"漏洞"。

#### Wasserstein距离的对偶计算（Kantorovich-Rubinstein）

直接计算W₁需要求解最优传输问题（O(n³)），不可行。利用对偶：

```
W_1(P, Q) = sup_{||f||_Lip ≤ 1} { E_P[f(x)] - E_Q[f(x)] }
```

其中 `||f||_Lip ≤ 1` 表示f是1-Lipschitz连续的。

**用神经网络实现**（WGAN思路）：
- 训练一个**Wasserstein判别器** f_ω（critic network）
- 对f_ω施加**谱归一化(Spectral Normalization)**强制Lipschitz约束
- f_ω的目标：最大化两个分布上的期望差

#### 三个网络的训练

```
网络1: 策略网络 π_θ (actor)
  - 输入: 观测 o_t
  - 输出: 偏航/变桨动作
  - 目标: 最大化发电功率 - λ_W · W_estimate

网络2: 价值网络 V_ψ (critic)  
  - 输入: 状态 s_t
  - 输出: 价值估计
  - 目标: 最小化TD误差

网络3: Wasserstein判别器 f_ω (domain critic) ← 新增
  - 输入: 轨迹特征 h(τ)
  - 输出: 标量分数
  - 约束: 谱归一化保证1-Lipschitz
  - 目标: 最大化 E_FLORIS[f(h)] - E_FAST[f(h)]
```

#### 完整训练算法

```
初始化: π_θ, V_ψ, f_ω
初始化: FLORIS环境 env_F, FAST.Farm环境 env_FF (或代理)

for epoch = 1, 2, ..., N:
    
    # Step 1: 收集两个仿真器的轨迹
    τ_F  = {rollout(π_θ, env_F)  for _ in range(B)}    # FLORIS轨迹集
    τ_FF = {rollout(π_θ, env_FF) for _ in range(B)}    # FAST.Farm轨迹集
    
    # Step 2: 提取轨迹特征
    h_F  = [encode(τ) for τ in τ_F]     # 每条轨迹编码为固定长度向量
    h_FF = [encode(τ) for τ in τ_FF]
    
    # Step 3: 更新Wasserstein判别器 (n_critic步)
    for k = 1, ..., n_critic:
        L_ω = -( mean(f_ω(h_F)) - mean(f_ω(h_FF)) )   # 最大化距离 → 最小化负距离
        ω ← ω - η_ω · ∇_ω L_ω
        # 谱归一化自动维护Lipschitz约束
    
    # Step 4: 估计Wasserstein距离
    W_estimate = mean(f_ω(h_F)) - mean(f_ω(h_FF))
    
    # Step 5: 更新策略（在FLORIS上用PPO，加Wasserstein罚项）
    advantages = compute_GAE(τ_F, V_ψ)
    L_policy = PPO_loss(π_θ, τ_F, advantages) - λ_W · W_estimate
    θ ← θ + η_θ · ∇_θ L_policy
    
    # Step 6: 更新价值网络
    L_value = MSE(V_ψ(s_t), R_t)
    ψ ← ψ - η_ψ · ∇_ψ L_value
```

#### 轨迹编码方法

将变长轨迹编码为固定长度特征向量：

```
方法1 - 统计特征（简单高效）：
  h(τ) = [mean(powers), std(powers), mean(yaw_angles), std(yaw_angles),
           mean(wind_speed), cumulative_del, total_power]
  维度: ~20-30

方法2 - 分段平均（保留时序信息）：
  将长度T的轨迹分为K段，每段取均值
  h(τ) = [avg_segment_1, avg_segment_2, ..., avg_segment_K]
  维度: K × obs_dim

方法3 - GRU编码器（最灵活）：
  h(τ) = GRU_encoder(o_1, o_2, ..., o_T)  # 最后hidden state
  维度: hidden_size (128-256)
```

#### 谱归一化实现

```python
import torch.nn as nn

class WassersteinCritic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_dim, 256)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(256, 256)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(256, 1)),
        )
    
    def forward(self, x):
        return self.net(x)
```

谱归一化保证每层权重矩阵的谱范数（最大奇异值）≤ 1，从而整个网络是1-Lipschitz的。

#### 替代方案：Sinkhorn距离（可微分的OT）

如果不想用判别器网络，可以直接用Sinkhorn算法计算近似Wasserstein距离：

```python
import ot  # Python Optimal Transport库

def sinkhorn_penalty(X_floris, X_fast, reg=0.1):
    """
    X_floris: [n_traj, feature_dim] FLORIS轨迹特征
    X_fast:   [n_traj, feature_dim] FAST.Farm轨迹特征
    """
    n = X_floris.shape[0]
    a = torch.ones(n) / n  # 均匀权重
    b = torch.ones(n) / n
    cost = torch.cdist(X_floris, X_fast)  # 欧氏距离矩阵
    W = ot.sinkhorn2(a, b, cost, reg=reg)  # 可微分
    return W
```

Sinkhorn的优势：**完全可微分**，可以直接backprop到策略参数。

### C.3 实际实现问题

#### FAST.Farm太慢怎么办？

FAST.Farm运行一个episode需要分钟级，不适合大量采样。三个解决方案：

```
方案1: 离线采集 + 判别器
  - 预先用当前策略在FAST.Farm上跑N条轨迹（offline dataset）
  - 训练判别器只需要这些固定数据
  - 每隔K个训练epoch重新采集一批FAST.Farm数据

方案2: FAST.Farm代理模型
  - 用FAST.Farm数据训练一个NN代理模型（dynamics surrogate）
  - NN代理替代FAST.Farm参与在线训练
  - 速度接近FLORIS，精度接近FAST.Farm

方案3: 多精度渐进训练
  - Phase 1: 纯FLORIS训练（快速迭代）
  - Phase 2: FLORIS + Wasserstein罚项（用预采集的FAST.Farm数据）
  - Phase 3: 在FAST.Farm上fine-tune（少量步数）
```

### C.4 理论分析框架

这是此方案最有学术价值的部分——你可以用DRO理论分析这个罚项的性质：

```
定理框架（需要你证明）：

设 π* = argmax_π { J_FLORIS(π) - λ_W · W_1(D^π_F, D^π_FF) }

则对任意转移核 P_real 满足 W_1(P_FLORIS, P_real) ≤ δ：

  J_real(π*) ≥ J_FLORIS(π*) - C · (δ + W_1(D^π*_F, D^π*_FF))

其中 C 是Lipschitz常数。
```

**含义：** 如果Wasserstein罚项成功地使策略在两个仿真器上行为一致（W₁小），那么策略在真实环境中的性能下界是可保证的。

这个性能保证的证明可以作为**论文的核心理论贡献**，直接利用你在Wasserstein DRO方面的专长。

### C.5 优缺点

```
✅ 优点：
  - 规避了Wasserstein DR-MDP的NP-Hard内层优化
  - 利用WGAN计算技巧（成熟、稳定）
  - 可以给出理论性能保证（利用你的DRO背景）
  - 自然地结合多精度仿真器
  - 创新性极高（没有人将WGAN思路用于风电场sim-to-real）
  - λ_W有直观含义：仿真器可信度

❌ 缺点：
  - 需要FAST.Farm数据（至少离线采集一批）
  - 额外的判别器网络增加训练复杂度
  - Wasserstein罚项可能导致训练不稳定（WGAN的老问题）
  - 理论分析需要一定数学功底（但这正是你的优势）
  - 实现比方案A/B复杂
  - 需要选择轨迹编码方法（影响距离计算质量）
```

### C.6 关键参考文献

- Arjovsky et al., "Wasserstein GAN" (ICML 2017) — WGAN计算框架
- Zhang et al., "Policy Optimization as Wasserstein Gradient Flows" (ICML 2018) — Wasserstein策略优化
- Baheri et al., "WAVE: Wasserstein Adaptive Value Estimation" (ICLR 2025) — Sinkhorn正则化
- Shen et al., "WDGRL" (AAAI 2018) — Wasserstein域适应
- POT库 (Python Optimal Transport) — Sinkhorn可微分实现

---

## 三方案对比总结

| 维度 | 方案A (RARL) | 方案B (KL-DR-SAC) | 方案C (Wass罚项) |
|------|-------------|-------------------|-----------------|
| 鲁棒性类型 | 最坏情况扰动 | KL模糊集内最坏分布 | 跨仿真器行为一致性 |
| 数学基础 | 零和博弈 | KL-DRO + 最大熵 | Wasserstein OT + 域适应 |
| 内层优化 | 无（对抗者是NN） | 闭式解（KL对偶） | 对偶近似（判别器/Sinkhorn）|
| 需要FAST.Farm？ | 否（纯FLORIS） | 否（纯FLORIS） | 是（至少离线数据） |
| 理论保证 | 无形式化保证 | KL球内最优 | 可证明迁移下界 |
| 计算开销 | +对抗者训练 (~30%) | +KL计算 (~10%) | +判别器+双环境 (~50%) |
| 创新定位 | 应用创新 | 方法+应用 | 理论+方法+应用 |
| 实现难度 | ★★ | ★★★ | ★★★★ |
| 论文影响力 | ★★★ | ★★★★ | ★★★★★ |
| 建议角色 | Baseline对比 | 主方案 | 扩展贡献/独立论文 |

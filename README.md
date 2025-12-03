# 理论
## World Model
+ **Tokenizer**：  
    - 输入：环境的原始视觉数据（如 Minecraft 画面）。  
    - 输出：低维连续的 latent 状态（压缩的环境表示），以及重建的视觉帧（用于自监督训练）。  
    - 作用：将高维、冗余的视觉输入“翻译”为 Agent 可处理的紧凑状态，同时保留关键环境信息。
+ **Dynamics**：  
    - 输入：当前 latent 状态、动作、信号水平 τ（噪声强度）、步长 d（时间尺度）。
    - 输出：下一时刻的 latent 状态（预测值）。  
    - 作用：学习“动作如何改变环境状态”的规律，提供“状态转换预测”能力，是 WM 作为“内部模拟器”的核心。

**与环境的关系**：WM 直接处理环境的原始输出（视觉数据），并通过 latent 状态间接向 Agent 传递环境信息，是 Agent 感知环境的“中介”。

## Agent
+ **Policy Head**：  
    - 输入：WM 预测的 latent 状态、任务标签（如“挖钻石”）。  
    - 输出：动作分布（离散动作的概率，如“向前移动”“使用镐子”）。  
    - 作用：决定“该做什么动作”，是 Agent 的“决策器”。
+ **Reward Head**：  
    - 输入：WM 预测的 latent 状态、任务标签。  
    - 输出：预测的即时奖励（如“挖到钻石得 1 分”）。  
    - 作用：学习“什么状态对任务有利”，为后续强化学习提供监督信号。
+ **Value Head**：  
    - 输入：WM 预测的 latent 状态、任务标签。  
    - 输出：预测的长期价值（未来累积奖励的期望）。  
    - 作用：辅助评估动作的长期收益，用于计算 λ-returns 和优势函数（支撑 PMPO 训练）。

**与 WM 的关系**：Agent 不直接处理原始环境数据，而是基于 WM 输出的 latent 状态做决策；同时，Agent 的训练依赖 WM 提供的“想象轨迹”（通过 Dynamics 预测生成）。

## Trainer
+ **分阶段训练控制**：  
    - 阶段 1（WM 预训练）：调度无标注/弱标注数据，训练 Tokenizer（图像重建损失）和 Dynamics（Shortcut Forcing 损失），冻结 WM 后进入下一阶段。  
    - 阶段 2（Agent 微调）：调度带任务标签的数据，联合训练 Dynamics（微调）、Policy Head（行为克隆损失）、Reward Head（奖励预测损失）。  
    - 阶段 3（想象训练）：调度初始上下文数据，通过 WM 生成想象轨迹，用 PMPO 算法优化 Policy Head（结合 λ-returns 和优势函数）。
+ **评估与推理**：  
    - 离线评估：在测试环境（如 Minecraft 钻石挑战）中运行训练好的 Agent，统计任务成功率（如挖到钻石的比例）。  
    - 实时推理：优化模型推理速度（如 K=4 步 Shortcut Forcing 采样），确保 Agent 能在 20FPS 帧率下实时交互。

**与其他模块的关系**：Trainer 不直接参与数据处理或决策，而是通过控制“数据流向”“损失计算”“参数更新”，协调 WM 和 Agent 完成训练，是整个系统的“组织者”。

---

![画板](https://cdn.nlark.com/yuque/0/2025/jpeg/38709574/1763351194250-0691312c-14ab-4a2e-af79-e38b54f5a42a.jpeg)

---

## Tokenizer 预训练
输入：视频数据（b, t, 3 ,H, W） 

步骤：

+ 预处理
    - <font style="color:rgba(0, 0, 0, 0.85);">标准化</font>
    - <font style="color:rgba(0, 0, 0, 0.85);">随机裁剪</font>
+ 卷积嵌入
    - 展开
    - 卷积
+ 随机特征掩码
+ 拼接可学习 Latent Token
    - <font style="color:rgba(0, 0, 0, 0.85);">加入一组可学习的 “Latent Token”（空间 Token）作为后续Tokenizer 的输出</font>
+ Transformer 编码
+ 提取 Latent 
+ 解码重建
    - 上映射
    - Transformer 编码
    - 提取 Latent 
    - <font style="color:rgba(0, 0, 0, 0.85);">转置卷积</font>
+ 参数更新

损失：$ L(\theta) = L_{MSE}(\theta) + 0.2L_{LPIPS}(\theta) $

维度：

+ $ (b, t, 3 ,H, W)\to (bt, p_hp_w, h)\to (bt, p_hp_w+l, h)\to (bt, l, h)\to (b, t, l, h) $
+ $ (bt, l, h)\to (bt, p_hp_w+l, h)\to (bt, p_hp_w, h)\to (bt, h, p_h, p_w)\to (b, t, 3 ,H, W) $

---

## Dynamics 预训练
输入：视频数据（b, t, 3 ,H, W） 动作序列（b, t, 1）信号（b, t, 1）步长（1,）

步骤：

+ 数据预处理
    - tokenizer处理视频数据为latent
    - 动作，信号，步长embedding
+ <font style="color:rgba(0, 0, 0, 0.85);">生成噪声latent</font>
    - $ \tilde{z}=\tau z + (1-\tau)\cdot\text{noise}
 $
+ 拼接latent
    - 展平<font style="color:rgba(0, 0, 0, 0.85);">噪声latent</font>
    - 拼接动作，信号，步长embedding
    - 拼接<font style="color:rgba(0, 0, 0, 0.85);">引入 8 个可学习的 “寄存器 Token”用于存储长期时序信息</font>
+ Shortcut Forcing
    - 随机步长
    - 按步长阶段性预测
    - $ w(\tau) = 0.9\tau + 0.1 $$ L = \sum w(\tau) MSE(pred, target) $

---

## Agent 微调
输入：视频数据（b, t, 3 ,H, W） 动作序列（b, t, 1）任务标签（b, ）奖励信号（b, t），$ \tau = 0.9, d = 0.25 $

步骤：

+ 数据预处理：视频标准化，动作离散化，奖励标准化
+ Tokenizer 编码
+ Dynamics预测
+ 任务嵌入
+ 对齐Dynamics预测latent与任务嵌入embedding
+ Policy Head输出预测动作，Reward Head输出预测奖励，Value Head输出预测价值

损失：

+ 行为克隆损失：$ CE(logits, actions) $
+ 奖励预测损失：$ MSE(pred\_reward, reward) $
+ Dynamics 微调损失：$ w(\tau) MSE(pred\_z, z) $

---

## 想象训练
输入：

+ 真实环境的初始观测（10帧视频，用于启动想象）(1, 10, 3, H, W)
+ 任务标签

步骤：

+ Dynamics 生成初始 latent
+ 取最后一帧latent作为想象起点$ z_0 $(1, l, h）
+ 自回归生成K步轨迹
    - 当前状态（从想象起点出发）
    - 融合任务信息
    - Policy Head输出动作分布并采样动作
    - Dynamics预测下一状态
    - Reward/Value Head预测
    - 记录轨迹
+ 计算λ-returns和优势函数
+ PMPO策略优化
    - 策略损失
    - KL 约束
---

# 实践

## Usage
创建环境：`docker build -t embody:latest .`
运行环境：`docker run -it -v (PWD):/app embody:v2`
环境信息：详见 [Minerl](https://minerl.readthedocs.io/en/v0.4.4/index.html)

使用示例
```python
import minerl
from config import GameConfig, TrainConfig
from main import train_tokenizer
from model import Tokenizer

def workflow():
    trainer = Trainer(
        TrainConfig.train_tokenizer_epochs, 
        TrainConfig.train_dynamics_epochs, 
        TrainConfig.train_agent_epochs
    )
    data = minerl.data.make(
        Tasks[0],
        data_dir=GameConfig.DATA_DIR,
        num_workers=4
    )
    train_tokenizer(trainer, data)
```

## 项目结构
dreamer4/
├── requirements.txt    # 依赖清单
├── README.md
/models
/data
├── /TASK          # 储存Minecraft VPT 数据集，数据来源[https://zenodo.org/records/12659939](https://zenodo.org/records/12659939)
/code
├── config.py          # 训练参数配置（学习率、网络维度、训练轮数等）
├── main.py            # 训练入口（调度分阶段训练）
├── model.py           # 核心模型定义（Tokenizer、Dynamics、Agent）
├── trainer.py         # 训练逻辑实现（分阶段训练、损失计算、轨迹生成）
├── logger_manager.py  # 日志管理（控制台+文件双输出）
└── logs/              # 训练日志目录

## TODO
- Tokenizer
    - [ ] GQA
    - [ ] Flash Attention
    - [ ] RoPE
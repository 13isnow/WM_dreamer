import os

class GameConfig:
    TASKS = [
        "MineRLTreechop-v0",
    ]
    ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(ROOT_PATH, "data")
    MODEL_DIR = os.path.join(ROOT_PATH, "models")

class DataConfig:
    BATCH_SIZE = 8
    SEQ_LEN = 16
    BATCHES = 10

class TrainConfig:
    # train
    train_tokenizer_epochs = 10
    train_dynamics_epochs = 20
    train_agent_epochs = 30
    # dim
    token_dim = 128
    action_dim = 64
    hidden_dim = 256
    # image
    image_dim = (64, 64)
    stride = 4
    kernel_size = 4
    padding = 0
    # attention
    num_tokens = 8
    mask_prob = 0.15
    num_heads = 4
    num_layers = 2
    # dynamics
    tau_dim = 10
    tau_range = (0.0, 1.0)
    step_dim = 4
    step_range = [0.25, 0.5, 0.75, 1.0]
    num_registers = 8
    # agent
    num_tasks = len(GameConfig.TASKS)
    policy_layers = [token_dim, hidden_dim, action_dim]
    value_layers = [token_dim, hidden_dim, 1]
    reward_layers = [token_dim, hidden_dim, 1]
    # PMPO
    K_steps = 100
    # args
    gamma = 0.99
    lambda_r = 0.95
    alpha = 0.5
    beta = 0.3
    # optimizer
    weight_decay = 1e-6
    T_max = 50
    tokenizer_lr = 1e-4
    dynamics_lr = 5e-5
    agent_lr = 3e-5


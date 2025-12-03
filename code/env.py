import torch

def extract_pov(obs):
    return torch.Tensor(obs["pov"])


Tasks_ENV_MAP = {
    "MineRLTreechop-v0": {
        0: extract_pov
    },
}
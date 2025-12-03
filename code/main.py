import os
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
import minerl

from config import GameConfig, DataConfig, TrainConfig
from model import Tokenizer, Dynamics
from trainer import Trainer
from logger_manager import logger

Tasks = GameConfig.TASKS

class DataBuffer(IterableDataset):
    def __init__(self, data, attrs):
        self.data = data
        self.attrs = attrs

    def __iter__(self):
        for batch in self.data.batch_iter(
            batch_size=DataConfig.BATCH_SIZE,
            seq_len=DataConfig.SEQ_LEN,
            num_epochs=DataConfig.BATCHES
        ):
            yield [batch[attr] for attr in self.attrs]


def train_tokenizer(trainer, data):
    tokenizer = Tokenizer(
        image_dim=TrainConfig.image_dim,
        stride=TrainConfig.stride,
        kernel_size=TrainConfig.kernel_size,
        padding=TrainConfig.padding,
        token_dim=TrainConfig.token_dim,
        num_tokens=TrainConfig.num_tokens,
        mask_prob=TrainConfig.mask_prob,
        num_heads=TrainConfig.num_heads,
        num_layers=TrainConfig.num_layers,
    )
    optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=TrainConfig.tokenizer_lr, weight_decay=TrainConfig.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TrainConfig.T_max)
    logger.info("tokenizer train start")
    dataloader = DataBuffer(data, attrs=[0])
    loss = trainer.train_tokenizer(tokenizer, dataloader, optimizer, scheduler)
    tokenizer.save(os.path.join(GameConfig.MODEL_DIR, "tokenizer.pth"))
    logger.info(f"Tokenizer training completed")
    logger.info(f"Final Tokenizer Loss: {loss:.4f}")
    return tokenizer


def main():
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
    

if __name__ == "__main__":
    main()
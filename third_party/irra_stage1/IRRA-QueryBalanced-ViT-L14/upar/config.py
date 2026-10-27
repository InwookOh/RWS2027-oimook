"""Initial DESIGN.md settings and unchanged IRRA Adam/scheduler defaults."""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    img_size: tuple = (378, 126)
    stride_size: int = 14
    text_length: int = 77
    vocab_size: int = 49408
    temperature: float = 0.02
    cmt_depth: int = 4
    map_loss_weight: float = 1.0


@dataclass
class TrainConfig:
    batch_size: int = 32
    test_batch_size: int = 128
    num_workers: int = 8
    num_epoch: int = 60
    lr: float = 1e-5
    weight_decay: float = 4e-5
    alpha: float = 0.9
    beta: float = 0.999
    adam_eps: float = 1e-3
    milestones: tuple = (20, 50)
    gamma: float = 0.1
    warmup_factor: float = 0.1
    warmup_epochs: int = 5
    warmup_method: str = 'linear'
    lrscheduler: str = 'cosine'
    target_lr: float = 0.0
    power: float = 0.9
    seed: int = 1
    log_period: int = 100
    eval_query_batch_size: int = 32


@dataclass
class MemoryConfig:
    micro_batch_size: int = 1
    accumulation_steps: int = 32
    gradient_checkpointing: bool = True
    evaluation_micro_batch_size: int = 4

    def validate(self, batch_size=32):
        if min(self.micro_batch_size, self.accumulation_steps, self.evaluation_micro_batch_size) < 1:
            raise ValueError('Memory batch controls must be positive')
        if batch_size != 32 or self.micro_batch_size * self.accumulation_steps != batch_size:
            raise ValueError('Preserve Effective Batch 32')

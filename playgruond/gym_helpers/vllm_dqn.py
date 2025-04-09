import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import defaultdict
from typing import Optional, Dict, Any

class VLLM_DQN(nn.Module):
    """
    A simple DQN model for the VLLM agent.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super(VLLM_DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
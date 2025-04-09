from collections import defaultdict
import gymnasium as gym
import numpy as np
from vllm_dqn import VLLM_DQN
import torch

class VLLMAgent:
    """
    A class to manage the VLLM agent.
    """
    def __init__(self, env: gym.Env, config: dict):
        self.env = env
        self.config = config
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.state = None
        self.reward_range = (-1, 1)
        self.done = False
        self.info = {}
        self.stats = defaultdict(list)
        self.reset()
        
        # Initialize the DQN model
        # output_dim is 1 because we are using a continuous action space
        self.model = VLLM_DQN(input_dim=self.observation_space.shape[0], output_dim=1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss()
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99
        self.batch_size = 32
        self.memory = []
        self.memory_size = 10000
        self.learn_step_counter = 0
        self.learn_step = 0
        
    def get_action(self, observation: np.ndarray) -> np.float32:
        """
        Get the action from the action space.
        """
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()
        else:
            state = torch.FloatTensor(observation).unsqueeze(0)
            action = self.model(state).detach().numpy()
            # should be just a value from a single neuron
            action = np.clip(action, self.action_space.low, self.action_space.high)
            return action
    
    def update(self, action: np.float32, reward: float, next_observation: np.ndarray, done: bool):
        """
        Update the DQN model with the new observation.
        """
        self.memory.append((self.state, action, reward, next_observation, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        
        if len(self.memory) >= self.batch_size:
            batch = np.random.choice(len(self.memory), self.batch_size)
            states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])
            
            states = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1)
            
            q_values = self.model(states).gather(1, actions.long())
            next_q_values = self.model(next_states).max(1)[0].detach().unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
            loss = self.criterion(q_values, target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            self.learn_step_counter += 1
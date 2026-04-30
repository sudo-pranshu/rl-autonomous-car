import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(self):
        self.model = Net()
        self.target = Net()
        self.target.load_state_dict(self.model.state_dict())

        self.memory = deque(maxlen=10000)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.985
        self.epsilon_min = 0.05

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return torch.argmax(self.model(state_tensor)).item()

    def train(self):
        if len(self.memory) < 64:
            return

        batch = random.sample(self.memory, 64)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert efficiently
        states = torch.tensor(np.array(states), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        actions = torch.tensor(actions).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Q values
        q_values = self.model(states).gather(1, actions).squeeze()
        next_q = self.target(next_states).max(1)[0]

        # FIXED TARGET
        target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

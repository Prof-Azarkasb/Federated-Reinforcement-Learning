import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Hyperparameters based on Table 3
GAMMA = 0.95
CRITIC_LR = 0.001
POLICY_LR = 0.0001
BUFFER_SIZE = 405000
BATCH_SIZE = 1024
TAU = 0.01
NUM_AGENTS = 3
EPISODES = 100000
STATE_SIZE = 10  # Example state size (agent's observation + landmark positions)
ACTION_SIZE = 5  # Discretized 2D action space

# Dataset Loader class to load and preprocess Google Cluster Dataset
class DatasetLoader:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.preprocess_data()

    def preprocess_data(self):
        # Basic data cleaning: Handle missing values and outliers
        self.data.fillna(self.data.mean(), inplace=True)  # Fill missing values with column means
        self.data = self.data[self.data['failed'] == 0]  # Filter out failed tasks

        # Feature selection based on the experimental setup
        selected_features = ['time', 'resource_request', 'cpu_usage_distribution', 'priority', 'scheduling_class']
        self.data = self.data[selected_features]

        # Normalize features
        scaler = StandardScaler()
        self.data[selected_features] = scaler.fit_transform(self.data[selected_features])

    def get_random_sample(self):
        # Return a random sample from the dataset as a state for training
        sample = self.data.sample(n=1).values.flatten()
        return sample

# Policy Network (for action selection)
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)  # Output probability distribution

# Critic Network (for value estimation)
class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(CriticNetwork, self).__init__()
        self.rnn = nn.GRU(state_size + action_size, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        x, _ = self.rnn(x.unsqueeze(0))
        return self.fc(x.squeeze(0))

# Replay Buffer with Gumbel-Softmax mechanism
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Multi-Agent Environment Simulation
class MultiAgentEnv:
    def __init__(self, num_agents, dataset_loader):
        self.num_agents = num_agents
        self.dataset_loader = dataset_loader

    def reset(self):
        # Use a random sample from the dataset as initial states
        return np.array([self.dataset_loader.get_random_sample() for _ in range(self.num_agents)])

    def step(self, actions):
        # Simulate the next states and rewards
        next_states = np.array([self.dataset_loader.get_random_sample() for _ in range(self.num_agents)])
        rewards = np.random.rand(self.num_agents)
        dones = np.random.choice([False, True], self.num_agents, p=[0.95, 0.05])
        return next_states, rewards, dones

# Agent implementing CDC-based MARL
class MARLAgent:
    def __init__(self):
        self.policy_net = PolicyNetwork(STATE_SIZE, ACTION_SIZE)
        self.critic_net = CriticNetwork(STATE_SIZE, ACTION_SIZE)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=POLICY_LR)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=CRITIC_LR)
        self.buffer = ReplayBuffer(BUFFER_SIZE)

    def select_action(self, state):
        with torch.no_grad():
            probabilities = self.policy_net(torch.tensor(state, dtype=torch.float32))
            return np.random.choice(ACTION_SIZE, p=probabilities.numpy())  # Stochastic action selection

    def update_networks(self):
        if len(self.buffer.buffer) < BATCH_SIZE:
            return

        batch = self.buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        # Compute Q-values
        q_values = self.critic_net(states, actions.float())
        with torch.no_grad():
            max_next_q_values = self.critic_net(next_states, actions.float()).max(1)[0]
            target_q_values = rewards + GAMMA * max_next_q_values

        # Critic loss
        critic_loss = nn.functional.mse_loss(q_values.squeeze(), target_q_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Soft update (τ)
        for target_param, param in zip(self.critic_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

# Training Loop
def train_marl():
    dataset_loader = DatasetLoader("google_2019_cluster_sample.csv")  # Specify path to dataset
    env = MultiAgentEnv(NUM_AGENTS, dataset_loader)
    agents = [MARLAgent() for _ in range(NUM_AGENTS)]

    for episode in range(EPISODES):
        states = env.reset()
        done = False
        episode_rewards = np.zeros(NUM_AGENTS)

        while not done:
            actions = [agent.select_action(state) for agent, state in zip(agents, states)]
            next_states, rewards, dones = env.step(actions)

            for i, agent in enumerate(agents):
                agent.buffer.add((states[i], actions[i], rewards[i], next_states[i]))

            states = next_states
            done = any(dones)

            for agent in agents:
                agent.update_networks()

        print(f"Episode {episode + 1}: Avg Reward = {np.mean(episode_rewards):.2f}")

# Run Training
if __name__ == "__main__":
    train_marl()

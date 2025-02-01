import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import gym
from collections import deque
import time
import pandas as pd

# Hyperparameters
GAMMA_SHORT = 0.95
GAMMA_LONG = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 32
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
NUM_AGENTS = 3  # Number of decentralized agents
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
TRAINING_STEPS = 400

# Dataset loading and preprocessing
class DatasetLoader:
    def __init__(self, dataset_path):
        self.dataset = pd.read_csv(dataset_path)
        self.clean_data()
    
    def clean_data(self):
        # Basic data cleaning (fill missing values, remove duplicates, etc.)
        self.dataset.fillna(0, inplace=True)
        self.dataset.drop_duplicates(inplace=True)
    
    def get_features_and_labels(self):
        # Selecting important features for task scheduling and resource allocation
        features = self.dataset[['time', 'cpu_usage_distribution', 'resource_request', 'priority', 'failed']]
        labels = self.dataset['scheduling_class']
        return features, labels

    def get_batch(self, batch_size):
        # Randomly sample a batch for training
        return self.dataset.sample(n=batch_size)

# Policy Network Architecture
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Decentralized Agent
class Agent:
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.policy_net = PolicyNetwork(state_space, action_space)
        self.target_net = PolicyNetwork(state_space, action_space)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.epsilon = 1.0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_space)
        else:
            with torch.no_grad():
                return self.policy_net(torch.tensor(state, dtype=torch.float32)).argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + (GAMMA_SHORT * next_q_values * (1 - dones))
        
        loss = nn.functional.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

# Decentralized Environment
class DecentralizedEnvironment:
    def __init__(self, env_name, dataset_path):
        self.env = gym.make(env_name)
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.agents = [Agent(self.state_space, self.action_space) for _ in range(NUM_AGENTS)]
        self.dataset_loader = DatasetLoader(dataset_path)  # Load dataset here

    def run_episode(self):
        states = [self.env.reset() for _ in range(NUM_AGENTS)]
        dones = [False] * NUM_AGENTS
        total_rewards = [0] * NUM_AGENTS
        start_time = time.time()
        steps = 0
        
        while not all(dones):
            for i, agent in enumerate(self.agents):
                if not dones[i]:
                    action = agent.select_action(states[i])
                    next_state, reward, done, _ = self.env.step(action)
                    agent.store_transition(states[i], action, reward, next_state, done)
                    agent.optimize_model()
                    states[i] = next_state
                    total_rewards[i] += reward
                    dones[i] = done
                    steps += 1
                    
                    if steps % TRAINING_STEPS == 0:
                        agent.update_target_network()

        task_completion_time = time.time() - start_time
        convergence_rate = steps / task_completion_time if task_completion_time > 0 else 0
        
        print(f"Episode Metrics: Total Reward: {sum(total_rewards)}, Convergence Rate: {convergence_rate:.3f}")
        
        for agent in self.agents:
            agent.decay_epsilon()

        return sum(total_rewards)

    def train(self, num_episodes):
        for episode in range(num_episodes):
            reward = self.run_episode()
            print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {reward}")

if __name__ == "__main__":
    env_name = "CartPole-v1"
    num_episodes = 100
    dataset_path = 'google_2019_cluster_sample.csv'  # Path to your dataset
    decentralized_env = DecentralizedEnvironment(env_name, dataset_path)
    decentralized_env.train(num_episodes)

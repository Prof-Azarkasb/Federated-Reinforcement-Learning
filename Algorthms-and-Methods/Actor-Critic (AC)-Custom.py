import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import time
import pandas as pd
from collections import namedtuple
from sklearn.preprocessing import StandardScaler

# Hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.99

# Actor Network using LSTM
class Actor(nn.Module):
    def __init__(self, state_space, action_space):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(state_space, 128, batch_first=True)
        self.fc = nn.Linear(128, action_space)
    
    def forward(self, x):
        x, _ = self.lstm(x.unsqueeze(0))  # LSTM expects 3D input (batch, seq, feature)
        x = torch.relu(x[:, -1, :])  # Take last LSTM output
        return torch.softmax(self.fc(x), dim=-1)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        action = torch.multinomial(probs, 1).item()
        return action, torch.log(probs[0, action])

# Critic Network using LSTM
class Critic(nn.Module):
    def __init__(self, state_space):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(state_space, 128, batch_first=True)
        self.fc = nn.Linear(128, 1)
    
    def forward(self, x):
        x, _ = self.lstm(x.unsqueeze(0))
        x = torch.relu(x[:, -1, :])
        return self.fc(x)

# Actor-Critic Agent
class ACAgent:
    def __init__(self, state_space, action_space):
        self.actor = Actor(state_space, action_space)
        self.critic = Critic(state_space)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)

    def update(self, log_prob, reward, state_value, next_state_value):
        # Compute advantage
        advantage = reward + GAMMA * next_state_value - state_value

        # Actor loss (policy gradient)
        actor_loss = -log_prob * advantage

        # Critic loss (TD learning)
        critic_loss = advantage.pow(2)

        # Update actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

# Dataset Class for Google Cluster Dataset
class GoogleClusterDataset:
    def __init__(self, dataset_path):
        # Load dataset
        self.data = pd.read_csv(dataset_path)
        
        # Preprocess features: Standardize and normalize
        self.features = self.data[['cpu_usage', 'memory_usage', 'priority', 'execution_time']].values
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

        # Task-related features and actions
        self.tasks = self.data[['cpu_usage', 'memory_usage', 'priority', 'execution_time']].values
        self.num_tasks = len(self.data)
        self.current_idx = 0

    def get_next_task(self):
        # Return the next task's features
        task = self.tasks[self.current_idx]
        self.current_idx = (self.current_idx + 1) % self.num_tasks
        return task

    def reset(self):
        self.current_idx = 0

# Environment for Actor-Critic Training using Google Cluster Dataset
class ACEnvironment:
    def __init__(self, dataset_path):
        self.dataset = GoogleClusterDataset(dataset_path)
        self.state_space = 4  # Task features: CPU, Memory, Priority, Execution Time
        self.action_space = 3  # Actions: 0 = "Not scheduled", 1 = "Low priority", 2 = "High priority"
        self.agent = ACAgent(self.state_space, self.action_space)

    def run_episode(self):
        state = self.dataset.get_next_task()  # Get the first task as the initial state
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        total_reward = 0
        done = False
        start_time = time.time()

        while not done:
            action, log_prob = self.agent.actor.select_action(state)
            state_value = self.agent.critic(torch.from_numpy(state).float().unsqueeze(0))
            next_state, reward, done, _ = self.simulate_task_execution(action)
            next_state_value = self.agent.critic(torch.from_numpy(next_state).float().unsqueeze(0)) if not done else torch.tensor(0.0)

            self.agent.update(log_prob, reward, state_value, next_state_value)
            state = next_state
            total_reward += reward

        task_completion_time = time.time() - start_time
        print(f"Episode Metrics: Total Reward: {total_reward}, Task Completion Time: {task_completion_time:.3f}s")
        return total_reward

    def simulate_task_execution(self, action):
        # Simulate reward based on task scheduling action
        if action == 0:  # Not scheduled
            reward = -1  # Penalize for not scheduling the task
        elif action == 1:  # Low priority
            reward = 1  # Reward for scheduling with low priority
        elif action == 2:  # High priority
            reward = 2  # Reward for scheduling with high priority
        
        # Fetch the next task's state for the next step
        next_state = self.dataset.get_next_task()
        return next_state, reward, False, {}

    def train(self, num_episodes):
        for episode in range(num_episodes):
            reward = self.run_episode()
            print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {reward}")

if __name__ == "__main__":
    # Path to the Google 2019 Cluster Dataset (replace with actual path)
    dataset_path = "google_cluster_2019.csv"

    # Training setup for Actor-Critic environment
    num_episodes = 100
    ac_env = ACEnvironment(dataset_path)
    ac_env.train(num_episodes)

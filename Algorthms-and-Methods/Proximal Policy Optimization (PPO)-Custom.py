import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import time
import pandas as pd
from collections import namedtuple
from sklearn.preprocessing import StandardScaler

# Hyperparameters based on Table 13
GAMMA = 0.99  # Discount Factor
LEARNING_RATE = 0.0003  # Adam Optimizer Learning Rate
CLIP_EPS = 0.2  # Clipped Objective for Stable Updates
BATCH_SIZE = 64  # Batch Size for Training
EPOCHS = 10  # Multiple Epochs for Refining Policy
ENTROPY_COEFF = 0.01  # Entropy Regularization for Exploration

# Load and Preprocess Google 2019 Cluster Dataset
class GoogleClusterDataset:
    def __init__(self, dataset_path):
        # Load dataset
        self.data = pd.read_csv(dataset_path)
        
        # Preprocess features: Standardize and normalize
        self.features = self.data[['cpu_usage', 'memory_usage', 'priority', 'task_execution_time']].values
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

        # Define task information and rewards based on task completion and resource usage
        self.tasks = self.data[['cpu_usage', 'memory_usage', 'priority', 'task_execution_time']].values
        self.num_tasks = len(self.data)
        self.current_idx = 0

    def get_next_task(self):
        # Return the next task's features as the state for the agent
        task = self.tasks[self.current_idx]
        self.current_idx = (self.current_idx + 1) % self.num_tasks
        return task

    def reset(self):
        self.current_idx = 0

# Actor-Critic Network for PPO
class PPOActorCritic(nn.Module):
    def __init__(self, state_space, action_space):
        super(PPOActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_space, 128)
        self.fc2 = nn.Linear(128, 128)

        # Actor head
        self.actor_fc = nn.Linear(128, action_space)

        # Critic head
        self.critic_fc = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

    def get_action_and_log_prob(self, state):
        x = self.forward(state)
        logits = self.actor_fc(x)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[0, action])
        return action, log_prob

    def evaluate_action(self, state, action):
        x = self.forward(state)
        logits = self.actor_fc(x)
        probs = torch.softmax(logits, dim=-1)
        log_prob = torch.log(probs.gather(1, action))
        value = self.critic_fc(x)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)  # Entropy for exploration
        return log_prob, value, entropy

# PPO Agent
class PPOAgent:
    def __init__(self, state_space, action_space):
        self.policy = PPOActorCritic(state_space, action_space)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.memory = []

    def store_transition(self, transition):
        self.memory.append(transition)

    def update_policy(self):
        # Convert memory to PyTorch tensors
        states = torch.stack([t.state for t in self.memory])
        actions = torch.tensor([t.action for t in self.memory]).unsqueeze(1)
        old_log_probs = torch.stack([t.log_prob for t in self.memory])
        rewards = [t.reward for t in self.memory]
        masks = torch.tensor([1 - int(t.done) for t in self.memory])

        # Compute returns and advantages
        returns, advantages = self.compute_advantages(rewards, masks)
        
        # PPO policy update with multiple epochs
        for _ in range(EPOCHS):
            log_probs, values, entropies = self.policy.evaluate_action(states, actions)
            ratio = torch.exp(log_probs - old_log_probs.detach())
            advantage = (returns - values).detach()

            # PPO Clipped Objective
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - values).pow(2).mean()
            entropy_loss = -ENTROPY_COEFF * entropies.mean()

            loss = actor_loss + 0.5 * critic_loss + entropy_loss  # Include entropy regularization

            # Update policy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.memory.clear()

    def compute_advantages(self, rewards, masks):
        returns = []
        G = 0
        for r, m in zip(reversed(rewards), reversed(masks)):
            G = r + GAMMA * G * m
            returns.insert(0, G)
        returns = torch.tensor(returns)
        advantages = returns - returns.mean()
        return returns, advantages

# Environment for PPO Training using Google Cluster Dataset
class PPOEnvironment:
    def __init__(self, dataset_path):
        self.dataset = GoogleClusterDataset(dataset_path)
        self.state_space = 4  # Number of features in each task (CPU, Memory, Priority, Execution Time)
        self.action_space = 3  # Actions: 0 = "Not scheduled", 1 = "Low priority", 2 = "High priority"
        self.agent = PPOAgent(self.state_space, self.action_space)

    def run_episode(self):
        state = self.dataset.get_next_task()  # Get the first task as the initial state
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        total_reward = 0
        done = False
        step_count = 0

        while not done:
            action, log_prob = self.agent.policy.get_action_and_log_prob(state_tensor)
            
            # Simulate reward based on action (dummy reward based on action)
            reward = self.simulate_task_execution(action)
            
            # Store transition
            transition = Transition(state_tensor, action, log_prob, reward, done)
            self.agent.store_transition(transition)
            
            state = self.dataset.get_next_task()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            total_reward += reward
            step_count += 1

            if step_count >= 100:  # Limit for episode length
                done = True

        # After episode ends, update policy
        self.agent.update_policy()

        return total_reward

    def simulate_task_execution(self, action):
        # Simulate reward based on task scheduling action
        # This function can be refined based on how task scheduling impacts rewards in the system
        if action == 0:  # Not scheduled
            return -1
        elif action == 1:  # Low priority
            return 1
        elif action == 2:  # High priority
            return 2
        return 0

    def train(self, num_episodes):
        for episode in range(num_episodes):
            reward = self.run_episode()
            print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {reward}")

# Data structure for storing transition tuples
Transition = namedtuple('Transition', ['state', 'action', 'log_prob', 'reward', 'done'])

if __name__ == "__main__":
    # Path to the Google 2019 Cluster Dataset
    dataset_path = "google_cluster_2019.csv"  # Replace with the actual path

    # Training setup for PPO environment
    num_episodes = 100
    ppo_env = PPOEnvironment(dataset_path)
    ppo_env.train(num_episodes)

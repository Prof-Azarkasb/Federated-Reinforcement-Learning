import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.99
ENTROPY_BETA = 0.01
BATCH_SIZE = 1024

# Load the Google 2019 Cluster Sample Dataset
class GoogleClusterDataset:
    def __init__(self, dataset_path):
        # Load the dataset (CSV file)
        self.data = pd.read_csv(dataset_path)
        
        # Data preprocessing: remove irrelevant columns and handle missing values
        self.data = self.data.dropna(subset=["resource_request", "priority", "cpu_usage_distribution"])
        self.data = self.data[['resource_request', 'priority', 'cpu_usage_distribution', 'failed', 'cpu_usage_distribution']]
        
        # Normalize the features
        self.scaler = StandardScaler()
        self.data[['resource_request', 'cpu_usage_distribution']] = self.scaler.fit_transform(self.data[['resource_request', 'cpu_usage_distribution']])

    def get_features_and_labels(self):
        # Extract features and labels
        features = self.data[['resource_request', 'cpu_usage_distribution']].values
        labels = self.data['failed'].values  # Binary classification (failed or not)
        return features, labels

# Policy Network with Two Hidden Layers
class PolicyNetwork(nn.Module):
    def __init__(self, state_space, action_space):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_space, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mu = nn.Linear(128, action_space)  # Mean of Gaussian
        self.sigma = nn.Linear(128, action_space)  # Standard deviation
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = torch.tanh(self.mu(x))  # Mean in (-1,1)
        sigma = torch.softplus(self.sigma(x)) + 1e-5  # Ensuring positive std
        return mu, sigma
    
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        mu, sigma = self.forward(state)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action.numpy(), log_prob

# Baseline Network (Value Function Approximation)
class ValueNetwork(nn.Module):
    def __init__(self, state_space):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_space, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # Linear activation

# Policy Gradient Agent
class PGAgent:
    def __init__(self, state_space, action_space):
        self.policy = PolicyNetwork(state_space, action_space)
        self.value_net = ValueNetwork(state_space)
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=LEARNING_RATE)
        self.log_probs = []
        self.rewards = []
    
    def store_transition(self, log_prob, reward):
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
    
    def update_policy(self):
        discounted_rewards = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + GAMMA * G
            discounted_rewards.insert(0, G)
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        
        policy_gradient = []
        entropy_term = 0
        for log_prob, G in zip(self.log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * G)
            entropy_term += -log_prob.exp() * log_prob  # Entropy regularization
        
        loss_policy = torch.stack(policy_gradient).sum() - ENTROPY_BETA * entropy_term
        loss_value = ((self.value_net(torch.tensor(self.rewards).float()) - discounted_rewards) ** 2).mean()
        
        self.optimizer_policy.zero_grad()
        loss_policy.backward()
        self.optimizer_policy.step()
        
        self.optimizer_value.zero_grad()
        loss_value.backward()
        self.optimizer_value.step()
        
        self.log_probs = []
        self.rewards = []

# Training Environment
class PGEnvironment:
    def __init__(self, env_name, dataset_path):
        # Initialize the environment and dataset
        self.env = gym.make(env_name)
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.shape[0]  # Continuous actions
        self.agent = PGAgent(self.state_space, self.action_space)

        # Initialize dataset
        self.dataset = GoogleClusterDataset(dataset_path)
        self.features, self.labels = self.dataset.get_features_and_labels()

    def run_episode(self):
        state, _ = self.env.reset()
        total_reward = 0
        done = False
        episode_data = []  # Store episode data for FRL
        
        # Integrate dataset for resource allocation and task scheduling
        for feature, label in zip(self.features, self.labels):
            # Use the dataset feature to adjust state or actions dynamically
            # This is an abstraction, depending on task scheduling and resource allocation needs
            action, log_prob = self.agent.policy.select_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            self.agent.store_transition(log_prob, reward)
            episode_data.append((state, action, reward, label))  # Add dataset info for FRL
            
            state = next_state
            total_reward += reward
        
        # Update policy and value networks
        self.agent.update_policy()
        
        # After each episode, print and return data
        return total_reward, episode_data
    
    def train(self, num_episodes):
        for episode in range(num_episodes):
            reward, episode_data = self.run_episode()
            print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {reward}")

            # After each episode, we could analyze the data or adjust strategies based on the dataset
            # Example of incorporating data feedback or scheduling insights
            if episode % 100 == 0:  # Example checkpoint for dataset feedback
                print(f"Episode Data (Sample): {episode_data[:2]}")  # Example print for validation

if __name__ == "__main__":
    env_name = "Pendulum-v1"  # Continuous action space
    dataset_path = "path_to_google_cluster_data.csv"  # Path to the Google Cluster Sample dataset
    num_episodes = 10000  # Training for convergence
    
    pg_env = PGEnvironment(env_name, dataset_path)
    pg_env.train(num_episodes)

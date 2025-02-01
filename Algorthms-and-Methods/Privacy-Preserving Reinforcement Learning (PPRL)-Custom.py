import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
from collections import deque
import time
from sklearn.preprocessing import StandardScaler

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
NOISE_SCALE = 0.5  # Differential privacy noise scale (σ = 0.5)
CLIPPING_THRESHOLD = 1.0  # Gradient clipping norm
TASK_COMPLETION_THRESHOLD = 0.90  # Convergence Criteria
PRIVACY_LOSS_THRESHOLD = 1.5  # Maximum allowed privacy loss (ϵ)

# Reward Function Coefficients
LAMBDA_1 = 0.3  # Resource wastage penalty
LAMBDA_2 = 1.0  # Task completion reward

# Load the Google Cluster Dataset
def load_google_cluster_data(file_path):
    # Load the dataset from the provided path (Kaggle CSV format)
    data = pd.read_csv(file_path)

    # Select relevant features (for simplicity, we'll select a few features as an example)
    features = ['cpu_usage_distribution', 'average_usage', 'resource_request', 'priority', 'failed']
    
    # Drop rows with missing values for simplicity
    data = data[features].dropna()

    # Preprocessing: Normalize the features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Split into states and rewards (simplified example)
    states = data_scaled[:, :-1]  # All features except the last column (failed)
    rewards = data_scaled[:, -1]  # The last column (failed) can be used as a simple reward proxy
    
    return states, rewards

# Privacy-Preserving Hierarchical RL Network
class PPRLNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPRLNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)  # Ensuring probability output

# Privacy-Preserving Agent
class PPRLAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.policy_net = PPRLNet(state_dim, action_dim)
        self.target_net = PPRLNet(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = 1.0
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
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
        target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

        loss = nn.functional.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        # Apply Differential Privacy with Gradient Clipping
        for param in self.policy_net.parameters():
            param.grad = self._clip_and_add_noise(param.grad)
        
        self.optimizer.step()

    def _clip_and_add_noise(self, grad):
        grad = torch.clamp(grad, -CLIPPING_THRESHOLD, CLIPPING_THRESHOLD)  # Clip gradients
        noise = torch.normal(0, NOISE_SCALE, size=grad.shape).to(grad.device)  # Add noise
        return grad + noise

    def decay_epsilon(self):
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Privacy-Preserving RL Environment
class PPRLEnvironment:
    def __init__(self, state_dim, action_dim, data_file):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent = PPRLAgent(state_dim, action_dim)
        
        # Load the dataset
        self.states, self.rewards = load_google_cluster_data(data_file)

    def run_episode(self, episode_idx):
        state = self.states[episode_idx % len(self.states)]
        total_reward = 0
        done = False
        step_count = 0

        while not done:
            action = self.agent.select_action(state)
            next_state = self.states[(episode_idx + step_count + 1) % len(self.states)]
            reward = self.rewards[episode_idx % len(self.rewards)]
            
            # Store transition
            self.agent.store_transition(state, action, reward, next_state, done)
            self.agent.optimize_model()
            
            state = next_state
            total_reward += reward
            step_count += 1

        self.agent.decay_epsilon()
        self.agent.update_target_network()

        return total_reward

    def train(self, num_episodes, data_file):
        for episode in range(num_episodes):
            reward = self.run_episode(episode)
            print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {reward}")

if __name__ == "__main__":
    state_dim = 32  # Task features + system metrics
    action_dim = 10  # Task assignment, delay adjustments, etc.
    num_episodes = 10000  # As per Table 5
    data_file = 'google_cluster_data.csv'  # Path to the dataset

    pprl_env = PPRLEnvironment(state_dim, action_dim, data_file)
    pprl_env.train(num_episodes, data_file)

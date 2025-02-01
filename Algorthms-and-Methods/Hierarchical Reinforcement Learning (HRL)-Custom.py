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
GAMMA = 0.99
MANAGER_LR = 1e-4  # Manager learning rate
WORKER_LR = 1e-4  # Worker learning rate
BATCH_SIZE = 32
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
GOAL_UPDATE_FREQUENCY = 5  # High-level agent updates goal every 5 steps
ALPHA_RI = 0.5  # Weight for intrinsic reward

# Google Cluster Dataset Path (Update with actual path)
DATASET_PATH = "google_2019_cluster_sample.csv"  # Replace with actual path

# Manager (Dilated LSTM) and Worker (Standard LSTM) Networks
class DilatedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, goal_dim):
        super(DilatedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim + goal_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # Output layer for goal prediction

    def forward(self, x, goal_embedding):
        x_goal = torch.cat((x, goal_embedding), dim=-1)  # Combine state and goal embedding
        lstm_out, _ = self.lstm(x_goal)
        return self.fc(lstm_out[:, -1, :])  # Use last LSTM output

class WorkerLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(WorkerLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # Output layer for action prediction

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Use last LSTM output

# Data Preprocessing for Google Cluster Dataset
def load_and_preprocess_data(path):
    # Load dataset
    data = pd.read_csv(path)

    # Clean the data (e.g., handle missing values, remove duplicates)
    data = data.dropna()  # Example: Drop rows with missing values
    data = data.drop_duplicates()

    # Example: Feature selection, using 'resource_request' and 'priority' as state features
    # We also normalize the data (could be done differently depending on the problem)
    data['cpu_request'] = data['resource_request'].apply(lambda x: float(x.split(' ')[0]))  # Example to extract CPU usage
    data['memory_request'] = data['resource_request'].apply(lambda x: float(x.split(' ')[1]))  # Example for memory usage

    features = data[['cpu_request', 'memory_request', 'priority']].values
    return features

# Agent class for High-Level (Manager) and Low-Level (Worker) HRL
class HRLAgent:
    def __init__(self, state_space, action_space, is_manager=False):
        self.action_space = action_space
        self.is_manager = is_manager
        
        # Use different architectures for Manager and Worker
        if self.is_manager:
            self.policy_net = DilatedLSTM(state_space, 256, 16)  # Manager uses goal embedding
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=MANAGER_LR)
        else:
            self.policy_net = WorkerLSTM(state_space, 256)  # Worker uses regular LSTM
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=WORKER_LR)

        self.memory = deque(maxlen=MEMORY_SIZE)
        self.target_net = self.policy_net  # No separate target network for now

    def select_action(self, state, goal_embedding, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_space)
        else:
            with torch.no_grad():
                if self.is_manager:
                    return self.policy_net(state, goal_embedding).argmax().item()  # Manager's goal prediction
                else:
                    return self.policy_net(state).argmax().item()  # Worker action prediction

    def store_transition(self, state, action, reward, next_state, done, goal_embedding=None):
        self.memory.append((state, action, reward, next_state, done, goal_embedding))

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones, goal_embeddings = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        goal_embeddings = torch.tensor(goal_embeddings, dtype=torch.float32)

        # Calculate intrinsic rewards (αRI)
        intrinsic_rewards = torch.norm(goal_embeddings, p=2, dim=-1)  # Example: Intrinsic reward as norm of goal

        if self.is_manager:
            # Manager's update (maximize goal prediction)
            q_values = self.policy_net(states, goal_embeddings).squeeze()
            next_q_values = self.target_net(next_states, goal_embeddings).squeeze()
        else:
            # Worker update (maximize action prediction)
            q_values = self.policy_net(states).squeeze()
            next_q_values = self.target_net(next_states).squeeze()

        # Combined reward = extrinsic (RE) + intrinsic (αRI)
        total_rewards = rewards + ALPHA_RI * intrinsic_rewards

        # Update Q-values using Bellman equation
        target_q_values = total_rewards + (GAMMA * next_q_values * (1 - dones))

        loss = nn.functional.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Hierarchical Reinforcement Learning Environment
class HRLEnvironment:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.high_level_agent = HRLAgent(self.state_space, self.action_space, is_manager=True)  # Manager
        self.low_level_agent = HRLAgent(self.state_space, self.action_space, is_manager=False)   # Worker
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # Load Google Cluster Dataset and preprocess
        self.cluster_data = load_and_preprocess_data(DATASET_PATH)

    def run_episode(self):
        state = self.env.reset()
        done = False
        total_reward = 0
        steps = 0
        start_time = time.time()

        # Metric trackers
        data_sent = 0
        computational_effort = 0
        delays = []

        while not done:
            # Manager determines a goal every GOAL_UPDATE_FREQUENCY steps
            if steps % GOAL_UPDATE_FREQUENCY == 0:
                goal_embedding = self.high_level_agent.select_action(state, state, self.epsilon)  # Get goal from Manager

            # Worker tries to achieve the goal set by the Manager
            action = self.low_level_agent.select_action(state, goal_embedding, self.epsilon)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            steps += 1

            # Store transition and optimize both agents
            self.high_level_agent.store_transition(state, action, reward, next_state, done, goal_embedding)
            self.low_level_agent.store_transition(state, action, reward, next_state, done, goal_embedding)
            self.high_level_agent.optimize_model()
            self.low_level_agent.optimize_model()

            # Update metrics
            data_sent += 0.05  # Dummy communication overhead in MB/s per action
            computational_effort += 0.02  # Dummy GFLOPS per action
            delays.append(0.6)  # Dummy delay per step

            state = next_state

        # Calculate evaluation metrics
        task_completion_time = time.time() - start_time
        convergence_rate = steps / task_completion_time if task_completion_time > 0 else 0
        communication_overhead = data_sent / task_completion_time if task_completion_time > 0 else 0
        avg_delay = sum(delays) / len(delays) if delays else 0

        print(f"Episode Metrics:")
        print(f"  Total Reward: {total_reward}")
        print(f"  Convergence Rate (Episodes/sec): {convergence_rate:.3f}")
        print(f"  Communication Overhead (MB/sec): {communication_overhead:.3f}")
        print(f"  Computational Efficiency (GFLOPS): {computational_effort:.3f}")
        print(f"  Task Completion Time (Seconds): {task_completion_time:.3f}")
        print(f"  Average Delay (ms): {avg_delay:.3f}")

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return total_reward

    def train(self, num_episodes):
        for episode in range(num_episodes):
            reward = self.run_episode()
            print(f"Episode {episode + 1}/{num_episodes} - Reward: {reward}, Epsilon: {self.epsilon:.3f}")

if __name__ == "__main__":
    env_name = "CartPole-v1"  # Example environment
    num_episodes = 100  # Number of episodes

    hrl_environment = HRLEnvironment(env_name)
    hrl_environment.train(num_episodes)

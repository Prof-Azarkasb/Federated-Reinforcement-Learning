import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import gym
from collections import deque
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
MEMORY_SIZE = 10000
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05
CONVERGENCE_CRITERIA = 0.001

# Load and preprocess the dataset
def load_and_preprocess_data():
    # Load the Google 2019 Cluster Sample dataset
    df = pd.read_csv("google_2019_cluster_sample.csv")

    # Feature Engineering: Selecting important features
    features = ['time', 'resource_request', 'cpu_usage_distribution', 'priority', 'average_usage']
    target = 'failed'  # The outcome we're predicting (success or failure of task execution)
    
    # Select relevant features and target variable
    X = df[features]
    y = df[target]
    
    # Handle missing values (if any)
    X = X.fillna(X.mean())
    
    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# DQN Network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Deep Q-Network Agent
class DQNAgent:
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.policy_net = DQN(state_space, action_space)
        self.target_net = DQN(state_space, action_space)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = 1.0
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.loss_criterion = nn.MSELoss()

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
        target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

        loss = self.loss_criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if loss.item() < CONVERGENCE_CRITERIA:
            print("Model has converged.")
            return

    def decay_epsilon(self):
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# DQN Environment
class DQNEnvironment:
    def __init__(self, env_name, X_train, X_test, y_train, y_test):
        self.env = gym.make(env_name)
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.agent = DQNAgent(self.state_space, self.action_space)

        # Load dataset for task scheduling and resource allocation
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def run_episode(self):
        state = self.env.reset()
        total_reward = 0
        done = False
        start_time = time.time()
        step_count = 0
        delay = []

        while not done:
            action = self.agent.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.agent.store_transition(state, action, reward, next_state, done)
            self.agent.optimize_model()
            state = next_state
            total_reward += reward
            step_count += 1
            delay.append(1)  # Example delay in ms

        task_completion_time = time.time() - start_time
        avg_delay = sum(delay) / len(delay) if delay else 0

        print(f"Episode Metrics:")
        print(f"  Total Reward: {total_reward}")
        print(f"  Task Completion Time (sec): {task_completion_time:.3f}")
        print(f"  Average Delay (ms): {avg_delay:.2f}")

        self.agent.decay_epsilon()
        self.agent.update_target_network()

        return total_reward

    def train(self, num_episodes):
        for episode in range(num_episodes):
            reward = self.run_episode()
            print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {reward}")

if __name__ == "__main__":
    env_name = "CartPole-v1"
    num_episodes = 100

    # Load and preprocess dataset
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    dqn_env = DQNEnvironment(env_name, X_train, X_test, y_train, y_test)
    dqn_env.train(num_episodes)

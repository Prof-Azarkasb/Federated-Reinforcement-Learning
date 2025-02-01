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
LEARNING_RATE = 0.0001  # Fine-tuning rate
BATCH_SIZE = 32
MEMORY_SIZE = 10000
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
TRANSFER_SAMPLES = 10000  # Number of transfer samples
TRAINING_EPISODES = 500  # As specified in the table

# DQN Network Architecture
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Transfer Learning Agent
class TransferLearningAgent:
    def __init__(self, state_space, action_space, source_model=None):
        self.action_space = action_space
        self.policy_net = DQN(state_space, action_space)
        self.target_net = DQN(state_space, action_space)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = 1.0
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Transfer Learning: Load source model weights
        if source_model:
            self.policy_net.load_state_dict(source_model.state_dict())
            self.target_net.load_state_dict(source_model.state_dict())
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)

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

        loss = nn.functional.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Load Google 2019 Cluster Dataset
def load_google_cluster_dataset(file_path):
    dataset = pd.read_csv(file_path)
    dataset = dataset.dropna()  # Drop missing values for simplicity
    dataset = dataset.sample(frac=1).reset_index(drop=True)  # Shuffle dataset
    return dataset

# Data Preprocessing (Feature Engineering)
def preprocess_data(dataset):
    features = dataset[['cpu_usage_distribution', 'resource_request', 'priority', 'failed']]  # Example features
    labels = dataset['task_id']  # Example label (for supervised tasks, can change as per requirement)
    # Scale/Normalize data as necessary
    return features, labels

# Transfer Learning Environment with Dataset Integration
class TransferLearningEnvironment:
    def __init__(self, env_name, source_model=None, dataset=None):
        self.env = gym.make(env_name)
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.agent = TransferLearningAgent(self.state_space, self.action_space, source_model)
        self.dataset = dataset

    def run_episode(self):
        state = self.env.reset()
        total_reward = 0
        done = False
        start_time = time.time()

        while not done:
            action = self.agent.select_action(state)
            next_state, reward, done, _ = self.env.step(action)

            # Reward Structure
            task_completion_bonus = 10 if reward > 0 else 0
            balanced_usage_bonus = 5 if np.random.rand() > 0.5 else 0
            missed_deadline_penalty = -10 if reward < 0 else 0
            overutilization_penalty = -5 if np.random.rand() > 0.9 else 0

            adjusted_reward = reward + task_completion_bonus + balanced_usage_bonus + missed_deadline_penalty + overutilization_penalty

            self.agent.store_transition(state, action, adjusted_reward, next_state, done)
            self.agent.optimize_model()
            state = next_state
            total_reward += adjusted_reward

        # Convergence Criteria
        task_completion_time = time.time() - start_time
        print(f"Episode Reward: {total_reward}, Task Completion Time: {task_completion_time:.2f}s")

        self.agent.decay_epsilon()
        self.agent.update_target_network()

        return total_reward

    def train(self, num_episodes):
        for episode in range(num_episodes):
            reward = self.run_episode()
            print(f"Episode {episode + 1}/{num_episodes} - Reward: {reward}")

# Pretrain Source Model
def pretrain_source_model(env_name, num_pretrain_episodes, dataset=None):
    source_env = gym.make(env_name)
    source_agent = TransferLearningAgent(source_env.observation_space.shape[0], source_env.action_space.n)
    
    # Pretrain using the dataset if available
    if dataset is not None:
        features, _ = preprocess_data(dataset)
        for feature in features.itertuples():
            state = np.array(feature[1:])  # Convert feature data to state format
            done = False
            while not done:
                action = source_agent.select_action(state)
                next_state = state  # For simplicity, use state as next_state in this context
                reward = random.random()  # Reward can be set based on task success/failure
                source_agent.store_transition(state, action, reward, next_state, done)
                source_agent.optimize_model()
                state = next_state
            source_agent.decay_epsilon()

    return source_agent.policy_net  # Return pretrained model

if __name__ == "__main__":
    # Load dataset
    dataset_path = "path_to_google_2019_cluster_sample.csv"  # Replace with the actual path
    dataset = load_google_cluster_dataset(dataset_path)

    # Pretrain a model on the source task using the dataset
    source_env_name = "CartPole-v0"
    num_pretrain_episodes = 1000  # Pretrain for 1000 episodes
    source_model = pretrain_source_model(source_env_name, num_pretrain_episodes, dataset)
    
    # Transfer Learning on the target task
    target_env_name = "CartPole-v1"
    num_target_episodes = TRAINING_EPISODES  # 500 episodes for target training

    transfer_env = TransferLearningEnvironment(target_env_name, source_model, dataset)
    transfer_env.train(num_target_episodes)

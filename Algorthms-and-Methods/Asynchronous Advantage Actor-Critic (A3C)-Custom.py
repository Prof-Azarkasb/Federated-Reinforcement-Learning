import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import pandas as pd
import gym
import numpy as np
import time
from collections import namedtuple

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.0001
NUM_WORKERS = 16  # Parallel agents
EPISODES_PER_WORKER = 10000  # Adjusted for more episodes
BETA = 0.01  # Entropy regularization coefficient for exploration
DISCOUNT_FACTOR = 0.99  # Discount factor for long-term rewards

# Actor-Critic Network
class ActorCriticNetwork(nn.Module):
    def __init__(self, state_space, action_space):
        super(ActorCriticNetwork, self).__init__()
        self.fc = nn.Linear(state_space, 128)
        
        # Actor head
        self.actor_head = nn.Linear(128, action_space)
        
        # Critic head
        self.critic_head = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return x

    def get_action_and_value(self, state):
        x = self.forward(state)
        logits = self.actor_head(x)
        value = self.critic_head(x)
        action_probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(action_probs, 1).item()
        return action, value, torch.log(action_probs[action])

# Custom Environment using Google Cluster Dataset
class ClusterTaskEnvironment:
    def __init__(self, dataset_path):
        self.dataset = pd.read_csv(dataset_path)  # Load the dataset
        self.state_space = len(self.dataset.columns) - 1  # Exclude the target column
        self.action_space = 2  # Example: Two possible actions, e.g., "assign" or "skip"
        self.current_index = 0  # To keep track of the current sample in the dataset
    
    def reset(self):
        self.current_index = 0
        return self._get_state(self.current_index)
    
    def step(self, action):
        # Simulate a step: Extract resource usage, CPU, memory, etc. from the dataset
        state = self._get_state(self.current_index)
        
        # Dummy reward calculation: You can improve this with your domain-specific logic
        reward = self._calculate_reward(state, action)
        
        # Move to the next task in the dataset
        self.current_index += 1
        done = self.current_index >= len(self.dataset)
        
        if done:
            self.current_index = 0  # Reset to the beginning of the dataset for the next episode
            
        next_state = self._get_state(self.current_index)
        return next_state, reward, done, {}
    
    def _get_state(self, index):
        # Here you extract the features you want from the dataset, for example:
        task_data = self.dataset.iloc[index]
        state = task_data.drop("reward")  # Drop the target column (reward, for example)
        return np.array(state, dtype=np.float32)
    
    def _calculate_reward(self, state, action):
        # Define a simple reward mechanism based on the task and action
        # For example, a reward based on CPU and memory usage, job priority, etc.
        cpu_usage = state["cpu_usage"]
        memory_usage = state["memory_usage"]
        reward = 1 if action == 0 else -1  # Dummy logic; you can replace with more complex logic
        return reward

# Worker class for A3C with the custom environment
class A3CWorker(mp.Process):
    def __init__(self, global_model, optimizer, env, worker_id):
        super(A3CWorker, self).__init__()
        self.global_model = global_model
        self.optimizer = optimizer
        self.local_model = ActorCriticNetwork(global_model.fc.in_features, global_model.actor_head.out_features)
        self.env = env
        self.worker_id = worker_id
        self.total_rewards = []

    def run(self):
        for episode in range(EPISODES_PER_WORKER):
            state = self.env.reset()
            done = False
            episode_reward = 0
            start_time = time.time()
            
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action, value, log_prob = self.local_model.get_action_and_value(state_tensor)
                
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                
                # Advantage Calculation
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                _, next_value, _ = self.local_model.get_action_and_value(next_state_tensor)
                advantage = reward + GAMMA * next_value * (1 - int(done)) - value

                # Loss calculation with Entropy Regularization
                actor_loss = -log_prob * advantage.detach()
                critic_loss = advantage.pow(2)
                entropy_loss = -BETA * torch.sum(torch.softmax(log_prob, dim=-1) * log_prob)
                loss = actor_loss + critic_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                
                # Copy local gradients to the global model
                for local_param, global_param in zip(self.local_model.parameters(), self.global_model.parameters()):
                    global_param._grad = local_param.grad

                # Step optimizer to update global model
                self.optimizer.step()
                
                state = next_state

            # Print episode metrics
            task_completion_time = time.time() - start_time
            print(f"Worker {self.worker_id}, Episode {episode + 1}/{EPISODES_PER_WORKER}")
            print(f"  Total Reward: {episode_reward}")
            print(f"  Task Completion Time (sec): {task_completion_time:.3f}")

            self.total_rewards.append(episode_reward)

# Main A3C Training Function
def train_a3c(dataset_path):
    env = ClusterTaskEnvironment(dataset_path)
    state_space = len(env.dataset.columns) - 1  # excluding the reward column
    action_space = env.action_space
    global_model = ActorCriticNetwork(state_space, action_space)
    global_model.share_memory()
    
    optimizer = optim.Adam(global_model.parameters(), lr=LEARNING_RATE)

    # Start multiple worker processes
    workers = [A3CWorker(global_model, optimizer, env, worker_id=i) for i in range(NUM_WORKERS)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

if __name__ == "__main__":
    dataset_path = "google_2019_cluster_sample.csv"  # Path to the dataset file
    train_a3c(dataset_path)

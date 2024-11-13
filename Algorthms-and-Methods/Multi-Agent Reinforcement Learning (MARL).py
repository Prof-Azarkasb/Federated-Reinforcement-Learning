import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from collections import deque
import random
import time

# Hyperparameters
GAMMA = 0.95
LEARNING_RATE = 0.001
BUFFER_SIZE = 10000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 10
NUM_AGENTS = 2
EPISODES = 200

# Q-Network for Agents
class QNetwork(nn.Module):
    def __init__(self, state_space, action_space):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_space, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_space)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay buffer for experience replay
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# Multi-Agent Environment Wrapper
class MultiAgentEnvironment:
    def __init__(self, env_name, num_agents):
        self.env = gym.make(env_name)
        self.num_agents = num_agents

    def reset(self):
        states = [self.env.reset() for _ in range(self.num_agents)]
        return states

    def step(self, actions):
        next_states, rewards, dones, infos = [], [], [], []
        for action in actions:
            state, reward, done, info = self.env.step(action)
            next_states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return next_states, rewards, dones, infos

# Agent for MARL
class Agent:
    def __init__(self, state_space, action_space):
        self.q_network = QNetwork(state_space, action_space)
        self.target_network = QNetwork(state_space, action_space)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.action_space = action_space
        self.steps = 0

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice(range(self.action_space))
        else:
            with torch.no_grad():
                return torch.argmax(self.q_network(torch.tensor(state, dtype=torch.float32))).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.buffer.add((state, action, reward, next_state, done))

    def update_q_network(self):
        if self.buffer.size() < BATCH_SIZE:
            return
        
        # Sample a batch
        batch = self.buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Q-Learning target
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + GAMMA * max_next_q_values * (1 - dones)

        # Loss and backpropagation
        loss = nn.functional.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Multi-Agent Training Loop
def train_marl(env_name):
    env = MultiAgentEnvironment(env_name, NUM_AGENTS)
    agents = [Agent(env.env.observation_space.shape[0], env.env.action_space.n) for _ in range(NUM_AGENTS)]

    # Metrics Tracking
    start_time = time.time()
    rewards_per_episode = []
    communication_overhead = 0.0  # Placeholder for communication overhead

    for episode in range(EPISODES):
        states = env.reset()
        episode_rewards = np.zeros(NUM_AGENTS)
        done = False
        steps = 0
        epsilon = max(0.1, 1 - episode / (EPISODES / 2))  # Epsilon decay

        while not done:
            actions = [agent.select_action(state, epsilon) for agent, state in zip(agents, states)]
            next_states, rewards, dones, _ = env.step(actions)

            # Store experiences and accumulate rewards
            for i, agent in enumerate(agents):
                agent.store_experience(states[i], actions[i], rewards[i], next_states[i], dones[i])
                episode_rewards[i] += rewards[i]
            
            states = next_states
            steps += 1
            done = any(dones)

            # Agent Updates
            for agent in agents:
                agent.update_q_network()
            
            # Communication Overhead Tracking
            communication_overhead += steps / NUM_AGENTS  # Placeholder metric

        rewards_per_episode.append(np.mean(episode_rewards))
        
        # Update target networks occasionally
        if episode % TARGET_UPDATE_FREQ == 0:
            for agent in agents:
                agent.update_target_network()

        # Metrics at episode end
        avg_reward = np.mean(episode_rewards)
        avg_task_completion_time = time.time() - start_time
        avg_decision_accuracy = (np.sum(episode_rewards) / steps) * 100
        avg_computational_efficiency = steps / avg_task_completion_time  # Placeholder GFLOPS

        # Print episode metrics
        print(f"Episode {episode + 1}/{EPISODES}")
        print(f"  Avg Reward per Agent: {avg_reward:.2f}")
        print(f"  Avg Task Completion Time: {avg_task_completion_time:.2f} seconds")
        print(f"  Avg Decision Accuracy: {avg_decision_accuracy:.2f}%")
        print(f"  Avg Computational Efficiency (GFLOPS): {avg_computational_efficiency:.2f}")
        print(f"  Communication Overhead (MB/sec): {communication_overhead:.2f}")

    # Final performance metrics
    print("\nTraining Complete")
    print(f"Average Reward per Episode: {np.mean(rewards_per_episode):.2f}")
    print(f"Total Time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    env_name = "CartPole-v1"  # You can replace with a multi-agent compatible environment
    train_marl(env_name)

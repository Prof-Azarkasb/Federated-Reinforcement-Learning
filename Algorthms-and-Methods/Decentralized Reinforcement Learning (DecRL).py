import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import gym
from collections import deque
import time

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 32
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
NUM_AGENTS = 3  # Number of decentralized agents

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

# Decentralized Agent
class Agent:
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.policy_net = DQN(state_space, action_space)
        self.target_net = DQN(state_space, action_space)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

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

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Decentralized Environment with Multiple Agents
class DecentralizedEnvironment:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.agents = [Agent(self.state_space, self.action_space) for _ in range(NUM_AGENTS)]

    def run_episode(self):
        states = [self.env.reset() for _ in range(NUM_AGENTS)]
        dones = [False] * NUM_AGENTS
        total_rewards = [0] * NUM_AGENTS
        start_time = time.time()

        # Metrics initialization
        steps = 0
        data_sent = 0
        computational_effort = 0
        delays = []

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

                    # Update metrics for each action
                    data_sent += 0.05  # Simulated MB/sec per action
                    computational_effort += 0.02  # Simulated GFLOPS per action
                    delays.append(0.6)  # Dummy delay in ms per step

        # Calculate evaluation metrics at the end of the episode
        task_completion_time = time.time() - start_time
        convergence_rate = steps / task_completion_time if task_completion_time > 0 else 0
        communication_overhead = data_sent / task_completion_time if task_completion_time > 0 else 0
        avg_delay = sum(delays) / len(delays) if delays else 0

        # Log metrics after each episode
        print("Episode Metrics for all Agents:")
        print(f"  Total Reward (all agents): {sum(total_rewards)}")
        print(f"  Convergence Rate (Episodes/sec): {convergence_rate:.3f}")
        print(f"  Communication Overhead (MB/sec): {communication_overhead:.3f}")
        print(f"  Computational Efficiency (GFLOPS): {computational_effort:.3f}")
        print(f"  Task Completion Time (Seconds): {task_completion_time:.3f}")
        print(f"  Average Delay (ms): {avg_delay:.3f}")

        # Epsilon decay and target network update for all agents
        for agent in self.agents:
            agent.decay_epsilon()
            agent.update_target_network()

        return sum(total_rewards)

    def train(self, num_episodes):
        for episode in range(num_episodes):
            reward = self.run_episode()
            print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {reward}")

if __name__ == "__main__":
    env_name = "CartPole-v1"  # Environment name
    num_episodes = 100  # Number of episodes

    decentralized_env = DecentralizedEnvironment(env_name)
    decentralized_env.train(num_episodes)

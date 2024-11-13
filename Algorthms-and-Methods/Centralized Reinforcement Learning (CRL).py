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
MEMORY_SIZE = 10000
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

# Centralized DQN Network
class CentralizedDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CentralizedDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Centralized Reinforcement Learning Agent
class CentralizedRLAgent:
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.policy_net = CentralizedDQN(state_space, action_space)
        self.target_net = CentralizedDQN(state_space, action_space)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = 1.0
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def select_action(self, state):
        # Epsilon-greedy action selection
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

# Centralized Environment
class CentralizedEnvironment:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.agent = CentralizedRLAgent(self.state_space, self.action_space)

    def run_episode(self):
        state = self.env.reset()
        total_reward = 0
        done = False
        start_time = time.time()

        # Metrics
        step_count = 0
        communication_overhead = 0
        computational_efficiency = 0
        decision_accuracy = 0
        delay = []

        while not done:
            action = self.agent.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.agent.store_transition(state, action, reward, next_state, done)
            self.agent.optimize_model()
            state = next_state
            total_reward += reward
            step_count += 1

            # Metric calculations
            communication_overhead += 0.05  # Example MB/sec for centralization
            computational_efficiency += 0.02  # Example GFLOPS per step
            decision_accuracy += (1 if reward > 0 else 0) / step_count
            delay.append(1)  # Example delay in ms

        # Episode metrics
        task_completion_time = time.time() - start_time
        avg_delay = sum(delay) / len(delay) if delay else 0

        # Print metrics after each episode
        print(f"Episode Metrics:")
        print(f"  Total Reward: {total_reward}")
        print(f"  Communication Overhead (MB/sec): {communication_overhead:.2f}")
        print(f"  Computational Efficiency (GFLOPS): {computational_efficiency:.2f}")
        print(f"  Decision Accuracy (%): {decision_accuracy * 100:.2f}")
        print(f"  Task Completion Time (sec): {task_completion_time:.3f}")
        print(f"  Average Delay (ms): {avg_delay:.2f}")

        # Decay epsilon and update target network
        self.agent.decay_epsilon()
        self.agent.update_target_network()

        return total_reward

    def train(self, num_episodes):
        for episode in range(num_episodes):
            reward = self.run_episode()
            print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {reward}")

if __name__ == "__main__":
    # Centralized environment setup and training
    env_name = "CartPole-v1"
    num_episodes = 100

    centralized_env = CentralizedEnvironment(env_name)
    centralized_env.train(num_episodes)

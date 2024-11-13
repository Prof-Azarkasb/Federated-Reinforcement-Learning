import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import time

# Hyperparameters
LEARNING_RATE = 0.01
GAMMA = 0.99

# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_space, action_space):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_space, 128)
        self.fc2 = nn.Linear(128, action_space)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        action = torch.multinomial(probs, 1).item()
        return action, torch.log(probs[0, action])

# Policy Gradient Agent
class PGAgent:
    def __init__(self, state_space, action_space):
        self.policy = PolicyNetwork(state_space, action_space)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.log_probs = []
        self.rewards = []

    def store_transition(self, log_prob, reward):
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def update_policy(self):
        # Discounted reward calculation
        discounted_rewards = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + GAMMA * G
            discounted_rewards.insert(0, G)
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        # Policy Gradient update
        policy_gradient = []
        for log_prob, G in zip(self.log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * G)
        loss = torch.stack(policy_gradient).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reset after each episode
        self.log_probs = []
        self.rewards = []

# Environment for Policy Gradient Training
class PGEnvironment:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.agent = PGAgent(self.state_space, self.action_space)

    def run_episode(self):
        state = self.env.reset()
        total_reward = 0
        done = False
        start_time = time.time()

        # Metrics tracking
        step_count = 0
        decision_accuracy = 0
        computational_efficiency = 0
        delays = []

        while not done:
            action, log_prob = self.agent.policy.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.agent.store_transition(log_prob, reward)
            state = next_state
            total_reward += reward
            step_count += 1

            # Metrics calculation (examples)
            computational_efficiency += 0.02  # Placeholder GFLOPS per step
            decision_accuracy += (1 if reward > 0 else 0) / step_count
            delays.append(1)  # Example delay in ms

        # Episode metrics
        task_completion_time = time.time() - start_time
        avg_delay = sum(delays) / len(delays) if delays else 0

        # Print metrics
        print(f"Episode Metrics:")
        print(f"  Total Reward: {total_reward}")
        print(f"  Computational Efficiency (GFLOPS): {computational_efficiency:.2f}")
        print(f"  Decision Accuracy (%): {decision_accuracy * 100:.2f}")
        print(f"  Task Completion Time (sec): {task_completion_time:.3f}")
        print(f"  Average Delay (ms): {avg_delay:.2f}")

        # Policy update
        self.agent.update_policy()

        return total_reward

    def train(self, num_episodes):
        for episode in range(num_episodes):
            reward = self.run_episode()
            print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {reward}")

if __name__ == "__main__":
    # Training setup for Policy Gradient environment
    env_name = "CartPole-v1"
    num_episodes = 100

    pg_env = PGEnvironment(env_name)
    pg_env.train(num_episodes)

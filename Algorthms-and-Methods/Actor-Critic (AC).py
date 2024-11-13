import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import time

# Hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.99

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_space, action_space):
        super(Actor, self).__init__()
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

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_space):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_space, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Actor-Critic Agent
class ACAgent:
    def __init__(self, state_space, action_space):
        self.actor = Actor(state_space, action_space)
        self.critic = Critic(state_space)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)

    def update(self, log_prob, reward, state_value, next_state_value):
        # Calculate advantage
        advantage = reward + GAMMA * next_state_value - state_value

        # Actor loss
        actor_loss = -log_prob * advantage

        # Critic loss
        critic_loss = advantage.pow(2)

        # Update actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

# Environment for Actor-Critic Training
class ACEnvironment:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.agent = ACAgent(self.state_space, self.action_space)

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
            # Select action and get state value from critic
            action, log_prob = self.agent.actor.select_action(state)
            state_value = self.agent.critic(torch.from_numpy(state).float().unsqueeze(0))
            next_state, reward, done, _ = self.env.step(action)

            # Get next state value for advantage calculation
            next_state_value = self.agent.critic(torch.from_numpy(next_state).float().unsqueeze(0)) if not done else torch.tensor(0.0)

            # Update agent
            self.agent.update(log_prob, reward, state_value, next_state_value)

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

        return total_reward

    def train(self, num_episodes):
        for episode in range(num_episodes):
            reward = self.run_episode()
            print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {reward}")

if __name__ == "__main__":
    # Training setup for Actor-Critic environment
    env_name = "CartPole-v1"
    num_episodes = 100

    ac_env = ACEnvironment(env_name)
    ac_env.train(num_episodes)

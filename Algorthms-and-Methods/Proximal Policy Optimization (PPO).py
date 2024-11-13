import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import time
from collections import namedtuple

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.0003
CLIP_EPS = 0.2
BATCH_SIZE = 64
EPOCHS = 10

# Actor-Critic Network for PPO
class PPOActorCritic(nn.Module):
    def __init__(self, state_space, action_space):
        super(PPOActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_space, 128)
        
        # Actor head
        self.actor_fc = nn.Linear(128, action_space)
        
        # Critic head
        self.critic_fc = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return x

    def get_action_and_log_prob(self, state):
        x = self.forward(state)
        logits = self.actor_fc(x)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[0, action])
        return action, log_prob

    def evaluate_action(self, state, action):
        x = self.forward(state)
        logits = self.actor_fc(x)
        probs = torch.softmax(logits, dim=-1)
        log_prob = torch.log(probs.gather(1, action))
        value = self.critic_fc(x)
        return log_prob, value

# PPO Agent
class PPOAgent:
    def __init__(self, state_space, action_space):
        self.policy = PPOActorCritic(state_space, action_space)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.memory = []

    def store_transition(self, transition):
        self.memory.append(transition)

    def update_policy(self):
        # Convert memory to PyTorch tensors
        states = torch.stack([t.state for t in self.memory])
        actions = torch.tensor([t.action for t in self.memory]).unsqueeze(1)
        old_log_probs = torch.stack([t.log_prob for t in self.memory])
        rewards = [t.reward for t in self.memory]
        masks = torch.tensor([1 - int(t.done) for t in self.memory])

        # Compute returns and advantages
        returns, advantages = self.compute_advantages(rewards, masks)
        
        # PPO policy update with multiple epochs
        for _ in range(EPOCHS):
            log_probs, values = self.policy.evaluate_action(states, actions)
            ratio = torch.exp(log_probs - old_log_probs.detach())
            advantage = (returns - values).detach()
            
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - values).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss

            # Update policy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.memory.clear()

    def compute_advantages(self, rewards, masks):
        returns = []
        G = 0
        for r, m in zip(reversed(rewards), reversed(masks)):
            G = r + GAMMA * G * m
            returns.insert(0, G)
        returns = torch.tensor(returns)
        advantages = returns - returns.mean()
        return returns, advantages

# Environment for PPO Training
class PPOEnvironment:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.agent = PPOAgent(self.state_space, self.action_space)

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
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            action, log_prob = self.agent.policy.get_action_and_log_prob(state_tensor)
            next_state, reward, done, _ = self.env.step(action)
            
            transition = Transition(state_tensor, action, log_prob, reward, done)
            self.agent.store_transition(transition)

            state = next_state
            total_reward += reward
            step_count += 1

            # Example metrics
            computational_efficiency += 0.02  # Placeholder GFLOPS
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
            self.agent.update_policy()
            print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {reward}")

# Data structure for storing transition tuples
Transition = namedtuple('Transition', ['state', 'action', 'log_prob', 'reward', 'done'])

if __name__ == "__main__":
    # Training setup for PPO environment
    env_name = "CartPole-v1"
    num_episodes = 100

    ppo_env = PPOEnvironment(env_name)
    ppo_env.train(num_episodes)

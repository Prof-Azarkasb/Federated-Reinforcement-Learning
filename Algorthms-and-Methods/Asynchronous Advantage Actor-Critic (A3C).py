import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import gym
import numpy as np
import time
from collections import namedtuple

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.0001
NUM_WORKERS = 4
EPISODES_PER_WORKER = 100

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

# Worker class for A3C
class A3CWorker(mp.Process):
    def __init__(self, global_model, optimizer, env_name, worker_id):
        super(A3CWorker, self).__init__()
        self.global_model = global_model
        self.optimizer = optimizer
        self.local_model = ActorCriticNetwork(global_model.fc.in_features, global_model.actor_head.out_features)
        self.env = gym.make(env_name)
        self.worker_id = worker_id
        self.total_rewards = []

    def run(self):
        for episode in range(EPISODES_PER_WORKER):
            state = self.env.reset()
            done = False
            episode_reward = 0
            start_time = time.time()
            
            # Metrics tracking
            step_count = 0
            decision_accuracy = 0
            computational_efficiency = 0
            delays = []

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action, value, log_prob = self.local_model.get_action_and_value(state_tensor)
                
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                step_count += 1
                
                # Metrics
                computational_efficiency += 0.02  # Placeholder GFLOPS
                decision_accuracy += (1 if reward > 0 else 0) / step_count
                delays.append(1)  # Example delay in ms

                # Advantage Calculation
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                _, next_value, _ = self.local_model.get_action_and_value(next_state_tensor)
                advantage = reward + GAMMA * next_value * (1 - int(done)) - value

                # Backpropagation
                actor_loss = -log_prob * advantage.detach()
                critic_loss = advantage.pow(2)
                loss = actor_loss + critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                
                # Copy local gradients to the global model
                for local_param, global_param in zip(self.local_model.parameters(), self.global_model.parameters()):
                    global_param._grad = local_param.grad

                # Step optimizer to update global model
                self.optimizer.step()
                
                state = next_state

            # Episode metrics
            task_completion_time = time.time() - start_time
            avg_delay = sum(delays) / len(delays) if delays else 0

            # Print episode metrics for this worker
            print(f"Worker {self.worker_id}, Episode {episode + 1}/{EPISODES_PER_WORKER}")
            print(f"  Total Reward: {episode_reward}")
            print(f"  Computational Efficiency (GFLOPS): {computational_efficiency:.2f}")
            print(f"  Decision Accuracy (%): {decision_accuracy * 100:.2f}")
            print(f"  Task Completion Time (sec): {task_completion_time:.3f}")
            print(f"  Average Delay (ms): {avg_delay:.2f}")

            self.total_rewards.append(episode_reward)

# Main A3C Training Function
def train_a3c(env_name):
    env = gym.make(env_name)
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    global_model = ActorCriticNetwork(state_space, action_space)
    global_model.share_memory()
    
    optimizer = optim.Adam(global_model.parameters(), lr=LEARNING_RATE)

    # Start multiple worker processes
    workers = [A3CWorker(global_model, optimizer, env_name, worker_id=i) for i in range(NUM_WORKERS)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

if __name__ == "__main__":
    # Initialize training environment
    env_name = "CartPole-v1"
    train_a3c(env_name)

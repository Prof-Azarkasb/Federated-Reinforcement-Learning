import numpy as np
import random
import time
import pandas as pd
from collections import defaultdict
from typing import List, Dict

# Define parameters
NUM_AGENTS = 5
NUM_EPISODES = 2000
LEARNING_RATE_ACTOR = 0.001
LEARNING_RATE_CRITIC = 0.002
DISCOUNT_FACTOR = 0.9
BATCH_SIZE = 64
CONVERGENCE_CRITERIA = {"task_time": 90, "energy_reduction": 10, "throughput": 12}
SYNC_INTERVAL = 10  # Period for agents to synchronize their policies
DATA_FILE_PATH = 'google_2019_cluster_sample.csv'  # Path to the dataset

# Load and preprocess the dataset
class Dataset:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.preprocess_data()

    def preprocess_data(self):
        # Data cleaning, handling missing values, outliers, and irrelevant features
        self.data = self.data.dropna()  # Removing rows with missing values
        self.data = self.data[self.data['failed'] == 0]  # Filtering out failed tasks

        # Feature Engineering: Normalize relevant features (for simplicity, we normalize CPU and memory usage)
        self.data['normalized_cpu_usage'] = (self.data['cpu_usage_distribution'] - self.data['cpu_usage_distribution'].min()) / \
                                             (self.data['cpu_usage_distribution'].max() - self.data['cpu_usage_distribution'].min())
        self.data['normalized_memory_usage'] = (self.data['average_usage'] - self.data['average_usage'].min()) / \
                                                (self.data['average_usage'].max() - self.data['average_usage'].min())
        
        # Select relevant columns for task attributes and resource usage
        self.data = self.data[['time', 'instance_events_type', 'scheduling_class', 'priority', 
                               'resource_request', 'normalized_cpu_usage', 'normalized_memory_usage']]

    def get_task_batch(self, batch_size):
        # Randomly sample a batch of tasks
        return self.data.sample(batch_size)

# Define a multi-agent environment
class Environment:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.state = None
        self.reset()

    def reset(self):
        self.state = self._generate_state()
        return self.state

    def _generate_state(self):
        task = self.dataset.get_task_batch(1).iloc[0]
        return {
            "task_size": random.randint(10, 100),  # Example task size (could be from dataset)
            "deadline": random.randint(1, 50),
            "cpu_status": task['normalized_cpu_usage'],
            "memory_status": task['normalized_memory_usage'],
            "queue_lengths": [random.randint(0, 10) for _ in range(NUM_AGENTS)],
            "delays": [random.uniform(0.1, 2.0) for _ in range(NUM_AGENTS)],
        }

    def step(self, agent_action):
        reward = self._calculate_reward(agent_action)
        self.state = self._generate_state()
        done = self._check_convergence()
        return self.state, reward, done

    def _calculate_reward(self, action):
        reward = 20 if action == "offload" else -5
        if self.state["queue_lengths"][0] > 5:
            reward -= 10
        return reward

    def _check_convergence(self):
        return (self.state["task_size"] <= CONVERGENCE_CRITERIA["task_time"] and
                random.uniform(0, 100) >= CONVERGENCE_CRITERIA["energy_reduction"] and
                random.uniform(0, 100) >= CONVERGENCE_CRITERIA["throughput"])

# Define a multi-agent system
class Agent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.policy = defaultdict(lambda: np.random.rand())  # Actor policy
        self.value_function = defaultdict(lambda: 0.0)  # Critic function

    def choose_action(self, state):
        return "offload" if np.random.rand() < self.policy[str(state)] else "local"

    def update_policy(self, state, action, reward, next_state):
        td_target = reward + DISCOUNT_FACTOR * self.value_function[str(next_state)]
        self.value_function[str(state)] += LEARNING_RATE_CRITIC * (td_target - self.value_function[str(state)])
        self.policy[str(state)] += LEARNING_RATE_ACTOR * (reward - self.value_function[str(state)])

    def sync_policy(self, global_policy):
        for state in global_policy:
            self.policy[state] = np.mean([self.policy[state], global_policy[state]])

# Collaborative Multi-Agent Reinforcement Learning (CMARL)
class CMARL:
    def __init__(self, num_agents: int, dataset: Dataset):
        self.env = Environment(dataset)
        self.agents = [Agent(i) for i in range(num_agents)]

    def train(self):
        for episode in range(NUM_EPISODES):
            state = self.env.reset()
            done = False
            episode_rewards = []

            while not done:
                actions = [agent.choose_action(state) for agent in self.agents]
                next_state, reward, done = self.env.step(actions[0])
                episode_rewards.append(reward)

                for agent in self.agents:
                    agent.update_policy(state, actions[0], reward, next_state)

                state = next_state

            if episode % SYNC_INTERVAL == 0:
                global_policy = self.aggregate_policies()
                for agent in self.agents:
                    agent.sync_policy(global_policy)

            if episode % 50 == 0:
                print(f"Episode {episode}: Avg Reward = {np.mean(episode_rewards):.2f}")

    def aggregate_policies(self):
        aggregated_policy = defaultdict(lambda: 0.0)
        for agent in self.agents:
            for state, action_value in agent.policy.items():
                aggregated_policy[state] += action_value
        for state in aggregated_policy:
            aggregated_policy[state] /= len(self.agents)
        return aggregated_policy

# Main function
def main():
    dataset = Dataset(DATA_FILE_PATH)
    cmarl = CMARL(num_agents=NUM_AGENTS, dataset=dataset)
    cmarl.train()
    print("Training complete!")

if __name__ == "__main__":
    main()

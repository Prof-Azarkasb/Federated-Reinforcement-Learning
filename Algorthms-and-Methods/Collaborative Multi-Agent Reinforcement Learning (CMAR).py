import numpy as np
import random
import time
from collections import defaultdict
from typing import List, Dict

# Define parameters
NUM_AGENTS = 5
NUM_EPISODES = 500
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.1
SYNC_INTERVAL = 10  # Period for agents to synchronize their Q-tables

# Initialize metrics storage
metrics_data = {
    "Data Privacy Protection": [],
    "Convergence Rate": [],
    "Communication Overhead": [],
    "Computational Efficiency": [],
    "Decision Accuracy": [],
    "Adaptability": [],
    "Scalability": [],
    "Data Security": [],
    "Task Completion Time": [],
    "Resource Utilization": [],
    "Average Delay": [],
}

# Define a simple grid environment
class Environment:
    def __init__(self, size: int = 5):
        self.size = size
        self.reset()

    def reset(self):
        self.state = (0, 0)  # Start at the top-left corner
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:   # Move up
            y = max(y - 1, 0)
        elif action == 1: # Move down
            y = min(y + 1, self.size - 1)
        elif action == 2: # Move left
            x = max(x - 1, 0)
        elif action == 3: # Move right
            x = min(x + 1, self.size - 1)

        self.state = (x, y)
        reward = -1  # Negative reward to encourage collaboration
        done = (x == self.size - 1 and y == self.size - 1)  # Goal is bottom-right corner
        if done:
            reward = 10
        return self.state, reward, done

# Define Agent
class Agent:
    def __init__(self, env, agent_id: int):
        self.env = env
        self.agent_id = agent_id
        self.q_table = defaultdict(lambda: np.zeros(4))
        self.decision_accuracy = 0.0  # Accuracy of decisions (Metric 5)

    def choose_action(self, state):
        if random.uniform(0, 1) < EPSILON:
            return random.choice([0, 1, 2, 3])  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + DISCOUNT_FACTOR * self.q_table[next_state][best_next_action]
        self.q_table[state][action] += LEARNING_RATE * (td_target - self.q_table[state][action])

    def sync_q_table(self, other_q_table):
        for state, actions in other_q_table.items():
            self.q_table[state] = np.mean([self.q_table[state], actions], axis=0)

# Collaborative Multi-Agent Learning (CMAR) with metrics
class CMAR:
    def __init__(self, num_agents: int):
        self.env = Environment()
        self.agents = [Agent(self.env, i) for i in range(num_agents)]

    def train(self):
        for episode in range(NUM_EPISODES):
            start_time = time.time()
            episode_rewards = []
            state = self.env.reset()
            done = False
            accuracy_count = 0  # Correct decisions counter (Metric 5)

            while not done:
                for agent in self.agents:
                    action = agent.choose_action(state)
                    next_state, reward, done = self.env.step(action)
                    agent.update_q_value(state, action, reward, next_state)
                    episode_rewards.append(reward)
                    if action == np.argmax(agent.q_table[state]):
                        accuracy_count += 1  # Count accurate actions
                    state = next_state
                    if done:
                        break

            # Metrics Collection
            self.collect_metrics(episode, accuracy_count, episode_rewards, start_time)
            
            # Synchronize Q-tables periodically
            if episode % SYNC_INTERVAL == 0:
                average_q_table = self.aggregate_q_tables()
                for agent in self.agents:
                    agent.sync_q_table(average_q_table)

            if episode % 50 == 0:
                print(f"Episode {episode} - Metrics: {self.calculate_episode_metrics(episode)}")

    def aggregate_q_tables(self):
        aggregated_q_table = defaultdict(lambda: np.zeros(4))
        for agent in self.agents:
            for state, actions in agent.q_table.items():
                if state in aggregated_q_table:
                    aggregated_q_table[state] += actions
                else:
                    aggregated_q_table[state] = actions
        # Averaging across agents
        for state in aggregated_q_table:
            aggregated_q_table[state] /= len(self.agents)
        return aggregated_q_table

    def collect_metrics(self, episode, accuracy_count, episode_rewards, start_time):
        # Metric 2: Convergence Rate (Episodes per second)
        metrics_data["Convergence Rate"].append(1 / (time.time() - start_time))

        # Metric 3: Communication Overhead (Sync data size)
        metrics_data["Communication Overhead"].append(len(self.agents) * 4 * len(episode_rewards))

        # Metric 4: Computational Efficiency (GFLOPS estimate, simplified)
        metrics_data["Computational Efficiency"].append((len(episode_rewards) * 1e-6) / (time.time() - start_time))

        # Metric 5: Decision Accuracy (%)
        decision_accuracy = accuracy_count / len(episode_rewards) * 100
        metrics_data["Decision Accuracy"].append(decision_accuracy)

        # Metric 9: Task Completion Time
        metrics_data["Task Completion Time"].append(time.time() - start_time)

        # Metric 11: Average Delay (Simulated as random values to represent processing delays)
        metrics_data["Average Delay"].append(np.random.randint(1, 5))

        # Placeholder metrics (1, 6, 7, 8, 10)
        metrics_data["Data Privacy Protection"].append(np.random.randint(7, 10))
        metrics_data["Adaptability"].append(np.random.randint(5, 10))
        metrics_data["Scalability"].append(np.random.randint(6, 10))
        metrics_data["Data Security"].append(np.random.randint(8, 10))
        metrics_data["Resource Utilization"].append(np.random.randint(50, 100))

    def calculate_episode_metrics(self, episode):
        return {metric: round(np.mean(values), 2) for metric, values in metrics_data.items()}

# Main function
def main():
    cmar = CMAR(num_agents=NUM_AGENTS)
    cmar.train()
    print("Final metrics:", cmar.calculate_episode_metrics(NUM_EPISODES - 1))

if __name__ == "__main__":
    main()

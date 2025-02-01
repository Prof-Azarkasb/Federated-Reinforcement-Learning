import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random

# Load and preprocess the Google 2019 Cluster Sample Dataset
def load_and_preprocess_data():
    # Load the dataset
    url = "https://www.kaggle.com/datasets/derrickmwiti/google-2019-cluster-sample/download"
    dataset = pd.read_csv(url)  # Assuming the data is available in CSV format
    
    # Data Cleaning and Preprocessing
    dataset = dataset.dropna()  # Remove missing values
    dataset = dataset[dataset['failed'] == 0]  # Filter out failed tasks
    dataset = dataset[['resource_request', 'priority', 'cpu_usage_distribution', 'time', 'start_time', 'end_time']]  # Relevant features
    
    # Normalize or scale features if necessary
    dataset[['resource_request', 'cpu_usage_distribution']] = dataset[['resource_request', 'cpu_usage_distribution']].apply(lambda x: (x - np.mean(x)) / np.std(x))
    
    return dataset

# Define PPO components

class PPO:
    def __init__(self, state_space, action_space, lr=0.0003, gamma=0.99, clip_value=0.2):
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.clip_value = clip_value
        
        # Define PPO policy and value networks
        self.policy_model = self.build_model()
        self.value_model = self.build_value_model()
        
        # Optimizers
        self.optimizer_policy = Adam(learning_rate=lr)
        self.optimizer_value = Adam(learning_rate=lr)
    
    def build_model(self):
        model = Sequential([
            Dense(64, input_dim=self.state_space, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_space, activation='softmax')  # Output probabilities for each action
        ])
        return model

    def build_value_model(self):
        model = Sequential([
            Dense(64, input_dim=self.state_space, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1)  # Single value for state-value estimation
        ])
        return model

    def get_action(self, state):
        state = np.array(state).reshape(1, -1)
        action_probs = self.policy_model(state)
        action = np.random.choice(self.action_space, p=action_probs[0])
        return action

    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = rewards + self.gamma * (1 - dones) * next_values - values
        return advantages

    def train(self, states, actions, rewards, next_states, dones):
        # Compute values for current and next states
        values = self.value_model(states)
        next_values = self.value_model(next_states)
        
        advantages = self.compute_advantages(rewards, values, next_values, dones)

        with tf.GradientTape() as tape:
            # Compute the loss for the policy
            action_probs = self.policy_model(states)
            action_probs = tf.gather(action_probs, actions, axis=1)
            ratio = action_probs / (self.old_action_probs + 1e-10)  # Clip ratio to avoid large policy updates
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_value, 1 + self.clip_value)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

        with tf.GradientTape() as value_tape:
            # Compute the value function loss
            value_loss = tf.reduce_mean(tf.square(rewards - values))

        # Update policy network
        grads = tape.gradient(policy_loss, self.policy_model.trainable_variables)
        self.optimizer_policy.apply_gradients(zip(grads, self.policy_model.trainable_variables))

        # Update value network
        grads = value_tape.gradient(value_loss, self.value_model.trainable_variables)
        self.optimizer_value.apply_gradients(zip(grads, self.value_model.trainable_variables))

# FogNode class with PPO integration
class FogNode:
    def __init__(self, id, state_space, action_space, data=None):
        self.id = id
        self.model = PPO(state_space, action_space)  # PPO model for each fog node
        self.local_data = data  # Placeholder for node-specific training data

    def train_local_model(self, max_steps=100):
        for step in range(max_steps):
            # Simulate environment interactions
            state = np.random.rand(self.model.state_space)  # Replace with real data from dataset
            action = self.model.get_action(state)
            reward = np.random.rand()  # Simulated reward for the action
            next_state = np.random.rand(self.model.state_space)
            done = np.random.choice([0, 1])  # Whether the episode ends
            
            # Train the model (PPO)
            self.model.train(np.array([state]), np.array([action]), np.array([reward]), np.array([next_state]), np.array([done]))

            if step % 10 == 0:
                print(f"FogNode {self.id} training step {step} completed.")

    def send_local_model(self):
        return self.model.policy_model.get_weights()  # Sending policy model weights

    def receive_global_model(self, global_weights):
        self.model.policy_model.set_weights(global_weights)  # Update with the global model weights

# Central server to aggregate models from fog nodes
class CentralServer:
    def __init__(self):
        self.global_model = PPO(state_space=10, action_space=5)  # Initialize the global PPO model
        self.fog_nodes = []

    def add_fog_node(self, fog_node):
        self.fog_nodes.append(fog_node)

    def aggregate_models(self):
        all_weights = [fog_node.send_local_model() for fog_node in self.fog_nodes]
        avg_weights = np.mean(all_weights, axis=0)
        self.global_model.policy_model.set_weights(avg_weights)

    def distribute_global_model(self):
        for fog_node in self.fog_nodes:
            fog_node.receive_global_model(self.global_model.policy_model.get_weights())

# Federated Learning Loop
def federated_learning_loop(central_server, max_iterations, target_accuracy):
    iteration = 0
    global_accuracy = 0
    
    while iteration < max_iterations and global_accuracy < target_accuracy:
        print(f"Iteration {iteration + 1}/{max_iterations}")
        
        # Local Training at each fog node
        for fog_node in central_server.fog_nodes:
            fog_node.train_local_model()  # Each fog node trains its model
        
        # Aggregate local models at the central server
        central_server.aggregate_models()
        
        # Distribute the new global model back to fog nodes
        central_server.distribute_global_model()
        
        # Simulate evaluation of the global model accuracy
        global_accuracy = np.random.rand()  # Simplified for demonstration
        print(f"Global model accuracy: {global_accuracy:.4f}")
        
        iteration += 1
    
    if global_accuracy >= target_accuracy:
        print("Target accuracy reached, stopping training.")
    else:
        print("Max iterations reached without reaching target accuracy.")

# Initialize and Run the Federated Learning System
def main():
    # Load the dataset
    dataset = load_and_preprocess_data()
    
    # Split data into tasks for fog nodes (for simplicity, using random sampling)
    num_fog_nodes = 3  # You can adjust the number of fog nodes as needed
    fog_node_data = np.array_split(dataset, num_fog_nodes)
    
    # Initialize the central server
    central_server = CentralServer()
    
    # Create and add fog nodes to the server
    for i in range(num_fog_nodes):
        fog_node = FogNode(id=i, state_space=10, action_space=5, data=fog_node_data[i])
        central_server.add_fog_node(fog_node)
    
    # Start the federated learning process
    federated_learning_loop(central_server, max_iterations=100, target_accuracy=0.95)

if __name__ == "__main__":
    main()

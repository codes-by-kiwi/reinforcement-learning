import os
import random
import gymnasium as gym
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

# Print environment & library versions
print("NumPy version:", np.__version__)
print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

def BestQDNModel(input_shape, action_space, learning_rate):
    """
    Build the Deep Q-Network model
    """
    X_input = Input(input_shape)
    X = Dense(512, activation="relu", kernel_initializer='he_uniform')(X_input)
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs=X_input, outputs=X, name='CartPole_DQN_model')
    model.compile(loss="mse", optimizer=RMSprop(learning_rate=learning_rate, rho=0.95, epsilon=0.01))
    return model

class DQNAgentNaive:
    def __init__(self, config):
        """Initialize the DQN agent"""
        self.env = gym.make('CartPole-v1')
        self.state_size = int(self.env.observation_space.shape[0])
        self.action_size = int(self.env.action_space.n)
        self.EPISODES = config.get("EPISODES", 1000)
        self.gamma = config.get("gamma", 0.95)
        self.epsilon = config.get("epsilon", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.001)
        self.epsilon_decay = config.get("epsilon_decay", 0.99)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.update_ratio = config.get("update_ratio", 1) 
        
        # Initialize Q-Network
        self.model = BestQDNModel(input_shape=(self.state_size,), 
                                  action_space=self.action_size, 
                                  learning_rate=self.learning_rate)

    def act(self, state):
        """Select an action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_values = self.model.predict(state, verbose=0)
            return np.argmax(q_values[0])  # Exploit

    def update(self, state, action, reward, next_state, done):
        """Update Q-values using the Bellman equation"""
        target = self.model.predict(state, verbose=0)
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
        
        # Train model using single step update
        self.model.train_on_batch(state, target)
        
        # Decay epsilon
        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def run(self):
        """Train the DQN agent"""
        total_steps = 0
        scores = []
        steps_per_episode = []

        for e in range(self.EPISODES):
            state, _ = self.env.reset()
            state = np.reshape(state, (1, self.state_size))
            done = False
            score = 0
            step_count = 0  # Track episode steps

            while not done and step_count < self.env.spec.max_episode_steps:
                action = self.act(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_state = np.reshape(next_state, (1, self.state_size))

                if done:
                    reward = -100  # Penalize failure

                self.update(state, action, reward, next_state, done)

                state = next_state
                score += 1
                total_steps += 1
                step_count += 1  # Ensure the episode does not exceed max steps

                if total_steps >= 100000:
                    print(f"Total steps reached {total_steps} at episode {e} with score {score}")
                    scores.append(score)
                    steps_per_episode.append(total_steps)
                    self.model.save("cartpole-dqn-naive-w500.keras")
                    print("Model saved as 'cartpole-dqn-naive-w500.keras'")
                    return steps_per_episode, scores

            scores.append(score)
            steps_per_episode.append(total_steps)
            print(f"Episode {e}/{self.EPISODES}, score: {score}, total_steps: {total_steps}, epsilon: {self.epsilon:.3f}")

        self.model.save("cartpole-dqn-naive-w500.keras")
        print("Model saved as 'cartpole-dqn-naive-w500.keras'")
        return steps_per_episode, scores

def plot_learning_curve(steps_per_episode, scores):
    """
    Plot learning curve: Return over Environment Steps
    """
    plt.figure(figsize=(8,6))
    plt.plot(steps_per_episode, scores, marker='.')
    plt.xlabel("Environment Steps")
    plt.ylabel("Episode Score (Return)")
    plt.title("Learning Curve: Return vs. Environment Steps")
    plt.grid()
    plt.savefig('learning_curve_naive_w500.png')
    plt.show()
    

if __name__ == "__main__":
    config = {
        "EPISODES": 1000,
        "gamma": 0.95,
        "epsilon": 1.0,
        "epsilon_min": 0.001,
        "epsilon_decay": 0.99,
        "learning_rate": 0.001,
        "update_ratio": 1.0,
    }
    agent = DQNAgentNaive(config)
    
    # Train the model and store results
    steps_per_episode, scores = agent.run()

    # Plot learning curve (training)
    if scores:
        plot_learning_curve(steps_per_episode, scores)


import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

print(torch.__version__)
print(gym.__version__)
print(np.__version__)

import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=512):
        super(ActorCritic, self).__init__()
        # Shared layer
        self.fc = nn.Linear(state_dim, hidden_size)

        # Policy (actor) head
        self.policy = nn.Linear(hidden_size, action_dim)

        # Value (critic) head
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.tanh(self.fc(x))
        policy_logits = self.policy(x)
        value = self.value(x)
        return policy_logits, value

def compute_returns_and_advantages(rewards, values, dones, gamma, lam):
    advantages = []
    gae = 0
    values = values + [0]  # Add a dummy for bootstrap

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns

def ppo_minibatch_update(model, optimizer, states, actions, log_probs_old, returns, advantages, clip_eps, epochs, mini_batch_size=256):
    dataset_size = len(states)
    indices = np.arange(dataset_size)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)
    log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, mini_batch_size):
            end = start + mini_batch_size
            minib_idx = indices[start:end]
            minib_states = states[minib_idx]
            minib_actions = actions[minib_idx]
            minib_old_log_probs = log_probs_old[minib_idx]
            minib_returns = returns[minib_idx]
            minib_advantages = advantages[minib_idx]

            logits, values = model(minib_states)
            dist = Categorical(logits=logits)
            log_probs_new = dist.log_prob(minib_actions)
            ratio = torch.exp(log_probs_new - minib_old_log_probs)

            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * minib_advantages
            actor_loss = -torch.min(ratio * minib_advantages, clipped).mean()

    # Critic loss (MSE)
            critic_loss = nn.MSELoss()(values.squeeze(), minib_returns)

    # Total loss
    # loss = actor_loss + 0.5 * critic_loss
            entropy = dist.entropy().mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Train function returning rewards
def train_ppo(model, optimizer, env, max_env_steps, batch_size, gamma, lambda_, clip_eps,
              epochs, mini_batch_size, run_id, csv_path):

    reward_per_episode = []
    env_steps = []
    step_counter = 0
    state, _ = env.reset()

    while step_counter < max_env_steps:
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        batch_steps = 0

        while batch_steps < batch_size:
            episode_reward = 0
            done=False

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                logits, value = model(state_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample()

                next_state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated

                states.append(state)
                actions.append(action.item())
                rewards.append(reward)
                dones.append(done)
                log_probs.append(dist.log_prob(action).item())
                values.append(value.item())

                state = next_state
                step_counter +=1
                batch_steps +=1
                episode_reward += reward

                if done or step_counter >= max_env_steps:
                    reward_per_episode.append(episode_reward)
                    env_steps.append(step_counter)
                    state, _ = env.reset()
                    break

            if step_counter >= max_env_steps:
                break

        advantages, returns = compute_returns_and_advantages(rewards, values, dones, gamma, lambda_)
        ppo_minibatch_update(model, optimizer, states, actions, log_probs, returns, advantages, clip_eps, epochs, mini_batch_size)

        print(f"Step {step_counter}/{max_env_steps} | Avg Reward (last 10): {np.mean(reward_per_episode[-10:]):.2f}")

    env.close()

    return reward_per_episode, env_steps

import csv
import os

config = {
    'learning_rate': 1e-3,
    'batch_size': 3000,
    'gamma': 0.99,
    'lambda': 0.95,
    'clip_eps': 0.4
}

csv_path = "ppo_sgd_results.csv"
if not os.path.exists(csv_path):
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Run", "EnvStep", "TotalSteps", "EpisodeReturn"])

MAX_ENV_STEPS = 1e6
REPEATS = 5
results = []

for rep in range(REPEATS):
    print(f"\n=== Run {rep+1}/{REPEATS} ===")
    seed = 42 + rep
    set_seed(seed)

    env = gym.make("CartPole-v1")
    env.reset(seed=seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    episode_return, env_step = train_ppo(
        model=model,
        optimizer=optimizer,
        env=env,
        max_env_steps=MAX_ENV_STEPS,
        batch_size=config['batch_size'],
        gamma=config['gamma'],
        lambda_=config['lambda'],
        clip_eps=config['clip_eps'],
        epochs=4,
        mini_batch_size=256,
        run_id=rep+1,
        csv_path=csv_path
    )

    results.append((episode_return, env_step))

import matplotlib.pyplot as plt
window_size = 50
plt.figure(figsize=(12, 6))

for i, (episode_return, env_step) in enumerate(results):
    episode_return_smooth = np.convolve(episode_return, np.ones(window_size)/window_size, mode='valid')
    env_step_aligned = env_step[window_size - 1:]
    plt.plot(env_step_aligned, episode_return_smooth, label=f'Run {i+1}', linewidth=1)

plt.xlabel('Environment Steps')
plt.ylabel('Smoothed Episode Reward (50)')
plt.title('PPO Performance with SGD and clipping over 5 Runs (~1000k steps each)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


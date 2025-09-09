import pandas as pd

df_ppo = pd.read_csv("ppo_results.csv")
df_sgd = pd.read_csv("ppo_sgd_results.csv")

import matplotlib.pyplot as plt

pivot_df = df_ppo.pivot_table(index="EnvStep", columns="Run", values="EpisodeReturn")

pivot_df["MeanReturn"] = pivot_df.mean(axis=1)

window_size = 50
smoothed = pivot_df["MeanReturn"].rolling(window=window_size, min_periods=1).mean()

plt.figure(figsize=(12, 6))
plt.plot(pivot_df.index, smoothed, label="PPO Mean Return", color="blue", linewidth=2)
plt.xlabel("Environment Steps")
plt.ylabel(f"Smoothed Episode Reward ({window_size})")
plt.title("PPO Average Performance Over 5 Runs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

ppo_pivot = df_ppo.pivot_table(index="EnvStep", columns="Run", values="EpisodeReturn")
sgd_pivot = df_sgd.pivot_table(index="EnvStep", columns="Run", values="EpisodeReturn")

ppo_pivot["MeanReturn"] = ppo_pivot.mean(axis=1)
sgd_pivot["MeanReturn"] = sgd_pivot.mean(axis=1)

window_size = 50
ppo_smoothed = ppo_pivot["MeanReturn"].rolling(window=window_size, min_periods=1).mean()
sgd_smoothed = sgd_pivot["MeanReturn"].rolling(window=window_size, min_periods=1).mean()

plt.figure(figsize=(12, 6))
plt.plot(ppo_pivot.index, ppo_smoothed, label="PPO", color="blue", linewidth=2)
plt.plot(sgd_pivot.index, sgd_smoothed, label="PPO + SGD", color="red", linewidth=2)

plt.xlabel("Environment Steps")
plt.ylabel(f"Smoothed Episode Reward ({window_size})")
plt.title("Comparison of PPO vs PPO + SGD Performance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

df = pd.read_csv("ac_a2c_results.csv")
df_ac = df[df["Method"] == "AC"].copy()
df_a2c = df[df["Method"] == "A2C"].copy()

ac_pivot = df_ac.pivot_table(index="EnvStep", columns="Run", values="EpisodeReturn")
ppo_pivot = df_ppo.pivot_table(index="EnvStep", columns="Run", values="EpisodeReturn")

ac_pivot["MeanReturn"] = ac_pivot.mean(axis=1)
ppo_pivot["MeanReturn"] = ppo_pivot.mean(axis=1)

window_size = 50
ac_smoothed = ac_pivot["MeanReturn"].rolling(window=window_size, min_periods=1).mean()
ppo_smoothed = ppo_pivot["MeanReturn"].rolling(window=window_size, min_periods=1).mean()

plt.figure(figsize=(12, 6))
plt.plot(ac_pivot.index, ac_smoothed, label="AC", color="red", linewidth=2)
plt.plot(ppo_pivot.index, ppo_smoothed, label="PPO", color="blue", linewidth=2)

plt.xlabel("Environment Steps")
plt.ylabel(f"Smoothed Episode Reward ({window_size})")
plt.title("Comparison of AC vs PPO Performance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

a2c_pivot = df_a2c.pivot_table(index="EnvStep", columns="Run", values="EpisodeReturn")
ppo_pivot = df_ppo.pivot_table(index="EnvStep", columns="Run", values="EpisodeReturn")

a2c_pivot["MeanReturn"] = a2c_pivot.mean(axis=1)
ppo_pivot["MeanReturn"] = ppo_pivot.mean(axis=1)

window_size = 50
a2c_smoothed = a2c_pivot["MeanReturn"].rolling(window=window_size, min_periods=1).mean()
ppo_smoothed = ppo_pivot["MeanReturn"].rolling(window=window_size, min_periods=1).mean()

plt.figure(figsize=(12, 6))
plt.plot(a2c_pivot.index, a2c_smoothed, label="A2C", color="red", linewidth=2)
plt.plot(ppo_pivot.index, ppo_smoothed, label="PPO", color="blue", linewidth=2)

plt.xlabel("Environment Steps")
plt.ylabel(f"Smoothed Episode Reward ({window_size})")
plt.title("Comparison of A2C vs PPO Performance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


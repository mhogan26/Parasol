import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

STEPS_PER_EPISODE = 20

df = pd.read_csv("training_rewards_PPO.txt", sep="\t")

# Recover episode and step-in-episode from the logged float
df["episode"] = np.floor(df["Episode.Step"] + 1e-9).astype(int)
df["counter"] = np.round((df["Episode.Step"] - df["episode"]) * STEPS_PER_EPISODE).astype(int)

# Keep only rewarded steps (round_counter = counter + 1 is even)
rewarded = df[(df["counter"] + 1) % 2 == 0]

# Mean reward per episode
per_ep = rewarded.groupby("episode")["Reward"].mean()

ep = per_ep.index.to_numpy()
m = per_ep.to_numpy()

plt.figure(figsize=(10, 6))
plt.plot(ep, m, marker="o", label="Mean reward per episode")
plt.xlabel("Episode")
plt.ylabel("Mean reward")
plt.title("Mean Reward per Episode (PPO)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reward_per_episode_PPO_11_19.pdf", dpi=150)
plt.show()
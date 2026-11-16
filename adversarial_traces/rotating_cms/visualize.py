import pandas as pd
import matplotlib.pyplot as plt

# Read the tab-separated file
df = pd.read_csv("training_rewards_PPO.txt", sep="\t")

# Plot Reward vs Step (colored by episode)
plt.figure(figsize=(10, 6))
plt.plot(df["Episode.Step"], df["Reward"], marker="o", linestyle="-")
plt.xlabel("Episode.Step")
plt.ylabel("Reward")
plt.title("Reward over Steps")
plt.grid(True)
plt.tight_layout()
plt.savefig("reward_plot.pdf", dpi=150)
plt.show()
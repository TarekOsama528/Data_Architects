import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare the data
df = pd.read_csv('Datasets/Dataset_3.csv')
df.columns = [
    "Timestamp", "Signal_Strength", "SNR", "Call_Duration", "Environment",
    "Attenuation", "Distance_to_Tower", "Call_Type", "Incoming_Outgoing"
]
df.dropna(subset=["SNR", "Signal_Strength", "Call_Duration", "Call_Type"], inplace=True)
df = df.sort_values(by="Call_Duration")

# --- Plot 1: Continuous line plot for Call Duration vs SNR and Signal Strength ---
fig, ax1 = plt.subplots(figsize=(12, 6))

sns.lineplot(data=df, x="Call_Duration", y="SNR", ax=ax1, label="SNR", color="tab:blue")
ax1.set_ylabel("SNR (dB)", color="tab:blue")
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
sns.lineplot(data=df, x="Call_Duration", y="Signal_Strength", ax=ax2, label="Signal Strength", color="tab:red")
ax2.set_ylabel("Signal Strength (dBm)", color="tab:red")
ax2.tick_params(axis='y', labelcolor='tab:red')

ax1.set_xlabel("Call Duration (s)")
plt.title("Call Duration vs SNR and Signal Strength")
fig.tight_layout()
plt.show()

# --- Plot 2: Call Type vs Mean SNR and Signal Strength ---
agg_df = df.groupby("Call_Type")[["SNR", "Signal_Strength"]].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=agg_df, x="Call_Type", y="SNR", label="Avg SNR", marker='o', color="tab:blue")
sns.lineplot(data=agg_df, x="Call_Type", y="Signal_Strength", label="Avg Signal Strength", marker='o', color="tab:red")

plt.title("Average SNR and Signal Strength by Call Type")
plt.xlabel("Call Type")
plt.ylabel("Value")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

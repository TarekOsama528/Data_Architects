import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, pearsonr

print("Script started!")

# Load the CSV file (assuming it's comma-separated)
try:
    df = pd.read_csv("train.csv")
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"First 3 rows:\n{df.head(3)}")
except FileNotFoundError:
    print("Error: The file 'train.csv' was not found.")
    exit()
except Exception as e:
    print("Error reading the CSV:", e)
    exit()

# Convert 'Timestamp' to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')

# Drop rows where Timestamp is invalid
df.dropna(subset=['Timestamp'], inplace=True)

# Extract hour and assign Time of Day
df['Hour'] = df['Timestamp'].dt.hour

def get_time_of_day(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 23:
        return 'Evening'
    else:
        return 'Late Night'

df['Time of Day'] = df['Hour'].apply(get_time_of_day)

# === Summary Table ===
summary = df.groupby(['Time of Day', 'Environment'])[['Signal Strength (dBm)', 'SNR']].agg(['mean', 'std']).reset_index()
print("\n=== Summary Statistics ===")
print(summary)

# === Visualizations ===
sns.set(style="whitegrid")
plt.ioff()  # Turn off interactive mode so plots wait to be closed manually

# Signal Strength Plot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Time of Day', y='Signal Strength (dBm)', hue='Environment')
plt.title('Signal Strength by Time of Day and Environment')
plt.tight_layout()
plt.show()

# SNR Plot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Time of Day', y='SNR', hue='Environment')
plt.title('SNR by Time of Day and Environment')
plt.tight_layout()
plt.show()

# === ANOVA Test ===
print("\n=== ANOVA Test Results ===")
for period in df['Time of Day'].unique():
    subset = df[df['Time of Day'] == period]
    print(f"\n-- {period} --")
    for metric in ['Signal Strength (dBm)', 'SNR']:
        groups = [group[metric].dropna() for name, group in subset.groupby('Environment')]
        if len(groups) > 1:
            f_stat, p_val = f_oneway(*groups)
            print(f"{metric}: F = {f_stat:.2f}, p = {p_val:.4f}")
        else:
            print(f"{metric}: Not enough groups for ANOVA.")

# === Correlation with Hour ===
print("\n=== Correlation with Hour ===")
for metric in ['Signal Strength (dBm)', 'SNR']:
    corr, p_val = pearsonr(df['Hour'], df[metric])
    print(f"{metric}: Correlation = {corr:.2f}, p = {p_val:.4f}")

# Keep the window open if running as a script
input("\nPress Enter to exit...")




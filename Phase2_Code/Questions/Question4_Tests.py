import pandas as pd
from scipy.stats import pearsonr, ttest_ind, f_oneway
from datetime import datetime

# Load the dataset
df = pd.read_csv("Datasets/Dataset_3.csv")

# Ensure proper datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Extract time and date components
df['Hour'] = df['Timestamp'].dt.hour
df['Date'] = df['Timestamp'].dt.date

# Drop any rows with missing SNR or Signal Strength
df = df.dropna(subset=['SNR', 'Signal Strength'])

# 1. Pearson Correlation: SNR vs. Signal Strength
corr, p_corr = pearsonr(df['SNR'], df['Signal Strength'])
print(f"Pearson Correlation between SNR and Signal Strength: {corr:.2f}, p-value: {p_corr:.4f}")

# 2. T-Test between Peak (18-22) and Off-Peak (0-6)
peak = df[(df['Hour'] >= 18) & (df['Hour'] <= 22)]
off_peak = df[(df['Hour'] >= 0) & (df['Hour'] <= 6)]

# T-test for SNR
t_snr, p_snr = ttest_ind(peak['SNR'], off_peak['SNR'], equal_var=False)
print(f"T-test for SNR (Peak vs. Off-Peak): t = {t_snr:.2f}, p = {p_snr:.4f}")

# T-test for Signal Strength
t_sig, p_sig = ttest_ind(peak['Signal Strength'], off_peak['Signal Strength'], equal_var=False)
print(f"T-test for Signal Strength (Peak vs. Off-Peak): t = {t_sig:.2f}, p = {p_sig:.4f}")

# 3. ANOVA for time intervals (Morning, Afternoon, Evening, Night)
def get_time_period(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 24:
        return 'Evening'
    else:
        return 'Night'

df['Time Period'] = df['Hour'].apply(get_time_period)

# Group by time period
groups_snr = [group['SNR'].values for name, group in df.groupby('Time Period')]
groups_sig = [group['Signal Strength'].values for name, group in df.groupby('Time Period')]

# ANOVA tests
f_snr, p_anova_snr = f_oneway(*groups_snr)
f_sig, p_anova_sig = f_oneway(*groups_sig)

print(f"ANOVA for SNR by Time Period: F = {f_snr:.2f}, p = {p_anova_snr:.4f}")
print(f"ANOVA for Signal Strength by Time Period: F = {f_sig:.2f}, p = {p_anova_sig:.4f}")

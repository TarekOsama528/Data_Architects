import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, shapiro
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load dataset
df = pd.read_csv('Datasets/Dataset_3.csv', parse_dates=['Timestamp'])

# Check normality
def test_normality(series):
    stat, p = shapiro(series.dropna())
    return 'normal' if p > 0.05 else 'non-normal'

# 1. CORRELATION TEST
# Determine correlation type
snr_norm = test_normality(df['SNR'])
sig_norm = test_normality(df['Signal Strength (dBm)'])

if snr_norm == 'normal':
    snr_corr, snr_p = pearsonr(df['Distance to Tower (km)'], df['SNR'])
else:
    snr_corr, snr_p = spearmanr(df['Distance to Tower (km)'], df['SNR'])

if sig_norm == 'normal':
    sig_corr, sig_p = pearsonr(df['Distance to Tower (km)'], df['Signal Strength (dBm)'])
else:
    sig_corr, sig_p = spearmanr(df['Distance to Tower (km)'], df['Signal Strength (dBm)'])

print(f"SNR Correlation: {snr_corr:.3f}, p = {snr_p:.3f}")
print(f"Signal Strength Correlation: {sig_corr:.3f}, p = {sig_p:.3f}")



# 2. TWO-WAY ANOVA
# Bin Distance to Tower
df['DistanceBin'] = pd.cut(df['Distance to Tower (km)'],
                           bins=[-np.inf, 1, 3, np.inf],
                           labels=['Near', 'Medium', 'Far'])

# Prepare categorical columns
df['Call Type'] = df['Call Type'].astype('category')
df['DistanceBin'] = df['DistanceBin'].astype('category')

# Corrected ANOVA for SNR
model_snr = ols('SNR ~ C(DistanceBin) * C(Q("Call Type"))', data=df).fit()
anova_snr = sm.stats.anova_lm(model_snr, typ=2)

# Corrected ANOVA for Signal Strength
model_sig = ols('Q("Signal Strength (dBm)") ~ C(DistanceBin) * C(Q("Call Type"))', data=df).fit()
anova_sig = sm.stats.anova_lm(model_sig, typ=2)


print("\nANOVA - SNR")
print(anova_snr)

print("\nANOVA - Signal Strength")
print(anova_sig)

# Optional: Visualization
plt.figure(figsize=(9, 6)) 
sns.boxplot(data=df, x='DistanceBin', y='SNR', hue='Call Type')
plt.title('SNR by Distance and Call Type')
plt.show()

plt.figure(figsize=(9, 6)) 
sns.boxplot(data=df, x='DistanceBin', y='Signal Strength (dBm)', hue='Call Type')
plt.title('Signal Strength by Distance and Call Type')
plt.show()

# Optional: Visualization without binning distances
# Continuous plot for SNR vs Distance to Tower
plt.figure(figsize=(12, 6))  # Wider figure size
sns.lineplot(data=df, x='Distance to Tower (km)', y='SNR', hue='Call Type', marker='o', palette='Set2')
plt.title('SNR vs Distance to Tower')
plt.xlabel('Distance to Tower (km)')
plt.ylabel('SNR')
plt.grid(True)
plt.show()

# Continuous plot for Signal Strength vs Distance to Tower
plt.figure(figsize=(12, 6))  # Wider figure size
sns.lineplot(data=df, x='Distance to Tower (km)', y='Signal Strength (dBm)', hue='Call Type', marker='o', palette='Set2')
plt.title('Signal Strength vs Distance to Tower')
plt.xlabel('Distance to Tower (km)')
plt.ylabel('Signal Strength (dBm)')
plt.grid(True)
plt.show()


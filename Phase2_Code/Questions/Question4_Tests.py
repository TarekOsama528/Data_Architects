import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, ttest_ind, f_oneway
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load dataset
df = pd.read_csv("../Datasets/Dataset_3.csv")
df.columns = df.columns.str.strip()

# Parse timestamp and extract hour
df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')
df['Hour'] = df['Timestamp'].dt.hour

# Define TimeGroup (Peak: 17–22, Off-Peak: others)
df['TimeGroup'] = df['Hour'].apply(lambda x: 'Peak' if 17 <= x <= 22 else 'Off-Peak')

# Drop NA rows for relevant analysis
df_clean = df.dropna(subset=['Signal Strength (dBm)', 'SNR', 'Hour'])

# ---------------------------------------------
# 1. Correlation Test + Scatter Plot
# ---------------------------------------------
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_clean, x='Signal Strength (dBm)', y='SNR', alpha=0.6)
sns.regplot(data=df_clean, x='Signal Strength (dBm)', y='SNR', scatter=False, color='red')
plt.title('Correlation between Signal Strength and SNR')
plt.tight_layout()
plt.show()

# Correlation Test
if abs(df_clean['Signal Strength (dBm)'].skew()) < 1 and abs(df_clean['SNR'].skew()) < 1:
    corr_type = "Pearson"
    corr_coef, p_val = pearsonr(df_clean['Signal Strength (dBm)'], df_clean['SNR'])
else:
    corr_type = "Spearman"
    corr_coef, p_val = spearmanr(df_clean['Signal Strength (dBm)'], df_clean['SNR'])

print(f"\n[Correlation Test] ({corr_type})")
print(f"Correlation Coefficient: {corr_coef:.3f}, p-value: {p_val:.3f}")

# ---------------------------------------------
# 2. Temporal Variation (T-test) + Boxplots
# ---------------------------------------------
# T-test: Peak vs Off-Peak
peak = df_clean[df_clean['TimeGroup'] == 'Peak']
offpeak = df_clean[df_clean['TimeGroup'] == 'Off-Peak']

t_signal, p_signal = ttest_ind(peak['Signal Strength (dBm)'], offpeak['Signal Strength (dBm)'], equal_var=False)
t_snr, p_snr = ttest_ind(peak['SNR'], offpeak['SNR'], equal_var=False)

print("\n[T-Test: Peak vs Off-Peak]")
print(f"Signal Strength: t = {t_signal:.3f}, p = {p_signal:.3f}")
print(f"SNR: t = {t_snr:.3f}, p = {p_snr:.3f}")

# Boxplots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(data=df_clean, x='TimeGroup', y='Signal Strength (dBm)', palette="pastel")
plt.title('Signal Strength by Time Group')

plt.subplot(1, 2, 2)
sns.boxplot(data=df_clean, x='TimeGroup', y='SNR', palette="pastel")
plt.title('SNR by Time Group')

plt.tight_layout()
plt.show()

# ---------------------------------------------
# Optional ANOVA on time of day (Morning/Afternoon/Evening)
# ---------------------------------------------
df_clean['TimeOfDay'] = pd.cut(df_clean['Hour'], bins=[-1, 11, 17, 23], labels=["Morning", "Afternoon", "Evening"])

anova_groups = [g['Signal Strength (dBm)'].dropna() for _, g in df_clean.groupby('TimeOfDay')]
f_stat, anova_p = f_oneway(*anova_groups)

print("\n[ANOVA: Signal Strength by Time of Day]")
print(f"F-statistic = {f_stat:.3f}, p-value = {anova_p:.3f}")

# ---------------------------------------------
# 3. Regression Analysis + Regression Line
# ---------------------------------------------
model = smf.ols("SNR ~ Q('Signal Strength (dBm)') + Hour", data=df_clean).fit()

print("\n[Multiple Linear Regression: SNR ~ Signal Strength + Hour]")
print(model.summary())

# Plot: Regression Line (SNR vs Signal Strength)
plt.figure(figsize=(8, 6))
sns.regplot(data=df_clean, x='Signal Strength (dBm)', y='SNR', line_kws={'color': 'red'})
plt.title('Regression: SNR vs Signal Strength')
plt.tight_layout()
plt.show()

# Optional: Plot residuals
plt.figure(figsize=(8, 6))
sns.residplot(data=df_clean, x='Signal Strength (dBm)', y='SNR', lowess=True)
plt.title("Regression Residuals")
plt.tight_layout()
plt.show()

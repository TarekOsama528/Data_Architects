import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load datasets
dataset2 = pd.read_csv("../Datasets/Dataset_2.csv")  # Fiber
dataset3 = pd.read_csv("../Datasets/Dataset_3.csv")  # Wireless

# ---------- Clean and Prepare Data ----------

# Fiber: drop rows with missing temp/humidity
dataset2_clean = dataset2.dropna(subset=['Temperature', 'Humidity'])

# Wireless: categorize distance
def categorize_distance(dist):
    if dist < 2:
        return 'Short'
    elif dist < 7:
        return 'Medium'
    else:
        return 'Long'

dataset3['Distance Category'] = dataset3['Distance to Tower (km)'].apply(categorize_distance)

# ---------- Statistical Models ----------

# Fiber: multiple linear regression
model_fiber = smf.ols('Q("SNR Receiver") ~ Q("Transmission Distance") + Temperature + Humidity', data=dataset2_clean).fit()

# Wireless: two-way ANOVA
model_wireless = smf.ols('SNR ~ C(Q("Distance Category")) + C(Environment) + C(Q("Distance Category")):C(Environment)', data=dataset3).fit()
anova_wireless = sm.stats.anova_lm(model_wireless, typ=2)

# ---------- Results Table for Fiber Model ----------
coeffs = model_fiber.params
conf_int = model_fiber.conf_int()
p_values = model_fiber.pvalues

regression_table = pd.DataFrame({
    'Coefficient': coeffs,
    'P-Value': p_values,
    'CI Lower': conf_int[0],
    'CI Upper': conf_int[1]
})

print("\n--- Regression Results (Fiber Dataset) ---\n")
print(regression_table)

print("\n--- ANOVA Results (Wireless Dataset) ---\n")
print(anova_wireless)

# ---------- Visualization 1: Wireless Boxplot ----------
plt.figure(figsize=(12, 6))
sns.boxplot(data=dataset3, x='Environment', y='SNR', hue='Distance Category')
plt.title('SNR by Environment and Distance Category (Wireless)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("snr_boxplot_wireless.png")
plt.close()

# ---------- Visualization 2: Fiber Regression Plot ----------
plt.figure(figsize=(12, 6))
sns.scatterplot(data=dataset2_clean, x="Transmission Distance", y="SNR Receiver", label='Data Points')
sns.regplot(data=dataset2_clean, x="Transmission Distance", y="SNR Receiver", scatter=False, color='red', label='Regression Line')
plt.title('SNR Receiver vs. Transmission Distance (Fiber)')
plt.legend()
plt.tight_layout()
plt.savefig("snr_regression_fiber.png")
plt.close()

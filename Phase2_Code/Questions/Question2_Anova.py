import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load Dataset 3 (update path accordingly)
df = pd.read_csv('Datasets/Dataset_3.csv')  # Or use pd.read_csv()

# Select relevant columns by index
# Column 2: SNR, Column 3: Call Duration, Column 8: Call Type
# Let's assume Signal Strength is in Column 4 (index 3) – update if different
df = df.iloc[:, [2, 3, 1, 8]]  # Adjust if Signal Strength is elsewhere
df.columns = ['SNR', 'Call_Duration', 'Signal_Strength', 'Call_Type']

# Clean data
df.dropna(subset=['SNR', 'Call_Duration', 'Signal_Strength', 'Call_Type'], inplace=True)

# ----------------------------
# ANOVA for SNR (Call Type + Call Duration)
# ----------------------------
model_snr = smf.ols('SNR ~ Call_Duration + C(Call_Type)', data=df).fit()
anova_snr = sm.stats.anova_lm(model_snr, typ=2)

print("ANOVA Results: Impact on SNR")
print(anova_snr)

# ----------------------------
# ANOVA for Signal Strength (Call Type + Call Duration)
# ----------------------------
model_signal = smf.ols('Signal_Strength ~ Call_Duration + C(Call_Type)', data=df).fit()
anova_signal = sm.stats.anova_lm(model_signal, typ=2)

print("\nANOVA Results: Impact on Signal Strength")
print(anova_signal)

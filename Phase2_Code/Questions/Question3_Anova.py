import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt

# Read data from dataset_1.csv
df = pd.read_csv('Datasets/dataset_1.csv')

# Debug: Print the columns in the dataset
print("Columns in the dataset:", df.columns.tolist())

# Rename columns to remove special characters
df.rename(columns={'Flow.Duration': 'FlowDuration', 'Fwd.IAT.Total': 'FwdIATTotal'}, inplace=True)

# Ensure the dataset has the required columns
required_columns = {'ProtocolName', 'FwdIATTotal', 'FlowDuration'}
if not required_columns.issubset(df.columns):
    missing_columns = required_columns - set(df.columns)
    raise ValueError(f"The dataset is missing the following required columns: {missing_columns}")

# Perform Two-Way ANOVA with FwdIATTotal as a continuous variable
model = ols('FlowDuration ~ C(ProtocolName) + FwdIATTotal + C(ProtocolName):FwdIATTotal', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print("=== Two-Way ANOVA Results ===")
print(anova_table)

# Optional: Plot for visual inspection
plt.figure(figsize=(10000, 6))
sns.boxplot(data=df, x='ProtocolName', y='FlowDuration', hue='ProtocolName')
plt.title('Flow Duration by Protocol')
plt.show()
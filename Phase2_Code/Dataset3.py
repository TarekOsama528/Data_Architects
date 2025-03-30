import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("C:/Users/osama/OneDrive/Desktop/Data_Architects/Datasets/train.csv")

# Generate descriptive statistics
desc_stats = df.describe().round(2)

# Save table as an image
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=desc_stats.values, 
                 colLabels=desc_stats.columns, 
                 rowLabels=desc_stats.index, 
                 cellLoc='center', 
                 loc='center',
                 colColours=['lightgray'] * desc_stats.shape[1])  

table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width([i for i in range(len(desc_stats.columns))])  

plt.title("Descriptive Statistics", fontsize=14, fontweight="bold", pad=20) 
plt.show()


df_numeric = df.select_dtypes(include=[np.number])

# Correlation Heatmap (Numeric Only)
plt.figure(figsize=(10, 6))
correlation_matrix = df_numeric.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Plot distributions for each numerical column
for col in df_numeric.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col].dropna(), bins=50, color='skyblue')
    plt.xlabel(f"{col}")
    plt.ylabel("Count")
    plt.title(f"Distribution of {col}")
    plt.grid(True)
    plt.show()

# Identify categorical columns
categorical_columns = [col for col in df.select_dtypes(include=['object']).columns
                       if 'time' not in col.lower() and 'date' not in col.lower()]

# Plot bar charts for each categorical column
for col in categorical_columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(y=df[col], palette='Set2', order=df[col].value_counts().index)
    plt.xlabel("Count")
    plt.ylabel(col)
    plt.title(f"Distribution of {col}")
    plt.show()



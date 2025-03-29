import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("C:/Users/osama/OneDrive/Desktop/Data_Architects/Datasets/Dataset-Unicauca-Version2-87Atts.csv")


# Descriptive statistics for numerical columns
print("Descriptive Statistics:\n")
print(df.describe())

# Plotting correlation heatmap
plt.figure(figsize=(10, 6))
#correlation_matrix = df.corr()
correlation_matrix = df.select_dtypes(include=['number']).corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Distribution plots for numerical features
numerical_columns = ['Flow.Duration', 'Total.Length.of.Fwd.Packets', 'Total.Length.of.Bwd.Packets', 'Fwd.IAT.Total', 'Bwd.IAT.Total']

plt.figure(figsize=(14, 10))
for col in numerical_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric
    df[col] = df[col].replace([np.inf, -np.inf], np.nan).dropna()  # Remove NaNs/Infs

    if df[col].nunique() > 1:  # Ensure at least two unique values
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=False, color='skyblue', bins=50)  # Disable KDE
        plt.title(f'Distribution of {col}')
        plt.show()
    else:
        print(f"Skipping {col} - only one unique value.")

# Bar plot for ProtocolName (categorical data)
plt.figure(figsize=(10, 6))
sns.countplot(y='ProtocolName', data=df, palette='Set2')
plt.title('Protocol Name Count Distribution')
plt.xlabel('Count')
plt.ylabel('Protocol Name')
plt.show()

# Bar plot for L7Protocol (potential dependent variable)
plt.figure(figsize=(10, 6))
sns.countplot(y='L7Protocol', data=df, palette='Set1')
plt.title('L7Protocol Count Distribution')
plt.xlabel('Count')
plt.ylabel('L7 Protocol')
plt.show()

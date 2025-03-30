import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("C:/Users/osama/OneDrive/Desktop/Data_Architects/Datasets/Dataset-Unicauca-Version2-87Atts.csv")


# Drop "L7Protocol" column if it exists
if "L7Protocol" in df.columns:
    df = df.drop(columns=["L7Protocol"])

# Get descriptive statistics (excluding "count" row)
desc_stats = df.describe().drop(index="count", errors="ignore")

# Convert values to 2 decimal places (removes scientific notation)
desc_stats = desc_stats.applymap("{:.2f}".format)

# Convert DataFrame to a string format for visualization
fig, ax = plt.subplots(figsize=(12, 6))  # Set figure size
ax.axis('off')  # Hide axes
table = ax.table(cellText=desc_stats.values,
                 colLabels=desc_stats.columns,
                 rowLabels=desc_stats.index,
                 cellLoc='center', loc='center')

# Set table style
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width([i for i in range(len(desc_stats.columns))])

plt.show()


# Plotting correlation heatmap
plt.figure(figsize=(10, 6))
#correlation_matrix = df.corr()
correlation_matrix = df.select_dtypes(include=['number']).corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# Convert relevant columns to numeric
numerical_columns = ['Flow.Duration', 'Total.Length.of.Fwd.Packets',
                     'Total.Length.of.Bwd.Packets', 'Fwd.IAT.Total', 'Bwd.IAT.Total']

# Define numerical columns
numerical_columns = ['Flow.Duration', 'Total.Length.of.Fwd.Packets',
                     'Total.Length.of.Bwd.Packets', 'Fwd.IAT.Total', 'Bwd.IAT.Total']

# Convert to numeric and clean data
for col in numerical_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)  # Remove infinities

# Convert Flow Duration to seconds (assuming it's in microseconds)
df['Flow.Duration'] = df['Flow.Duration'] / 1_000_000
df['Fwd.IAT.Total'] = df['Fwd.IAT.Total'] / 1_000_000
df['Bwd.IAT.Total'] = df['Bwd.IAT.Total'] / 1_000_000

# Plot each numerical column separately
for col in numerical_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col].dropna(), bins=50, color='skyblue')  # More bins for better detail
    plt.xlabel(f"{col} ({'10^6 seconds' if col in ['Flow.Duration', 'Fwd.IAT.Total', 'Bwd.IAT.Total'] else 'units'})")
    plt.ylabel("Count")
    plt.title(f"Distribution of {col}")
    plt.ticklabel_format(style='plain')  # Remove scientific notation
    plt.grid(True)
    plt.show()  #show each plot separately

# Bar plot for ProtocolName (categorical data)
plt.figure(figsize=(10, 6))
sns.countplot(y='ProtocolName', data=df, palette='Set2')
plt.title('Protocol Name Count Distribution')
plt.xlabel('Count')
plt.ylabel('Protocol Name')
plt.show()





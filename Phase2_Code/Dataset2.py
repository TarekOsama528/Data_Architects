import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("C:/Users/osama/OneDrive/Desktop/Data_Architects/Datasets/ocrdataset.csv")


# Generate descriptive statistics
desc_stats = df.describe().round(2)  # Round for better readability

# Create the figure
fig, ax = plt.subplots(figsize=(12, 6))  # Larger size for clarity
ax.axis('off')  # Hide axes

# Create the table with better styling
table = ax.table(cellText=desc_stats.values,
                 colLabels=desc_stats.columns,
                 rowLabels=desc_stats.index,
                 cellLoc='center',
                 loc='center',
                 colColours=['lightgray'] * desc_stats.shape[1])  # Header row shading

table.auto_set_font_size(False)
table.set_fontsize(10)  # Increase font size
table.auto_set_column_width([i for i in range(len(desc_stats.columns))])  # Auto-adjust width

# Add title
plt.title("Descriptive Statistics", fontsize=14, fontweight="bold", pad=20)
plt.show()



# Correlation Heatmap
# Convert "Good" to 1 and "Bad" to 0 (Assuming the column name is 'Label')
if 'Label' in df.columns:
    df['Label'] = df['Label'].map({'Good': 1, 'Bad': 0})

# Ensure all columns are numeric for correlation
df_numeric = df.select_dtypes(include=[np.number])

# Plot the heatmap
plt.figure(figsize=(10, 6))
correlation_matrix = df_numeric.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()




# Plot distributions for numerical columns
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

for col in numerical_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col].dropna(), bins=50, color='skyblue')  # Drop NaNs for cleaner plot
    plt.xlabel(f"{col}")
    plt.ylabel("Count")
    plt.title(f"Distribution of {col}")
    plt.ticklabel_format(style='plain')  # Avoid scientific notation
    plt.grid(True)
    plt.show()

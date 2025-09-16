import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Working directory
excel_path = r"C:\Users\jtgb0\OneDrive\Ambiente de Trabalho\Ground Truth.xlsx"
df = pd.read_excel(excel_path)

# Select relevant columns with the score data and innovation ID
df = df.iloc[:, [0, 3, 4, 5]]
df.columns = ["InnovationID", "Hotness", "Uniqueness", "Market Potential"]

# Convert to numeric
for col in ["Hotness", "Uniqueness", "Market Potential"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Descriptive statistics
stats = df[["Hotness", "Uniqueness", "Market Potential"]].describe().loc[["mean", "std", "min", "max"]].round(2)
print(stats)

# Melt dataframe for plotting
df_melted = df.melt(id_vars="InnovationID", value_vars=["Hotness","Uniqueness","Market Potential"],
                    var_name="Metric", value_name="Score")

# Grouped bar chart
plt.figure(figsize=(12,6))
sns.barplot(data=df_melted, x="InnovationID", y="Score", hue="Metric", palette="Set2")
plt.title("Scores per Innovation", fontsize=16)
plt.xlabel("Innovation ID", fontsize=12)
plt.ylabel("Score (1-10)", fontsize=12)
plt.ylim(0, 11)  
plt.xticks(rotation=45)
plt.legend(title="Metric")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr, t

# Working directories
results_folder = r"C:\Users\jtgb0\OneDrive\Ambiente de Trabalho\results2"
os.makedirs(results_folder, exist_ok=True)
rag_csv = r"C:\Users\jtgb0\OneDrive\Ambiente de Trabalho\results\results.csv"
benchmark_csv = r"C:\Users\jtgb0\OneDrive\Ambiente de Trabalho\results_benchmark\benchmark_results.csv"
ground_truth_excel = r"C:\Users\jtgb0\OneDrive\Ambiente de Trabalho\Ground Truth.xlsx"

# Set style for plots
sns.set_theme(style="whitegrid")

# Function to load ground truth data
def load_ground_truth(path):
    df = pd.read_excel(path).iloc[:, [0,3,4,5]]
    df.columns = ["idea_id","Hotness","Uniqueness","MarketPotential"]
    for col in ["Hotness","Uniqueness","MarketPotential"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# Function to normalize column names and types
def normalize_columns(df):
    rename_map = {
        "Innovation_id":"idea_id", "Innovation_ID":"idea_id", "Run":"run",
        "hotness":"Hotness", "uniqueness":"Uniqueness",
        "Market Potential":"MarketPotential", "market_potential":"MarketPotential"
    }
    df = df.rename(columns=rename_map)
    for col in ["Uniqueness","MarketPotential","Hotness","run"]:

        # Ensure numeric
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# Function to compute bootstrap statistics
def compute_bootstrap_stats(df, method_label):
    agg = df.groupby("idea_id").agg(
        Uniqueness_mean=("Uniqueness","mean"),
        Uniqueness_std=("Uniqueness","std"),
        MarketPotential_mean=("MarketPotential","mean"),
        MarketPotential_std=("MarketPotential","std"),
        Hotness_mean=("Hotness","mean"),
        Hotness_std=("Hotness","std"),
        n_runs=("run","count")
    ).reset_index()
    
    alpha = 0.05  # 95% CI
    
    for metric in ["Uniqueness","MarketPotential","Hotness"]:
        dfree = agg["n_runs"] - 1
        # t critical value for each row
        t_crit = t.ppf(1 - alpha/2, dfree.clip(lower=1))
        
        # CI margin
        agg[f"{metric}_95CI"] = t_crit * agg[f"{metric}_std"] / np.sqrt(agg["n_runs"].clip(lower=1))
        
        # CI bounds
        agg[f"{metric}_CI_lower"] = agg[f"{metric}_mean"] - agg[f"{metric}_95CI"]
        agg[f"{metric}_CI_upper"] = agg[f"{metric}_mean"] + agg[f"{metric}_95CI"]

    agg["method"] = method_label
    return agg

# Function to compute metrics vs ground truth
def compute_metrics_vs_gt(agg_df, gt_df, method_label):
    df = agg_df.merge(gt_df, on="idea_id")
    rows = []
    for metric, mean_col in [("Uniqueness","Uniqueness_mean"),
                             ("MarketPotential","MarketPotential_mean"),
                             ("Hotness","Hotness_mean")]:
        valid = df[[mean_col, metric]].dropna()
        rows.append({
            "method": method_label,
            "metric": metric,
            "MAE": mean_absolute_error(valid[metric], valid[mean_col]),
            "RMSE": np.sqrt(mean_squared_error(valid[metric], valid[mean_col])),
            "Pearson": pearsonr(valid[metric], valid[mean_col])[0] if len(valid)>1 else np.nan,
            "Spearman": spearmanr(valid[metric], valid[mean_col])[0] if len(valid)>1 else np.nan
        })
    return pd.DataFrame(rows)

# Function to compute exact match percentages
def compute_exact_matches_mean(stats_df, gt_df, method_label):

    metrics = ["Uniqueness","MarketPotential","Hotness"]
    
    df = stats_df.merge(gt_df, on="idea_id", how="left")
    
    rows = []
    for metric in metrics:
        mean_col = f"{metric}_mean"
        exact = (df[mean_col].round() == df[metric].round()).mean() # Round to nearest integer for both ground truth and predictions, since expert-given scores are from 1 to 10 integers
        rows.append({                                               
            "method": method_label,
            "metric": metric,
            "ExactMatchPercent": round(100 * exact, 1)
        })
    return pd.DataFrame(rows)

# Function to save CI plots
def save_ci_plot(df, results_dir, method_label):
    for metric, mean_col, ci_col in [
        ("Uniqueness","Uniqueness_mean","Uniqueness_95CI"),
        ("MarketPotential","MarketPotential_mean","MarketPotential_95CI"),
        ("Hotness","Hotness_mean","Hotness_95CI")
    ]:
        plt.figure(figsize=(12,5))
        
        # Compute lower and upper bounds
        y_mean = df[mean_col]
        y_lower = y_mean - df[ci_col]
        y_upper = y_mean + df[ci_col]
        
        # Plot mean line / points
        plt.plot(df["idea_id"], y_mean, 'o-', label='Mean')
        
        # Fill between lower and upper CI
        plt.fill_between(df["idea_id"], y_lower, y_upper, color='b', alpha=0.2, label='95% CI')
        
        plt.ylim(0,10)
        plt.title(f"{metric} — Mean ± 95% CI ({method_label})")
        plt.xlabel("Idea ID")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(results_dir,f"ci_{metric}_{method_label}.png"))
        plt.close()


# Function to save scatter plots vs ground truth
def save_scatter_vs_gt(df, gt_df, results_dir, method_label):
    df_merge = df.merge(gt_df, on="idea_id")
    metrics = [("Uniqueness","Uniqueness_mean"),
               ("MarketPotential","MarketPotential_mean"),
               ("Hotness","Hotness_mean")]
    
    for metric, mean_col in metrics:
        valid = df_merge[[mean_col, metric]].dropna()
        if len(valid) < 2: 
            continue
        
        plt.figure(figsize=(6,6))
        plt.scatter(valid[metric], valid[mean_col], alpha=0.7)
        plt.plot([0,10],[0,10],'r--')
        plt.xlabel("Ground Truth")
        plt.ylabel(f"{method_label} Prediction")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir,f"scatter_{metric}_{method_label}.png"))
        plt.close()

# Function to save exact match barplot
def save_exact_match_barplot(exact_df, results_dir):
    plt.figure(figsize=(10,6))
    sns.barplot(data=exact_df, x="metric", y="ExactMatchPercent", hue="method", errorbar=None)
    plt.title("")
    plt.ylabel("Exact Match %")
    plt.ylim(0,100)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "barplot_exact_matches.png"))
    plt.close()

# Function to save phase 2 choice percentage plot
def save_phase2_choice_percentage_plot(df, results_dir, method_label="RAG"):

    categories = ["A", "B", "C", "D", "E"]
    
    phase2_counts = df['phase2_choice'].value_counts(normalize=True) * 100
    phase2_counts = phase2_counts.reindex(categories, fill_value=0)
    
    plt.figure(figsize=(8,5))
    sns.barplot(x=phase2_counts.index, y=phase2_counts.values, palette="viridis")
    plt.ylabel("Percentage (%)")
    plt.xlabel("Second Retrieval Category")
    plt.title(f"")
    plt.ylim(0, 100)
    
    # Add value labels on top
    for i, v in enumerate(phase2_counts.values):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"phase2_choice_percentage_{method_label}.png"))
    plt.close()

# Load data
gt_df = load_ground_truth(ground_truth_excel)
rag_df = normalize_columns(pd.read_csv(rag_csv))
bench_df = normalize_columns(pd.read_csv(benchmark_csv))

# Bootstrap stats
rag_stats = compute_bootstrap_stats(rag_df, " Agentic RAG")
bench_stats = compute_bootstrap_stats(bench_df, "Base LLM")
rag_stats.to_csv(os.path.join(results_folder,"bootstrap_summary_RAG.csv"), index=False)
bench_stats.to_csv(os.path.join(results_folder,"bootstrap_summary_Base LLM.csv"), index=False)

# Metrics vs Ground Truth
rag_metrics = compute_metrics_vs_gt(rag_stats, gt_df, " Agentic RAG")
bench_metrics = compute_metrics_vs_gt(bench_stats, gt_df, "Base LLM")
metrics_compare = pd.concat([rag_metrics, bench_metrics], ignore_index=True)
metrics_compare.to_csv(os.path.join(results_folder,"metrics_comparison.csv"), index=False)

# Exact match 
rag_exact = compute_exact_matches_mean(rag_stats, gt_df, " Agentic RAG")
bench_exact = compute_exact_matches_mean(bench_stats, gt_df, "Base LLM")
exact_compare = pd.concat([rag_exact, bench_exact], ignore_index=True)
exact_compare.to_csv(os.path.join(results_folder,"exact_match_percentages.csv"), index=False)

# Save plots
save_ci_plot(rag_stats, results_folder, " Agentic RAG")
save_ci_plot(bench_stats, results_folder, "Base LLM")
save_scatter_vs_gt(rag_stats, gt_df, results_folder, " Agentic RAG")
save_scatter_vs_gt(bench_stats, gt_df, results_folder, "Base LLM")
save_exact_match_barplot(exact_compare, results_folder)
save_phase2_choice_percentage_plot(rag_df, results_folder, "Agentic RAG")

# Table
metrics_summary = metrics_compare.merge(exact_compare, on=["method","metric"])
print(metrics_summary.to_string(index=False))
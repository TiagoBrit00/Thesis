import os
import numpy as np
import fitz 
import statistics
import matplotlib.pyplot as plt
import seaborn as sns

# Working directories
companies = {
    "A": {
        "main_folder": r"C:\Users\jtgb0\OneDrive\Ambiente de Trabalho\Data\Company A",
        "output_pdf_folder": r"C:\Users\jtgb0\OneDrive\Ambiente de Trabalho\Output Data\Company A\All Unique",
        "output_text_folder": r"C:\Users\jtgb0\OneDrive\Ambiente de Trabalho\Output Data\Company A\Text Extracted"
    },
    "B": {
        "main_folder": r"C:\Users\jtgb0\OneDrive\Ambiente de Trabalho\Data\Company B",
        "output_pdf_folder": r"C:\Users\jtgb0\OneDrive\Ambiente de Trabalho\Output Data\Company B\All Unique",
        "output_text_folder": r"C:\Users\jtgb0\OneDrive\Ambiente de Trabalho\Output Data\Company B\Text Extracted"
    }
}

# Function to extract PDF statistics
def get_pdf_stats(folder_path):
    pdf_files = []
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, f))

    num_pdfs = len(pdf_files)
    pages_per_pdf = []

    for pdf_path in pdf_files:
        try:
            with fitz.open(pdf_path) as doc:
                pages_per_pdf.append(doc.page_count)
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")

    if pages_per_pdf:
        median_pages = statistics.median(pages_per_pdf)
        mean_pages = statistics.mean(pages_per_pdf)
        min_pages = min(pages_per_pdf)
        max_pages = max(pages_per_pdf)
    else:
        median_pages = mean_pages = min_pages = max_pages = 0

    return {
        "num_pdfs": num_pdfs,
        "median_pages": median_pages,
        "mean_pages": mean_pages,
        "min_pages": min_pages,
        "max_pages": max_pages,
        "pages_list": pages_per_pdf
    }

# Collect stats for companies
all_stats = {}
for company_id, config in companies.items():
    stats = get_pdf_stats(config["main_folder"])
    all_stats[company_id] = stats

    print(f"\nCompany {company_id}:")
    print(f"  Number of PDFs: {stats['num_pdfs']}")
    print(f"  Median pages: {stats['median_pages']}")
    print(f"  Mean pages: {stats['mean_pages']:.2f}")
    print(f"  Min pages: {stats['min_pages']}")
    print(f"  Max pages: {stats['max_pages']}")

# Visualizations
all_pages = [page for stats in all_stats.values() for page in stats['pages_list']]
if all_pages:
    min_pages, max_pages = min(all_pages), max(all_pages)
    bins = np.linspace(min_pages, max_pages, 21)

    for company_id, stats in all_stats.items():
        plt.figure(figsize=(8, 5))
        sns.histplot(stats['pages_list'], bins=bins, kde=False, color="skyblue")
        plt.xlabel("Number of Pages")
        plt.ylabel("Number of Documents")
        plt.title(f"Distribution of PDF Lengths - Company {company_id}")
        plt.xlim(min_pages, max_pages)
        plt.show()

    data = [stats['pages_list'] for stats in all_stats.values()]
    labels = [f"Company {cid}" for cid in all_stats.keys()]

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=data)
    plt.xticks(range(len(labels)), labels)
    plt.ylabel("Number of Pages")
    plt.title("PDF Length Distribution Across Companies")
    plt.show()
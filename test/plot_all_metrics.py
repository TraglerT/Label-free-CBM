import os
import json
import pandas as pd
import matplotlib.pyplot as plt


# Plotting function created by ChatGPT
# --------------------------------------------------
# USER INPUT
# --------------------------------------------------
base_dir = os.path.join(os.path.dirname(os.getcwd()), "saved_models")

Dataset = ["1_OG_cub_cbm_2025_12_15_17_19",
           "1_OG_FullyConnected_cub_cbm_2025_12_19_16_02",
           "1_OG_sparse_cub_cbm_2026_03_04_09_10",
           "1_Dataset_filtered_lowerLambda_cub_cbm_2026_01_14_13_49",
           "1_Dataset_filtered_sparse_cub_cbm_2026_03_05_18_16",
           "1_Dataset_unfiltered_lowerLambda_cub_cbm_2026_01_13_18_43",
           "1_Dataset_unfiltered_sparse_cub_cbm_2026_03_05_18_25",
           "1_self_cub_cbm_2026_03_02_19_51",
           "1_self_sparse_cub_cbm_2025_12_23_18_42",
           "1_self_tooSparse_cub_cbm_2025_12_23_19_16",
           "1_random_word_181_cub_cbm_2026_03_02_19_39",
           "1_ALIGN_OG_cub_cbm_2026_02_28_17_33",
           "1_ALIGN_OG_sparse_cub_cbm_2026_03_04_09_46",
           "1_ALIGN_random_words_178_cub_cbm_2026_03_10_18_53",
           ]          # Folder names
Dataset_names = ["Label-free CBM",
                 "Label-free CBM fully connected",
                 "Label-free CBM sparse",
                 "Filtered CUB Concepts",
                 "Filtered CUB Concepts sparse",
                 "Unfiltered CUB Concepts",
                 "Unfiltered CUB Concepts sparse",
                 "OUR model",
                 "OUR model sparse",
                 "OUR model too sparse",
                 "Random words",
                 "ALIGN",
                 "ALIGN sparse",
                 "ALIGN random words",
                 ]             # Display names

concept_set_mapping = {
    "cub_filtered.txt": "Label-free CBM",
    "cub_filtered_19-12_new.txt": "OUR concept set",
    "cub_random_words_200.txt": "random words",
    "cub_unfiltered_Attributes.txt": "CUB attributes",
    "cub_filtered_Attributes.txt": "CUB attributes",
}

# --------------------------------------------------
# COLLECT DATA
# --------------------------------------------------
rows = []

for folder, dataset_name in zip(Dataset, Dataset_names):
    folder_path = os.path.join(base_dir, folder)

    with open(os.path.join(folder_path, "metrics.txt"), "r") as f:
        metrics = json.load(f)

    with open(os.path.join(folder_path, "args.txt"), "r") as f:
        args = json.load(f)

    concept_set_name = os.path.basename(args["concept_set"])
    if concept_set_name in concept_set_mapping:
         concept_set_name = concept_set_mapping[concept_set_name]
    row = {
        "Dataset": dataset_name,
        "Concept set": concept_set_name,
        "# train concepts": metrics["concepts"]["Concepts before CLIP filter"],
        "# prediction concepts": metrics["concepts"]["Final concepts after training"],
        "VLM": args["clip_name"],
        "clip_cutoff": args["clip_cutoff"],
        "lambda": args["lam"],
        "clip similarity": round(metrics["metrics_concept"]["cleaned val similarity"]*100, 2),
        "weights": metrics["sparsity"]["Non-zero weights"],
        "sparsity (%)": round(metrics["sparsity"]["Percentage non-zero"] * 100, 2),
        "val accuracy": round(metrics["metrics_prediction"]["acc_val"]*100, 2),
    }

    rows.append(row)
# --------------------------------------------------
# CREATE TABLE (Datasets as Rows)
# --------------------------------------------------
df = pd.DataFrame(rows)
df = df.set_index("Dataset")

# Optional: enforce column order
column_order = [
    "Concept set",
    "# train concepts",
    "# prediction concepts",
    "VLM",
    "clip_cutoff",
    "lambda",
    "sparsity (%)",
    "weights",
    "clip similarity",
    "val accuracy",
]

df = df[column_order]

# --------------------------------------------------
# SAVE CSV
# --------------------------------------------------
df.to_csv("results.csv")

# --------------------------------------------------
# SAVE TABLE AS IMAGE
# --------------------------------------------------
fig_height = 1 + len(df) * 0.6
fig, ax = plt.subplots(figsize=(14, fig_height))
ax.axis('off')

table = ax.table(
    cellText=df.values,
    rowLabels=df.index,
    colLabels=df.columns,
    loc='center'
)

for (row, col), cell in table.get_celld().items():
    if row == 0:  # header row
         cell.set_linewidth(2.0)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.2)

plt.tight_layout()
plt.savefig("results.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved:")
print(" - results.csv")
print(" - results.png")
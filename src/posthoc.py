import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman

# def post_hoc(df):
#
#     wine = df["MntWines"]
#     meat = df["MntMeatProducts"]
#     sweet = df["MntSweetProducts"]
#     fruits = df["MntFruits"]
#     fish = df["MntFishProducts"]
#     gold = df["MntGoldProds"]
#
#     stat, p_value = friedmanchisquare(wine, meat, sweet, fruits, fish, gold)
#
#     print(f"Statystyka = {stat:.5f}, p-value = {p_value:.5f}")
#
#     if p_value < 0.05:
#         print("\nWykryto istotne różnice! Przeprowadzamy analizę post-hoc (Wilcoxon + Bonferroni)")
#
#         groups = {"Wino": wine, "Mięso": meat, "Słodycze": sweet, "Owoce": fruits, "Ryby": fish, "Złoto": gold}
#         pairs = list(combinations(groups.keys(), 2))
#         p_values = []
#
#         for g1, g2 in pairs:
#             stat, p = wilcoxon(groups[g1], groups[g2])
#             p_values.append(p)
#             print(f"Test Wilcoxona dla {g1} vs. {g2}: p-value = {p:.5f}")
#
#         corrected_p = np.array(p_values) * len(p_values)
#         corrected_p = np.minimum(corrected_p, 1)
#
#         print("\nWyniki po poprawce Bonferroniego:")
#         for (g1, g2), p_corr in zip(pairs, corrected_p):
#             if p_corr < 0.05:
#                 print(f"Istotna różnica między {g1} i {g2} (p = {p_corr:.5f})")
#             else:
#                 print(f"Brak istotnej różnicy między {g1} i {g2} (p = {p_corr:.5f})")
#
#     data = pd.melt(df, value_vars=["MntWines", "MntMeatProducts", "MntSweetProducts", "MntFruits", "MntFishProducts", "MntGoldProds"], var_name="Produkt",
#                    value_name="Kwota")
#
#     nazwa_map = {
#         "MntWines": "Wino",
#         "MntMeatProducts": "Mięso",
#         "MntSweetProducts": "Słodycze",
#         "MntFruits": "Owoce",
#         "MntFishProducts": "Ryby",
#         "MntGoldProds": "Złoto"
#     }
#     data["Produkt"] = data["Produkt"].map(nazwa_map)
#
#     plt.figure(figsize=(8, 6))
#     sns.boxplot(x="Produkt", y="Kwota", data=data)
#     # plt.title("Porównanie wydatków na różne produkty")
#     # plt.xticks(rotation=45, ha="right")
#     plt.tight_layout()
#     plt.show()
#
#     print("\nMacierz p-value z testu Dunn’a (korekta Bonferroniego):")
#     p_values_dunn = sp.posthoc_dunn(data, val_col='Kwota', group_col='Produkt', p_adjust='bonferroni')
#
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(p_values_dunn, annot=True, cmap="coolwarm", vmin=0, vmax=1, fmt=".3f", linewidths=0.5)
#     plt.tight_layout()
#     plt.show()

# def posthoc2(df):
#     columns = ["MntWines", "MntMeatProducts", "MntSweetProducts", "MntFruits", "MntFishProducts", "MntGoldProds"]
#     df = df[columns]
#
#     ranked = df.apply(rankdata, axis=1, result_type='broadcast')
#
#     mean_ranks = ranked.mean(axis=0)
#
#     stat, p = friedmanchisquare(*[df[col] for col in df.columns])
#     print(f"Test Friedmana: stat={stat:.4f}, p-value={p:.4f}")
#
#     ref_group = "MntFruits"
#     ref_idx = columns.index(ref_group)
#     k = len(columns)
#     n = df.shape[0]
#     z_crit = norm.ppf(1 - 0.05 / 2)
#
#     results = {}
#     for i, col in enumerate(columns):
#         if i == ref_idx:
#             continue
#         z = abs(mean_ranks.iloc[i] - mean_ranks.iloc[ref_idx]) / np.sqrt(k * (k + 1) / (6 * n))
#         p_val = 2 * (1 - norm.cdf(z))
#         results[col] = {"Z": z, "p-value": p_val, "Significant": z > z_crit}
#
#     results_df = pd.DataFrame(results).T
#     print("\nBonferroni-Dunn vs. Owoce:\n")
#     print(results_df)
#
#     plt.figure(figsize=(8, 5))
#     plt.bar(columns, mean_ranks)
#     plt.title("Średnie rangi (Bonferroni-Dunn względem 'Owoce')")
#     plt.ylabel("Średnia ranga")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()

def friedman_nemenyi(df, columns, labels=None, alpha=0.05):
    print("=== Test Friedmana ===")
    stat, p = friedmanchisquare(*[df[col] for col in columns])
    print(f"Statystyka: {stat:.4f}, p-value: {p}")

    if p >= alpha:
        print("Brak istotnych różnic – test Nemenyiego nie zostanie wykonany.")
        return

    print("Istotne różnice – wykonuję test Nemenyiego...\n")

    nemenyi_results = posthoc_nemenyi_friedman(df[columns])
    plt.figure(figsize=(8, 6))
    sns.heatmap(nemenyi_results, annot=True, vmin=0, vmax=1, fmt=".3f", linewidths=0.5)
    plt.title("Test Nemenyiego - wydatki na różne artykuły")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    ranks = df[columns].rank(axis=1)
    mean_ranks = ranks.mean().values
    method_labels = labels if labels else columns

    k = len(columns)
    N = len(df)
    q_alpha = 2.850
    cd = q_alpha * np.sqrt((k * (k + 1)) / (6 * N))
    print(f"Wartość krytyczna (CD): {cd:.5f}")

    plt.figure(figsize=(10, 2))
    y = 0.5

    for rank, label in sorted(zip(mean_ranks, method_labels)):
        plt.plot(rank, y, 'o', markersize=8, color='black')
        plt.text(rank, y + 0.05, label, ha='center', fontsize=10, rotation=45)

    min_rank = min(mean_ranks)
    max_rank = max(mean_ranks)

    plt.hlines(y - 0.1, min_rank, max_rank, color='lightgray')
    plt.hlines(y - 0.2, min_rank, min_rank + cd, color='black', linewidth=2)
    plt.vlines([min_rank, min_rank + cd], y - 0.25, y - 0.15, color='black')
    plt.text(min_rank + cd / 2, y - 0.35, f"CD = {cd:.2f}", ha='center')

    plt.title("CD Diagram")
    plt.yticks([])
    plt.xlabel("Średnia ranga")
    plt.tight_layout()
    plt.show()
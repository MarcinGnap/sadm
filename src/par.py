from scipy.stats import ttest_ind, ttest_rel
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.stats import levene, shapiro, ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# def test_normal(df_s1: DataFrame, df_s2: DataFrame = None, col1: str = None, col2: str = None, fil: str = None,
#                 dependent: bool = True, max_size_diff_ratio: float = 0.05):
#     if dependent:
#         df1 = df_s1[col1].dropna()
#         df2 = df_s2[col1].dropna()
#         stat1, p_norm1 = normaltest(df1)
#         stat2, p_norm2 = normaltest(df2)
#         len1, len2 = len(df1), len(df2)
#         size_diff_ratio = abs(len1 - len2) / min(len1, len2)
#
#         equal_length = size_diff_ratio <= max_size_diff_ratio
#     else:
#         df1 = df_s1[df_s1.League == fil][col1].dropna()
#         df2 = df_s1[df_s1.League == fil][col2].dropna()
#         stat1, p_norm1 = shapiro(df1)
#         stat2, p_norm2 = shapiro(df2)
#         equal_length = True
#
#     stat_var, p_var = levene(df1, df2)
#
#     normal = p_norm1 > 0.05 and p_norm2 > 0.05
#     equal_var = p_var > 0.05
#     if normal and equal_var and equal_length:
#         print("Dane spełniają założenia testu t-Studenta (rozkład normalny, równe wariancje i równoliczność zbiorów)")
#     else:
#         if not normal:
#             print("Co najmniej jedna z grup nie ma rozkładu normalnego")
#         if not equal_var:
#             print("Wariancje są różne")
#         if not equal_length:
#             print("Występuje różnoliczność zbiorów")
#
#     return normal and equal_var and equal_length

# def par_tStudent(df):
#     print("\nT-Student ind:")
#
#     df['InRelationship'] = (df['marital_Married'] + df['marital_Together']) > 0
#
#     married = df[df["marital_Married"] == 1]["Income"]
#     single = df[df["InRelationship"] == 1]["Income"]
#
#     stat, p_value = ttest_ind(married, single, equal_var=False)
#     print("Test t-Studenta dla danych niezależnych (przychód osób w związku vs singli):")
#
#     print(f"Statystyka: = {stat:.5f}, p-value = {p_value:}")
#
#     if p_value < 0.05:
#         print("Istotna różnica w dochodach między osobami w relacji a singlami\n")
#     else:
#         print("Brak istotnej różnicy w dochodach między osobami w relacji a singlami\n")
#
#
# def par_tStudentrel(df):
#     print("\nT-Student rel:")
#
#     filtered = df[df['Purchase_Made'] == 'Yes'].dropna()
#     before = filtered['Sales_Before']
#     after = filtered['Sales_After']
#
#     stat, p = ttest_rel(before, after)
#
#     print("Test t-Studenta dla danych zależnych (ilość zakupionych towarów przez klienta przed i po interwencji):")
#     print(f"t = {stat:.5f}, p = {p:}")
#
#     if p < 0.05:
#         print("Różnica jest statystycznie istotna.")
#     else:
#         print("Brak istotnej różnicy.")

def allpar():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

    df.columns = [col.replace(" (cm)", "").replace(" ", "_") for col in df.columns]

    # anova
    model = ols('sepal_width ~ C(species)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # levene - jednorodność wariancji
    levene_stat, levene_p = levene(
        df[df['species'] == 'setosa']['sepal_width'],
        df[df['species'] == 'versicolor']['sepal_width'],
        df[df['species'] == 'virginica']['sepal_width']
    )

    # shapiro-wilka - norm
    shapiro_stat, shapiro_p = shapiro(model.resid)

    # Test post-hoc Tukey HSD
    tukey = pairwise_tukeyhsd(endog=df['sepal_width'], groups=df['species'], alpha=0.05)

    plt.figure(figsize=(8, 5))
    sns.boxplot(x='species', y='sepal_width', data=df)
    plt.title('Rozkłady szerokości kielicha dla poszczególnych gatunków irysów')
    plt.tight_layout()
    plt.show()

    # t-welcha
    setosa_len = df[df['species'] == 'setosa']['sepal_length']
    versicolor_len = df[df['species'] == 'versicolor']['sepal_length']

    shapiro_setosa = shapiro(setosa_len)
    shapiro_versicolor = shapiro(versicolor_len)

    # t-welcha różnice avg
    welch_t_stat, welch_p_val = ttest_ind(setosa_len, versicolor_len, equal_var=False)

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df[df['species'].isin(['setosa', 'versicolor'])], x='species', y='sepal_length')
    plt.title('Porównanie rozkładów długości kielicha dla gatunków setosa i versicolor')
    plt.tight_layout()
    plt.show()

    print("Test ANOVA")
    print(anova_table)
    print(f"\nLevene (homogeniczność wariancji): stat = {levene_stat:.4f}, p = {levene_p:.4f}")
    print(f"Shapiro-Wilk (normalność reszt): stat = {shapiro_stat:.4f}, p = {shapiro_p:.4f}")
    print("\n=== Post-hoc Tukey HSD ===")
    print(tukey.summary())

    print("\nTest T-Welcha")
    print(f"Setosa – Shapiro: p = {shapiro_setosa.pvalue:.4f}")
    print(f"Versicolor – Shapiro: p = {shapiro_versicolor.pvalue:.4f}")
    print(f"T-Welcha: stat = {welch_t_stat:.4f}, p = {welch_p_val:.4e}")
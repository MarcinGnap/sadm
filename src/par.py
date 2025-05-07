from scipy.stats import ttest_ind, ttest_rel

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

def par_tStudent(df):
    print("\nT-Student ind:")

    df['InRelationship'] = (df['marital_Married'] + df['marital_Together']) > 0

    married = df[df["marital_Married"] == 1]["Income"]
    single = df[df["InRelationship"] == 1]["Income"]

    stat, p_value = ttest_ind(married, single, equal_var=False)
    print("Test t-Studenta dla danych niezależnych (przychód osób w związku vs singli):")

    print(f"Statystyka: = {stat:.5f}, p-value = {p_value:}")

    if p_value < 0.05:
        print("Istotna różnica w dochodach między osobami w relacji a singlami\n")
    else:
        print("Brak istotnej różnicy w dochodach między osobami w relacji a singlami\n")


def par_tStudentrel(df):
    print("\nT-Student rel:")

    filtered = df[df['Purchase_Made'] == 'Yes'].dropna()
    before = filtered['Sales_Before']
    after = filtered['Sales_After']

    stat, p = ttest_rel(before, after)

    print("Test t-Studenta dla danych zależnych (ilość zakupionych towarów przez klienta przed i po interwencji):")
    print(f"t = {stat:.5f}, p = {p:}")

    if p < 0.05:
        print("Różnica jest statystycznie istotna.")
    else:
        print("Brak istotnej różnicy.")
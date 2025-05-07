from scipy.stats import wilcoxon, friedmanchisquare


def nonpar_wilcoxon(df):
    print("\nWilcoxon:")

    meat = df["MntMeatProducts"]
    fish = df["MntFishProducts"]

    stat, p_value = wilcoxon(meat, fish)

    print(f"Statystyka = {stat:.5f}, p-value = {p_value}")

    alpha = 0.05
    if p_value < alpha:
        print("Istnieje istotna statystycznie różnica między wydatkami na mięso i ryby")
    else:
        print("Brak istotnych różnic między wydatkami na mięso i ryby")


def nonpar_friedman(df):
    print("\nFriedman:")

    wine = df["MntWines"]
    meat = df["MntMeatProducts"]
    sweet = df["MntSweetProducts"]

    stat, p_value = friedmanchisquare(wine, meat, sweet)

    print(f"Statystyka = {stat:.5f}, p-value = {p_value}")

    alpha = 0.05
    if p_value < alpha:
        print("Istnieje istotna różnica między wydatkami na różne kategorie produktów")
    else:
        print("Brak istotnych różnic w wydatkach")

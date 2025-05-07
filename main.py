import pandas as pd
from src import par, nonpar, posthoc, logreg, statsanaly

if __name__ == "__main__":

    df = pd.read_csv('./data/ifood_df.csv')
    df2 = pd.read_csv('./data/Sales_without_NaNs_v1.3.csv')


    print("\nDescriptive statistics:")
    statsanaly.statsanaly(df)

    print("\nParametric tests:")
    par.par_tStudent(df)
    par.par_tStudentrel(df2)

    print("\nNonparametric tests:")
    nonpar.nonpar_wilcoxon(df)
    nonpar.nonpar_friedman(df)

    print("\nFiredman + post-hoc nemenyi test:")
    columns = [
        "MntWines", "MntMeatProducts", "MntSweetProducts",
        "MntFruits", "MntFishProducts", "MntGoldProds"
    ]

    labels = ["Wino", "Mięso", "Słodycze", "Owoce", "Ryby", "Złoto"]
    posthoc.friedman_nemenyi(df, columns, labels)

    # print("\nLogistic regression:")
    # logreg.logreg(df)

    # married = df[df["marital_Married"] == 1]["Income"]
    # single = df[df["marital_Single"] == 1]["Income"]
    #
    # stat, p_value = ttest_ind(married, single, equal_var=False)
    #
    # print(f"Test t-Studenta: Statystyka: = {stat:.5f}, p-value = {p_value:.5f}")
    #
    # if p_value < 0.05:
    #     print("Istotna różnica w dochodach między osobami po ślubie a singlami")
    # else:
    #     print("Brak istotnej różnicy w dochodach między osobami po ślubie a singlami")

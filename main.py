import pandas as pd
from src import par, nonpar, posthoc, logreg, statsanaly
from scipy.stats import shapiro, normaltest, anderson


if __name__ == "__main__":

    df = pd.read_csv('./data/ifood_df.csv')
    df2 = pd.read_csv('./data/Sales_without_NaNs_v1.3.csv')

    # print("\nDescriptive statistics:")
    # statsanaly.statsanaly(df)

    print("\nParametric tests:")
    par.allpar()

    # print("\nNonparametric tests:")
    # nonpar.nonpar_wilcoxon(df)
    # nonpar.nonpar_friedman(df)
    #
    # print("\nFiredman + post-hoc nemenyi test:")
    # columns = [
    #     "MntWines", "MntMeatProducts", "MntSweetProducts",
    #     "MntFruits", "MntFishProducts", "MntGoldProds"
    # ]
    #
    # labels = ["Wino", "Mięso", "Słodycze", "Owoce", "Ryby", "Złoto"]
    # posthoc.friedman_nemenyi(df, columns, labels)



    # print("\nLogistic regression:")
    # logreg.logreg(df)

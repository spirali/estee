import pandas as pd
import matplotlib.pyplot as plt
import seaborn


def main():
    data = pd.read_pickle("dataset1.xz")

    columns = [c for c in data.columns if c.endswith("_avg")]

    for c in columns:
        data[c] /= data["min"]

    df = data.groupby("bandwidth")[columns].mean()

    seaborn.heatmap(df, cmap='RdYlGn_r', linewidths=0.5, annot=True)
    plt.show()


main()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    return parser.parse_args()

def draw_heatmap(*args, **kw):
    df = kw["data"].groupby(["bandwidth"])[kw["avg_columns"]].mean()
    seaborn.heatmap(df, cmap='RdYlGn_r', linewidths=0.5, annot=True)

def main():
    args = parse_args()
    data = pd.read_pickle(args.dataset)

    avg_columns = [c for c in data.columns if c.endswith("_avg")]
    min_columns = [c for c in data.columns if c.endswith("_min")]

    m = data[min_columns].min(axis=1)

    for c in avg_columns:
        data[c] /= m

    df = data.groupby(["cluster_name", "bandwidth"])[avg_columns].mean()
    print(df)

    fg = seaborn.FacetGrid(data, col='cluster_name')
    fg.map_dataframe(draw_heatmap, avg_columns=avg_columns) #, "bandwidth", *avg_columns)
#seaborn.heatmap(df, cmap='RdYlGn_r', linewidths=0.5, annot=True)

    plt.show()

if __name__ == "__main__":
    main()

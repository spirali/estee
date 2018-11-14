import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--heatmap", action="store_true")
    parser.add_argument("--violin", action="store_true")
    parser.add_argument("--boxplot", action="store_true")
    parser.add_argument("--lineplot", action="store_true")
    parser.add_argument('--ignore', action='append')
    return parser.parse_args()


def draw_heatmap(*args, **kw):
    df = kw["data"].groupby(["bandwidth"])[kw["avg_columns"]].mean()
    seaborn.heatmap(df, cmap='RdYlGn_r', linewidths=0.5, annot=True)


def draw_violin(*args, **kw):
    df = kw["data"][kw["avg_columns"]]
    df = df.melt(var_name='groups', value_name='vals')
    g = seaborn.violinplot(data=df, x="groups", y="vals")
    #g.set_yscale('log')


def draw_boxplot(*args, **kw):
    df = kw["data"][kw["avg_columns"]]
    df = df.melt(var_name='groups', value_name='vals')
    g = seaborn.boxplot(data=df, x="groups", y="vals")


def draw_lineplot(*args, **kw):
    line = kw["data"].melt(id_vars=['bandwidth'], value_vars=kw["avg_columns"])
    line = line.rename(index=str, columns={"variable": "scheduler"})
    line['scheduler'] = line['scheduler'].map(
        lambda x: x[:-4] if x.endswith("_avg") else x)

    line['bandwidth'] = line['bandwidth'].astype(str)
    g = seaborn.lineplot(x="bandwidth", y="value", hue="scheduler", data=line)
    g.set(yscale="log", xlabel="Bandwidth", ylabel="Makespan")


def main():
    args = parse_args()

    data = pd.read_pickle(args.dataset)

    if args.ignore:
        for c in args.ignore:
            del data[c]

    avg_columns = [c for c in data.columns if c.endswith("_avg")]
    min_columns = [c for c in data.columns if c.endswith("_min")]

    m = data[min_columns].min(axis=1)

    for c in avg_columns:
        data[c] /= m

    df = data.groupby(["cluster_name", "bandwidth"])[avg_columns].mean()

    if args.heatmap:
        fg = seaborn.FacetGrid(data, col='cluster_name')
        fg.map_dataframe(draw_heatmap, avg_columns=avg_columns)

    if args.violin:
        fg = seaborn.FacetGrid(data, col='cluster_name', row='bandwidth')
        fg.map_dataframe(draw_violin, avg_columns=avg_columns)

    if args.boxplot:
        fg = seaborn.FacetGrid(data, col='cluster_name', row='bandwidth')
        fg.map_dataframe(draw_boxplot, avg_columns=avg_columns)

    if args.lineplot:
        fg = seaborn.FacetGrid(data, col='cluster_name')
        fg.map_dataframe(draw_lineplot, avg_columns=avg_columns).add_legend()

    plt.show()

if __name__ == "__main__":
    main()

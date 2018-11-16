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
    return parser.parse_args()


def draw_heatmap(*args, **kw):
    df = kw["data"].groupby(["bandwidth", "scheduler_name"])["score"].mean()
    df = df.unstack()
    seaborn.heatmap(df, cmap='RdYlGn_r', linewidths=0.5, annot=True)


def draw_violin(*args, **kw):
    seaborn.violinplot(data=kw["data"], x="scheduler_name", y="score")


def draw_boxplot(*args, **kw):
    g = seaborn.boxplot(data=kw["data"], x="scheduler_name", y="score")
    g.set(ylabel="Makespan")


def draw_lineplot(*args, **kw):
    data = kw["data"]
    data['bandwidth'] = data['bandwidth'].astype(str)
    g = seaborn.lineplot(data=data, x="bandwidth", y="score", hue="scheduler_name")
    g.set(yscale="log", xlabel="Bandwidth", ylabel="Makespan")


def main():
    args = parse_args()
    data = pd.read_pickle(args.dataset)

    # normalize by minimum schedule found for each graph/cluster/bandwidth combination
    data["score"] = data\
        .groupby(["graph_id", "cluster_name", "bandwidth"])\
        .transform(lambda g: g / g.min())

    # calculate average for each graph/cluster/bandwidth/scheduler/imode combination
    data = data.groupby(["graph_id", "cluster_name", "bandwidth", "scheduler_name", "imode"])\
        ["score"].mean().reset_index()

    if args.heatmap:
        fg = seaborn.FacetGrid(data, col='cluster_name')
        fg.map_dataframe(draw_heatmap)

    if args.violin:
        fg = seaborn.FacetGrid(data, col='cluster_name', row='bandwidth')
        fg.map_dataframe(draw_violin)

    if args.boxplot:
        fg = seaborn.FacetGrid(data, col='cluster_name', row='bandwidth')
        fg.map_dataframe(draw_boxplot)

    if args.lineplot:
        fg = seaborn.FacetGrid(data, col='cluster_name')
        fg.map_dataframe(draw_lineplot).add_legend()

    plt.show()


if __name__ == "__main__":
    main()

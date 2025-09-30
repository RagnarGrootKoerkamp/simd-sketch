#!/usr/bin/env python

import pandas as pd
from pathlib import Path

file = Path("stats").read_text()

data = []
for line in file.splitlines():
    name, tp, n, k, s, b, sketch, dist, corr = line.split(" ")
    n = int(n)
    k = int(k)
    s = int(s)
    b = int(b)
    sketch = float(sketch)
    dist = float(dist)
    corr = float(corr)
    vals = [name, tp, n, k, s, b, sketch, dist, corr]
    if s == 65536:
        continue
    data.append(vals)

# Make a dataframe from data
df = pd.DataFrame(
    data, columns=["name", "type", "n", "k", "s", "b", "sketch", "dist", "corr"]
)

# print(df)
import tabulate

# Drop 'n' column
df2 = df.copy()
df2 = df2.drop(columns=["n", "k"])

print(
    tabulate.tabulate(
        df2,
        headers=df2.columns,
        tablefmt="orgtbl",
        floatfmt=[""] * 4 + [".1f", "0.3f", "0.4f"],
        showindex=False,
    )
)

df["sketch_one"] = df["sketch"] / df["n"]
df["dist_one"] = df["dist"] / (df["n"] * (df["n"] - 1) / 2)
df["sketch_thp_MB"] = 2.0 / df["sketch_one"]
df["dist_thp_M"] = 1 / df["dist_one"] / 1000000
df["c"] = 1 - df["corr"]


import matplotlib.pyplot as plt
import seaborn as sns

df["variant"] = df["name"] + " " + df["type"]

colormap = {
    "SimdSketch bottom": "pink",
    "SimdSketch bucket": "red",
    # "SimdSketch bottom": "#ffc107",
    # "SimdSketch bucket": "#ffc107",
    "BinDash-rs bucket": "black",
    "BinDash bottom": "lightblue",
    "BinDash bucket": "blue",
}

# for k, g in df.groupby(["variant", "type", "b"]):
#     plt.plot(g["sketch_thp_MB"], g["c"], label=None, color=colormap[k[0]], lw=0.7)
# for k, g in df.groupby(["variant", "type", "s"]):
#     plt.plot(
#         g["sketch_thp_MB"], g["c"], label=None, color=colormap[k[0]], lw=1, ls="--"
#     )
# sns.scatterplot(
#     data=df,
#     x="sketch_thp_MB",
#     y="c",
#     hue="variant",
#     size="s",
#     style="type",
#     sizes=[20, 80, 200],
#     palette=colormap,
# )
# plt.gca().invert_yaxis()
# plt.xscale("log")
# plt.yscale("log")
# # Logarithmic y scale towards 1
# plt.xlabel("Sketch throughput (MB/s)")
# plt.ylabel("1 - Correlation")
# plt.title("Correlation vs sketch throughput")
# plt.savefig("plots/sketching.svg", dpi=300, bbox_inches="tight")
# # plt.show()

for k, g in df.groupby(["variant", "type", "b"]):
    # Hide plot from legend.
    plt.plot(g["dist_thp_M"], g["c"], label="_x", color=colormap[k[0]], lw=0.7)
for k, g in df.groupby(["variant", "type", "s"]):
    plt.plot(g["dist_thp_M"], g["c"], label="_x", color=colormap[k[0]], lw=1, ls="--")
sns.scatterplot(
    data=df,
    x="dist_thp_M",
    y="c",
    hue="variant",
    size="s",
    style="type",
    sizes=[20, 80, 150, 200, 250],
    markers=["o", "*"],
    palette=colormap,
)
plt.gca().invert_yaxis()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Comparison throughput (M/s)")
plt.ylabel("1-correlation")
plt.title("Correlation vs comparison throughput")
plt.gcf().set_size_inches(11, 4.5)
plt.savefig("plots/comparison.svg", dpi=300, bbox_inches="tight")
plt.close()

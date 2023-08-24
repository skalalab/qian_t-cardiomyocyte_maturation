# -*- coding: utf-8 -*-
"""
Created on Mon May  1 09:03:09 2023

@author: ddesa
"""

from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

path_dataset = Path(r"Z:\0-Projects and Experiments\TQ - cardiomyocyte maturation\datasets\RNA Seq")
path_csv = path_dataset / "RNAseq_normalized_heatmap.csv"
all_df = pd.read_csv(path_csv)

#seaborn.clustermap(data, pivot_kws=None, method='average', metric='euclidean', 
#z_score=None, standard_scale=None, figsize=(10, 10), cbar_kws=None, row_cluster=True, 
#col_cluster=True, row_linkage=None, col_linkage=None, row_colors=None, col_colors=None, 
# mask=None, dendrogram_ratio=0.2, colors_ratio=0.03, cbar_pos=(0.02, 0.8, 0.05, 0.18), 
#tree_kws=None, **kwargs)

list_cols = list(all_df.keys())


list_cols.remove("Category")
list_cols.remove("Gene")

# remove weird valuee
all_df = all_df[~all_df['Gene'].isin(['ACTN2'])]


# list_cols.remove("H9-D20")

# b = all_df[all_df ==1]
# b.groupby(["Gene"]).mean()
# b.groupby(["Gene"]).std()


# linkage methods = ["average", "weighted","centroid", "median"]

method = 'average'
metric='euclidean'

sns.clustermap(all_df[list_cols], 
            method = method, 
            metric=metric, 
            row_cluster=False,
            z_score=0, 
            # standard_scale=0, 
            cmap="inferno", 
            vmin=-2, 
            vmax=8, 
            dendrogram_ratio=(.2, .1), 
            cbar_pos=(0.03, .2, .05, .4),
            yticklabels=all_df['Gene'].values,
            figsize = (8, 20)
            )


path_output = Path(r"Z:\0-Projects and Experiments\TQ - cardiomyocyte maturation\figures\RNAseq")
plt.savefig(path_output / f"heatmap_{metric}_{method}.svg")
plt.show()

#%%%



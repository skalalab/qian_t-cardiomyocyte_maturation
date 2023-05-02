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
list_cols.remove("H9-D20")
sns.clustermap(all_df[list_cols], method = 'average', metric='euclidean', z_score=1, cmap="inferno", vmin=-2, vmax=8, 
               dendrogram_ratio=(.2, .15), cbar_pos=(0.03, .2, .05, .4))
plt.show()
# sns.heatmap(all_df[list_cols], method='average')
# plt.show() 
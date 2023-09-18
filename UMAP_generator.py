# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:21:16 2023

@author: ddesa from code by econtreras, jriendeau
"""

#UMAP code for CM maturation exp

#%% Section 1 - Import required packages

import umap.umap_ as umap
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

import holoviews as hv
hv.extension("bokeh")
from holoviews import opts
from holoviews.plotting import list_cmaps
import os
#from yellowbrick.text import UMAPVisualizer

#%% Section 2 - Read in and set up dataframe 

#Read in dataframe    
path_output = Path(r'Z:\0-Projects and Experiments\TQ - cardiomyocyte maturation\analysis\UMAPs')
CSV = path_output / 'Last_day.csv'
all_df = pd.read_csv(CSV)
df_data = all_df.copy()

#%% Section 3 - UMAP creation

list_features = [
    # 'nadh_intensity_mean',
    # "nadh_tau_mean_mean",
    # "nadh_a2_mean",
    # "nadh_t1_mean",
    # "nadh_t2_mean",
    # 'fad_intensity_mean',
    "fad_tau_mean_mean",
    "fad_a2_mean",
    "fad_t1_mean",
    "fad_t2_mean",
    # # "redox_ratio_mean",
    # "redox_ratio_norm_mean",
    # "area",
    # "flirr_mean",
    # "eccentricity",
        ]


#generate UMAP
data = df_data[list_features].values
scaled_data = StandardScaler().fit_transform(data)
reducer = umap.UMAP(
               n_neighbors= 10,
               min_dist= 1,   
               metric='euclidean',  #distance measure options are 'cosine', euclidean', 'manhattan', or 'minkowski'
               n_components=2,
               random_state=0
           )
       
fit_umap = reducer.fit(scaled_data)


## additional params for holoviews
#The legend_entries parameter will determine what group we are color-coding by
hover_vdim = "subtype"
legend_entries = "subtype"

#generate UMAP embedding
df_data = df_data.copy()
df_data["umap_x"] = fit_umap.embedding_[:,0]
df_data["umap_y"] = fit_umap.embedding_[:,1]

kdims = ["umap_x"]
vdims = ["umap_y", hover_vdim]
list_entries = np.unique(df_data[legend_entries])
#
                    
scatter_umaps = [hv.Scatter(df_data[df_data[legend_entries] == entry], kdims=kdims, 
                            vdims=vdims, label=entry) for entry in list_entries]

#Parameters to control plotting of UMAP in holoviews


colors = [
            "#17DA4C", "#1743DA", "#DA17A5", "#DAAD17"
            ]
for scatter, color in zip(scatter_umaps, colors):
    scatter.opts(color=color)

overlay = hv.Overlay(scatter_umaps)
overlay.opts(
    opts.Scatter(
        #fontsize={'labels': 14, 'xticks': 12, 'yticks': 12},
        fontscale=1.75,
        size = 4,
        alpha = 0.5,
        tools=["hover"],
        muted_alpha=0,
        aspect="equal",
        width=600, 
        height=600),
    opts.Overlay(
        title='Last imaging day: FAD',
        legend_opts={"click_policy": "hide"},
        show_legend=True,
        legend_position='right'
        )       
    )


#%%
#Saves an interactive holoviews plot as a .HTML file
pathsave = path_output #Path(os.path.join(path_output, "graphs"))
hv.save(overlay, pathsave / 'Last day, FAD parameters.html')


from IPython.display import display
fig = hv.render(overlay, backend='matplotlib')
display(fig)

print("UMAP complete")
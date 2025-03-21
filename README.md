# qian_t-cardiomyocyte_maturation
### description of code and data for "A label-free method to monitor metabolism during long-term culture of human pluripotent stem cell-derived cardiomyocytes" by T Qian, DE Desa, E Contreras Guzman, W Zhao, X Zhang, SP Palacek, MC Skala (2024).

* To reproduce scatter plots and bar graphs, the included Excel files can be used ('data' folder).
	- `cyto_merged_data.csv` contains the single cell data for H9 and Long QT lines 
	- `stiffness_all_props_cytoplasm.csv` contains the single cell data for H9 for the stiffness experiment. 
	- `train.py` contains the code to train the classifer and plot the confusion matrix, ROC curve, and feature importance.
	- `umap.Rmd` contains the code to plot UMAP. 
* To reproduce 2-photon FLIM data, the cell-analysis-tools repository (https://github.com/skalalab/cell-analysis-tools) can be used with the 'regionprops' script above. Fluorescence decays/fits available upon request.



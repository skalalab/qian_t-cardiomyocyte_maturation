from pathlib import Path
import re
import tifffile
import matplotlib.pylab as plt
import numpy as np


import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
#%%

path_dataset =  Path(r"Z:\0-Projects and Experiments\TQ - cardiomyocyte maturation\datasets\H9")

list_path_masks_cell = list(path_dataset.rglob("*_mask_cell.tiff"))

for path_mask_cell in list_path_masks_cell:
    pass

    mask_cell = tifffile.imread(path_mask_cell)
    filename_mask_nuclei = path_mask_cell.stem.rsplit("_",1)[0]
    mask_nuclei = tifffile.imread(path_mask_cell.parent / f"{filename_mask_nuclei}_nuclei.tiff")

    mask_cyto = mask_cell * np.invert(mask_nuclei > 0)
    
    # visualize plots 
    fig, ax = plt.subplots(1,3, figsize=(10,4))
    fig.suptitle(path_mask_cell.name)
    
    ax[0].set_axis_off()
    ax[0].set_title('whole_cell')
    ax[0].imshow(mask_cell)
    
    ax[1].set_axis_off()
    ax[1].set_title('nuclei')
    ax[1].imshow(mask_nuclei)
    
    ax[2].set_axis_off()
    ax[2].set_title('cyto')
    ax[2].imshow(mask_cyto)
    plt.show()
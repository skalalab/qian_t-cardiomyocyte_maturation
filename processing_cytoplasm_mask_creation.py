from pathlib import Path
import re
import tifffile
import matplotlib.pylab as plt
import numpy as np

from skimage.measure import regionprops
from skimage.morphology import closing, disk, remove_small_objects, label

from cell_analysis_tools.visualization import compare_images

from tqdm import tqdm

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
#%%

# path_dataset =  Path(r"Z:\0-Projects and Experiments\TQ - cardiomyocyte maturation\datasets\H9\masks\Edited")
# path_dataset =  Path(r"Z:\0-Projects and Experiments\TQ - cardiomyocyte maturation\datasets\Long QT")

# linux 
# path_dataset = Path(r"/mnt/Z/0-Projects and Experiments/TQ - cardiomyocyte maturation/datasets/H9/DAY 30/masks/Edited")
path_dataset = Path(r"Z:\Danielle\Misc\Darcie iPSCs\2023-11-29")


list_timepoint_folders = [p for p in path_dataset.glob("*") if p.is_dir()]

for path_timepoint_dir in tqdm(list_timepoint_folders[:]):
    pass
    
    path_timepoint_masks = path_timepoint_dir
    list_path_masks_cell = list(path_timepoint_masks.rglob("*_photons_cellpose.tiff"))
    
    for path_mask_cell in tqdm(list_path_masks_cell[:]):
        pass
        print(path_mask_cell.name)
        mask_cell = tifffile.imread(path_mask_cell)
        
        # plt.imshow(mask_cell)
        # plt.show()
        
        props = regionprops(mask_cell)
        
        # areas 
        list_areas = [p.area for p in props]
        # plt.imshow(mask_cell)
        # plt.show()
        
        # graph distribution of connected components to determien cutoff rea size
        # plt.title(path_mask_cell.name)
        # plt.hist(list_areas, histtype="step", bins=100)
        # plt.show()
        
        # remove small objects
        mask_no_small_objects = remove_small_objects(mask_cell,min_size=100)
        
        mask_temp = np.zeros_like(mask_cell)
        for label_value in np.unique(mask_no_small_objects):
            pass
        
            # isolate roi
            mask_roi = mask_cell == label_value
            # plt.imshow(mask_roi)
            # plt.show()
            
            # get largest region
            mask_roi_labels_mask = label(mask_roi)
            roi = sorted(regionprops(mask_roi_labels_mask), key=lambda r : r.area, reverse=True)[0]
            mask_largest = mask_roi_labels_mask == roi.label
            # compare_images("labeled", mask_roi_labels_mask, "largest region", mask_largest)
            # plt.imshow(mask_roi_labels_mask)
            # plt.show()
            
            # fill holes in roi
            mask_closing = closing(mask_largest,footprint=disk(2))
            # plt.imshow(mask_closing)
            # plt.show()
            
            mask_temp[mask_closing] = label_value
        
            # compare_images('original', mask_roi, "closing", mask_closing )
        
        mask_cell = mask_temp
        
        # compare before and after images
        # compare_images('original', mask_cell, "closing", mask_temp )
        
        # fix mask nuclei
        
        
        filename_mask_nuclei = path_mask_cell.stem.rsplit("_",1)[0]
        mask_nuclei = tifffile.imread(path_mask_cell.parent / f"{filename_mask_nuclei}_cellpose_nuclei.tiff")
    
        mask_cyto = mask_cell * np.invert(mask_nuclei > 0)
        
        # visualize plots 
        # fig, ax = plt.subplots(1,3, figsize=(10,4))
        # fig.suptitle(path_mask_cell.name)
        
        # ax[0].set_axis_off()
        # ax[0].set_title('whole_cell')
        # ax[0].imshow(mask_cell)
        
        # ax[1].set_axis_off()
        # ax[1].set_title('nuclei')
        # ax[1].imshow(mask_nuclei)
        
        # ax[2].set_axis_off()
        # ax[2].set_title('cyto')
        # ax[2].imshow(mask_cyto)
        # plt.show()
        
        filename_cyto = path_mask_cell.stem.rsplit('_',1)[0] + "_cyto.tiff"
        ## save mask 
        tifffile.imwrite(path_timepoint_masks / filename_cyto, mask_cyto)
    
    
    
    
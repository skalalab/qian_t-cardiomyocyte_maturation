from pathlib import Path
from cell_analysis_tools.io import load_image

from cell_analysis_tools.visualization import compare_images
from cell_analysis_tools.io import load_image

import tifffile

import re

import numpy as np

import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
#%%
suffixes = {
    'im_photons': '_photons.asc',
    'a1[%]': '_a1_%_.asc',
    'a2[%]': '_a2_%_.asc',
    't1': '_t1.asc',
    't2': '_t2.asc',
    'chi': '_chi.asc',
    'sdt': '.sdt'
 }
#%% LOAD DATA

path_dataset = Path(r"Z:\0-Projects and Experiments\TQ - cardiomyocyte maturation\dataset")

# select dirs
list_timepoint_dirs = list(path_dataset.glob("*"))

for dir_timepoint in list_timepoint_dirs[-1:]:
    pass

    list_all_files = list(map(str, list(dir_timepoint.rglob('*n_*'))))
    list_photon_images =  list(filter(re.compile(f".*{suffixes['im_photons']}").search, list_all_files))
    
    for path_im_photons in list_photon_images[:]:
        pass
    
        handle_nadh = Path(path_im_photons).stem.rsplit("_",1)[0]
    
        im_nadh = load_image(path_im_photons)
        im_a1 = load_image(dir_timepoint / (handle_nadh + suffixes['a1[%]']))
        im_a2 = load_image(dir_timepoint / (handle_nadh + suffixes['a2[%]']))
        im_t1 = load_image(dir_timepoint / (handle_nadh + suffixes['t1']))
        im_t2 = load_image(dir_timepoint / (handle_nadh + suffixes['t2']))
        
        ############## THRESHOLDING
        title_mask = ""
        
        #INTENSITY
        percentile_upper = 80
        mask_intensity = im_nadh > np.percentile(im_nadh,percentile_upper)
        
        title_intensity_thresh = f"int_percentile: {percentile_upper}"
        # plt.title(percentile_upper)
        # plt.imshow(mask_intensity)
        # plt.show()
        compare_images(im_nadh, "original", 
                       mask_intensity, "intensity",
                       suptitle=f"intensity percentile {percentile_upper}")

        # T1
        # bound_lower_t1 = 350 # ns to ps
        # bound_upper_t1 = 500 # ns to ps
        # mask_t1 = (im_t1 >bound_lower_t1)  &  (im_t1 <bound_upper_t1) 
        # title_t1_thresh = f" {bound_lower_t1} ps t1 < {bound_upper_t1}ps"
        # plt.title(title_t1_thresh)
        # plt.imshow(mask_t1)
        # plt.show()
        # compare_images(im_nadh, "original", mask_t1, "t1")
        
        # plot histogram
        # l = im_t1.flatten()
        # l = l[l < 5000]
        # plt.title("nadh t1")
        # plt.hist(l, histtype='step', bins=1000)
        # plt.show()
        
        # T2
        bound_lower_t2 = 2.5 * 1e3# 2.7 * 1e3 # ns to ps
        # bound_upper_t2 = 5 * 1e3 # np.max(im_t2) # 5 * 1e3 # ns to ps
        mask_t2 = (im_t2 >bound_lower_t2) # &  (im_t2 <bound_upper_t2) 
        title_t2_thresh = f" t2 > {bound_lower_t2}ps"
        # plt.title(title_t2_thresh)
        # plt.imshow(mask_t2)
        # plt.show()
        compare_images(im_nadh, "original", mask_t2 , "t2", suptitle=title_t2_thresh)

        # plot histogram
        l = im_t2.flatten()
        # l = l[l < 5000]
        # plt.title("nadh t2")
        # plt.hist(l, histtype='step', bins=1000)
        # plt.show()

        ### ASSEMBLE MASK
        mask = mask_intensity * mask_t2 # mask_t1
        compare_images(im_nadh, "original", 
                       mask, "combined mask",
                       suptitle="combined")

        # mask = mask_intensity
        
        # save mask
        path_output = Path(path_im_photons).parent / "masks_mitochondria"
        if not path_output.exists():
            path_output.mkdir(exist_ok=True)
        path_mask = path_output / f"{Path(path_im_photons).stem}_mask_mitochondria.tiff"
        mask_mito = np.array(mask, dtype=np.uint8) * 255
        
        # tifffile.imwrite(path_mask,mask_mito , imagej=True)
        
        # save images 
        path_figures = Path(path_im_photons).parent / "image_grid"
        if not path_figures.exists():
            path_figures.mkdir(exist_ok=True)
            
        
        ############## PLOT FIGURE
        # fig, ax = plt.subplots(2,3, figsize=(6,4))
        # fig.suptitle(f"{dir_timepoint.stem} | {handle_nadh}")
    
        # ax[0,0].set_title("intensity")
        # ax[0,0].imshow(im_nadh)
        # ax[0,0].set_axis_off()
        
        # ax[1,0].set_title("mask")    
        # ax[1,0].imshow(mask)
        # ax[1,0].set_axis_off()
        
        # ax[0,1].set_title("a1")    
        # ax[0,1].imshow(im_a1)
        # ax[0,1].set_axis_off()
    
        # ax[0,2].set_title("a2")    
        # ax[0,2].imshow(im_a2)
        # ax[0,2].set_axis_off()
        
        # ax[1,1].set_title("t1")    
        # ax[1,1].imshow(im_t1)
        # ax[1,1].set_axis_off()
        
        # ax[1,2].set_title("t2")    
        # ax[1,2].imshow(im_t2)
        # ax[1,2].set_axis_off()
        
        # plt.savefig(path_figures / f"{Path(path_im_photons).stem}_grid.png")
        
        plt.show()
        
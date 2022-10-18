import matplotlib.pyplot as plt
import matplotlib as mpl
# %matplotlib inline
mpl.rcParams['figure.dpi'] = 300
from cellpose import  io, models
from pathlib import Path
import tifffile

if __name__ == "__main__":
    
    #%% Load Images
    # HERE = Path(__file__).resolve().absolute().parent
    
    # 
    
    path_dataset = Path(r"Z:\0-Projects and Experiments\TQ - cardiomyocyte maturation\dataset")
    list_path_folders = list(path_dataset.glob("*"))
    
    for p in list_path_folders:
        pass
        files = list(p.glob("*n_photons.tiff"))
        
        ## parameters 
        save_results = False
        bool_mask_cell = False # if false makes nuclei masks

        #%%
        # RUN CELLPOSE
        
        ##### PARAMETERS TO ADJUST
        
        
        # masks cell
        # I think for the borders, cyto2 is fine. I have the flow_threshold and cellprob pretty low (-3.5) and cell diam at 25
        if bool_mask_cell:
            dict_Cellpose_params = {
                "gpu" : True,
                'model_type' : 'cyto2',
                'net_avg' : True,
                }
            
            dict_eval_params = {
                'diameter' : 25,
                'cellprob_threshold' : -3.5, 
                'flow_threshold' : -3.5 # model match threshold on GUI
                }
        else:
            # masks nuclei
            # For nuclei, the COBA adherent nuclei is working well. Diameter is at 10.8, flow at 0.4, and cellprob at 0
            dict_Cellpose_params = {
                "gpu" : True,
                'pretrained_model' : r'Z:\0-segmentation\cellpose\COBA\Models\Adherent Cell\AdherentNuclei.zip',
                # "Z:\0-segmentation\cellpose\COBA\Models\Adherent Cell\AdherentNuclei.zip"
                # "Z:\0-segmentation\cellpose\COBA\Models\Organoid\OrganoidCells.zip"
                # "Z:\0-segmentation\cellpose\COBA\Models\Organoid\OrganoidNuclei.zip"
                'net_avg' : True,
                }
            
            dict_eval_params = {
                'diameter' : 10.8,
                'cellprob_threshold' : 0.4, 
                'flow_threshold' : 0 # model match threshold on GUI
                }
            
        ###############################
        
        
        # DEFINE CELLPOSE MODEL
        # model_type='cyto' or model_type='nuclei'
        # model = models.Cellpose(gpu=False, model_type='cyto')
        model = models.CellposeModel(**dict_Cellpose_params)
        
        # define CHANNELS to run segementation on
        # grayscale=0, R=1, G=2, B=3
        # channels = [cytoplasm, nucleus]
        # if NUCLEUS channel does not exist, set the second channel to 0
        channels = [[0,0]]
        # IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
        # channels = [0,0] # IF YOU HAVE GRAYSCALE
        # channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
        # channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus
        
        # or if you have different types of channels in each image
        # channels = [[2,3], [0,0], [0,0]]
        
        # if diameter is set to None, the size of the cells is estimated on a per image basis
        # you can set the average cell `diameter` in pixels yourself (recommended) 
        # diameter can be a list or a single number for all images
        
        # you can run all in a list e.g.
        # >>> imgs = [io.imread(filename) in for filename in files]
        # >>> masks, flows, styles, diams = model.eval(imgs, diameter=None, channels=channels)
        # >>> io.masks_flows_to_seg(imgs, masks, flows, diams, files, channels)
        # >>> io.save_to_png(imgs, masks, flows, files)
        
        # or in a loop
        for chan, filename in zip(channels*len(files), files):
            pass
            img = tifffile.imread(filename)
            # masks, flows, styles, diams = model.eval(img, diameter=None, channels=chan)
            masks, flows, styles = model.eval(img, channels=chan, **dict_eval_params)
    
            # save results so you can load in gui
            if save_results:
                io.masks_flows_to_seg(img, masks, flows, filename, chan)
        
            # save results as png
            # io.save_masks(img, masks, flows, filename, tif=True)
            if bool_mask_cell:
                filename_mask =  f"{filename.stem.strip()}_mask_cell.tiff"
            else:
                filename_mask =  f"{filename.stem.strip()}_mask_nuclei.tiff"
                
            path_masks_folder = filename.parent / "masks"
            if not path_masks_folder.exists():
                path_masks_folder.mkdir(exist_ok=True)
            
            # tifffile.imwrite( path_masks_folder / filename_mask, masks)
            
        #%%
        
            # DISPLAY RESULTS
            from cellpose import plot
            
            
            fig = plt.figure(figsize=(12,5))
            fig.suptitle(f"{filename.parent.stem}  | {filename.name}")
            plot.show_segmentation(fig, img, masks, flows[0], channels=chan)
            plt.tight_layout()
            plt.show()
        
        
        
        
        
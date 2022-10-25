from pathlib import Path
import os
from tqdm import tqdm

# path_datset = Path(r"Z:\0-Projects and Experiments\TQ - cardiomyocyte maturation\datasets\H9 (copy)")
# path_dataset = Path(r"Z:\0-Projects and Experiments\TQ - cardiomyocyte maturation\datasets\Long QT")
# path_dataset = Path(r"Z:\0-Projects and Experiments\TQ - cardiomyocyte maturation\datasets\Long QT")
# path_dataset = Path(r"Z:\0-Projects and Experiments\TQ - cardiomyocyte maturation\datasets\stiffness\H9")

list_folders = [p for p in path_dataset.glob("*") if p.is_dir()]

for path_day in tqdm(list_folders[:]):
    pass
    prefix_day = path_day.stem.replace(' ','_')
    #iterate through cell lines
    path_cell_line_dir = [p for p in path_day.glob("*") if p.is_dir]
    for path_cell_line in path_cell_line_dir[:]:
        pass
        # grab all files
        list_all_files = [p for p in path_cell_line.rglob('*') if p.is_file()]
        
        #create prefix for each cell line
        cell_line = path_cell_line.stem
        
        prefix_all = f"H9_{prefix_day}_{path_cell_line.name}_"
        
        # prefix_day_formatted = prefix_day[:3] + '_' + prefix_day[3:]
        # if cell_line == '106':
        #     prefix_all = f"UCSD106i-2-5_{prefix_day_formatted}_"
        # elif cell_line == '102':
        #     prefix_all = f"UCSD102i-2-1_{prefix_day_formatted}_"
            
        for path_file in list_all_files:
            pass
            filename_new = prefix_all + path_file.name
            print(f"{path_file.name} ==> {filename_new}")
            # os.rename(path_file, path_file.parent / filename_new )
    
    
# for path_folder in list_folders:
#     pass
#     list_all_files = [path_file for path_file in path_folder.rglob("*") if path_file.is_file()]

#     prefix = f"H9_{path_folder.name.replace(' ', '_')}_"
#     for path_file in  list_all_files:
#         pass
#         filename_new = prefix + path_file.name
#         # os.rename(path_file, path_file.parent / filename_new )
from pathlib import Path
import os

# path_datset = Path(r"Z:\0-Projects and Experiments\TQ - cardiomyocyte maturation\datasets\H9 (copy)")


path_dataset = Path(r"Z:\0-Projects and Experiments\TQ - cardiomyocyte maturation\datasets\Long QT")


list_folders = [p for p in path_dataset.glob("*") if p.is_dir()]


for path_day in list_folders[:]:
    pass
    prefix_day = path_day.stem.replace(' ','_')
    #iterate through cell lines
    path_cell_line_dir = [p for p in path_day.glob("*") if p.is_dir]
    for path_cell_line in path_cell_line_dir:
        pass
        # grab all files
        list_all_files = [p for p in path_cell_line.glob('*') if p.is_file()]
        
        prefix_all = f"{path_cell_line.stem}_{prefix_day}_"
        
        for path_file in list_all_files:
            pass
            print(f"{path_file.name} ==> {prefix_all + path_file.name}")
    
    
# for path_folder in list_folders:
#     pass
#     list_all_files = [path_file for path_file in path_folder.rglob("*") if path_file.is_file()]

#     prefix = f"H9_{path_folder.name.replace(' ', '_')}_"
#     for path_file in  list_all_files:
#         pass
#         filename_new = prefix + path_file.name
#         # os.rename(path_file, path_file.parent / filename_new )
# Learning_microstructure_code_and_data
This repository contains all data files and codes generated and used for the paper 'Learning Microstructure in Active matter'.

## The data is given for 3 cases: 

- # Passive Brownian particles:
  Contains all the raw simulation files for different packing fractions (as .tar.xz files), Percus-Yevick data for those specific packing fractions (as .tar.xz files), the Jupyter notebook containing the DNN model and all the visualization, and the .py file used to obtain the symbolic regression expression (.pkl file is the obtained model).
  
- # Active Brownian case with isotropy:
  Contains jupyter notebook with the DNN model definition and visualization of the results, the .py file with the pysr code, and pther necessary additional files (note: which also includes the train DNN model and pysr model). The raw simulation file is the same file that is used in the asymmetric case (we have only taken the angular average for this case. It is suggested to work with the prepared .csv dataset which is also given).
  
- # Active Brownian case with anisotropy:
  Contains the raw simulation files (as .tar.xz format), the trained DNN model (.pth format), and the output PySR model (as .pkl file). 


<!-- 'go to file -> edit -> rename 'file'-> 'my_folder/file' -->
> **Note:** This repo is under active development.

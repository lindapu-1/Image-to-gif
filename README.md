# Environment setup

For running the program successfuly, please install the packages in a new environment(recommended) or the existing environment. Please run the following commands in terminal:
1. clone the repository
   
        git clone https://github.com/lindapu-1/t2v.git
        cd t2v
   
3. create a new environment and install the dependencies
   
        conda create -n t2v python=3.10 
        conda activate t2v  
        pip install -r requirements.txt 


Then the environment is ready. for running the notebook, you may need to install the ipykernal additionally. 

# Overview

I mainly implement the video generation code on my_program.ipynb. Please refer to the notebook and run the cell one by one. Note that you need to select t2v as the kernal in jupyter notebook.

**sometimes the code cannot run successfully in notebooke. if so, you can try to run the code in terminal by the following command:**  
        `python scripts/text-to-image.py`  
        `python scripts/text-to-video.py`






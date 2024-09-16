# Overview

**Pu Ruiling  
for STAT4766 project**


I mainly implement the video generation code on **my_program.ipynb**. Please refer to the notebook and run the cell for video generation. Before running the cell, please make sure the environment is set up correctly.



# Environment setup

For running the program successfuly, please install the packages in a new environment(recommended) or the existing environment. Please run the following commands in terminal:
1. clone the repository
   
        git clone https://github.com/lindapu-1/Image-to-gif.git
        cd Image-to-gif
   
3. create a new environment and install the dependencies
   
        conda create -n video python=3.10 
        conda activate video  
        pip install -r requirements.txt 


Then the environment is ready. For running the notebook, you may need to install the ipykernal additionally. 


Side-note:
**sometimes the code cannot run successfully in notebook. if so, you can try to run the code in terminal by the following command:**  

        python scripts/text-to-image.py  
        python -m scripts/text-to-video
        





#environment setup

For running the program successfuuly, we setup a new virtual environment for the whole responsitory. Please follow the steps below:
1. clone the repository
    git clone https://github.com/lindapu-1/Image-to-gif.git
    cd Image-to-gif
2. create a new environment and install the dependencie
    conda create -n video python=3.10
    conda activate video
    pip install -r requirements.txt
Then the environment is ready. for running the notebook, you need to install the ipykernal additionally. 

#Overview

I mainly implement the video generation code on my_program.ipynb. Please refer to the notebook and run the cell one by one. Note that you need to select t2v as the kernal in jupyter notebook.


**sometimes the code cannot run successfully in notebooke. if so, you can try to run the code in terminal by the following command:**

    python scripts/text-to-image.py
    python scripts/text-to-video.py

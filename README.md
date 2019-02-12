# Pytorch classifier setup
## Setup using a new virtual environment
Virtual environments in python are useful when you have multiple projects that each require different versions of the same packages. If you create a new virtual environment for a project, installations of packages in this environment will not be seen by other projects. In my case this was useful for installing multiple versions of the torchvision package. 

1. Create a new python virtual environment: 
   ```
   $ python3.7 -m venv pytorch_xai
   ```

2. Activate the virtual environment. (new packages will only be installed in this environment)
   ```
   $ source pytorch_xai/bin/activate
   ```

3. Follow the [pip installation guide of pytorch](https://pytorch.org/). Make sure you don’t install torchvision yet. 

   I installed `Stable -> Linux -> Pip -> python 3.7 -> No CUDA` which came down to 
   ```
   $ pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp37-cp37m-linux_x86_64.whl
   ```
   Only execute the first pip command to install pytorch, do not execute the torchvision installation command.

4. Install torchvision from source. (The pip version does not include the VOC dataset.)  
   ```
   $ pip install six pillow numpy
   $ git clone https://github.com/pytorch/vision.git
   $ cd vision
   $ python setup.py install
   ```

5. Install other useful libraries: 
   ```
   $ pip install matplotlib requests
   ```

6. Create a new PyCharm project from the explainable_ai repository. 

7. To execute the project in the virtual environment, go to `Settings -> Project: explainable_ai -> Project Interpreter`. 
Select `Python 3.7 (pytorch_xai)` as project interpreter. If it is not in the dropdown list, select `Show all -> +` 
to add a new one. A new window will popup. Select `existing environment` and enter the directory of the virtual 
environment: `~/pytorch_xai/bin/python`

8. Run `main.py`. During the first run, the VOC dataset and the ImageNet class index will be downloaded. This may take 
some time. 

# Setup without virtual environment
1. Follow the [pip installation guide of pytorch](https://pytorch.org/). Make sure you don’t install torchvision yet.

   I installed `Stable -> Linux -> Pip -> python 3.7 -> No CUDA` which came down to 
   ```
   $ pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp37-cp37m-linux_x86_64.whl
   ```
   Only execute the first pip command to install pytorch, do not execute the torchvision installation command.

2. Install torchvision from source. (The pip version does not include the VOC dataset.) 
   ```
   $ pip install six pillow numpy
   $ git clone https://github.com/pytorch/vision.git
   $ cd vision
   $ python setup.py install
   ```

3. Install other useful libraries: 
   ```
   $ pip install matplotlib requests
   ```

4. Create a new PyCharm project from the explainable_ai repository. 

5. Run `main.py`. During the first run, the VOC dataset and the ImageNet class index will be downloaded. This may take 
some time. 
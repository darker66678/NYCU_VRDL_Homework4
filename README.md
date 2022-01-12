# NYCU_VRDL_Homework4
This is Homework 4 of VRDL about Super resolution 
## Architecture
|```data``` put data in here

|```ISR``` this is reference github [ISR](https://github.com/idealo/image-super-resolution)

|```KAIR``` this is reference github [KAIR](https://github.com/cszn/KAIR)

|```divide_img.py``` image augmentation for training data
## Requirement
Run ```pip install -r requirements.txt```, install all the dependencies

I use Python 3.7, pytorch 1.10.0(CUDA 11.3), maybe you need to install in [Pytorch](https://pytorch.org/get-started/locally/)
## Data preprocessing
Download [data](https://drive.google.com/file/d/1GL_Rh1N-WjrvF_-YOKOyvq0zrV6TF4hb/view?usp=sharing)
and Put data in ```data```
## Training
Run ```sh KAIR/model.sh train``` for starting train

(check the data path is correct in ```KAIR/options/train_swinir_sr_classical.json```)
## Inference
Run ```sh KAIR/model.sh test [your_model.pth]``` 

Inference images will be save in ```KAIR/results```
## My model
You can download my [SwinIR model](https://drive.google.com/file/d/1vABk1ywzSWGuRGrzjRbP-JKzWdfXDyqF/view?usp=sharing) and start to infer
## Results
|Model|PSNR|
|---|--|
|RDN-C8-D6|26.93 
|RDN-C4-D12|26.52
|ESRGAN-C8-D6|26.25
|ESRGAN-C4-D6|25.81
|**SwinIR**|**27.81**
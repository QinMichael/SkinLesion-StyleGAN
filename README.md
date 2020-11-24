SkinLesion-StyleGAN
======================
### This is a GAN-based image synthesis method for skin lesions
The skin lesion style-based GANs is proposed according to the basic architecture of **style-based GANs** ([[paper]](https://arxiv.org/abs/1812.04948)). The proposed model modifies the structure of style control and noise input in the original generator, adjusts both the generator and discriminator to efficiently synthesize high-quality skin lesion images. The code is derived from the article [**A GAN-based image synthesis method for skin lesion classification**](https://doi.org/10.1016/j.cmpb.2020.105568) (Qin et al., 2020).
![ISIC_Generated](https://github.com/QinMichael/SkinLesion-StyleGAN/blob/main/Results/SL-StyleGAN_generated.jpg)

Description
------------------
This project provides a readily comprehensible implementation of modified StyleGAN for skin lesion images synthesis. The main modifications are:
* The structures of **AdaIN** and **random noise** in `synthesis network g` of each level are adjusted. The application of noise to the constant input is removed without observable drawbacks and the first synthesis network block contains only one convolution layer.
* The **mixing regularization** is left out. Thus, there is only one `latent code z` being utilized in the modified architecture.
* In the case of the relatively low computer computing power and small amount of training data, the architectures of generator and discriminator are redesigned, omitting **progressive growing process** of the networks.

Dataset
------------------
The dataset is provided by [**International Skin Imaging Collaboration (ISIC) 2018 classification challenge**](https://arxiv.org/abs/1902.03368). It consists of **10,015** dermatoscopic images of skin lesions.  **Seven diagnostic categories** are defined in this dataset.

Category | Description | Number of Samples | Proportion
-------- | ----------- |----------------- | --------
AKIEC | Actinic Keratoses and Intraepithelial Carcinoma | 327 | 3.27%
BCC | Basal cell carcinoma | 514 | 5.13%
BKL | Benign keratosis | 1099 | 10.97%
DF | Dermatofibroma | 115 | 1.15%
NV | Melanocytic nevi | 6705 | 66.95%
MEL | Melanoma | 1113 | 11.11%
VASC | Vascular skin lesions | 142 | 1.42%

Getting started
------------------
### Prerequisites
* NVIDIA GPU + CUDA CuDNN
* Python 3
* Keras + Tensorflow backend

### Training
#### Prepare your data
* Create  `'./Datasets/' ` in the main directory and put your data in it. 
* You need to check the code files `data_loader.py` and `train.py` to set the right dataset path.

#### Running
* `SL_StyleGAN.py` is the model file, and you can start your training by running the `train.py`.
* Create `'./SavedModels/'` in the main directory to save the models and you can restart with these model files.
Example:
![ISIC_MEL](https://github.com/QinMichael/SkinLesion-StyleGAN/blob/main/Results/ISIC_mel.png)

Updates
------------------
`2020.11.24`: Initial code release

Citation
------------------
If you want to use this code for your own research, please cite the following article:
``` 
@article{SkinLesion-StyleGAN,
  title   = {A GAN-based image synthesis method for skin lesion classification},
  author  = {Zhiwei Qin, Zhao Liu*, Ping Zhu*, Yongbo Xue},
  journal = {Computer Methods and Programs in Biomedicine},
  year    = {2020},
  doi     = {https://doi.org/10.1016/j.cmpb.2020.105568}
}
```

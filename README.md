SkinLesion-StyleGAN
======================
#### This is a GAN-based image synthesis method for skin lesions
The skin lesion style-based GANs is proposed according to the basic architecture of **style-based GANs** ([[paper]](https://arxiv.org/abs/1812.04948)). The proposed model modifies the structure of style control and noise input in the original generator, adjusts both the generator and discriminator to efficiently synthesize high-quality skin lesion images. The code is derived from the article [**A GAN-based image synthesis method for skin lesion classification**](https://doi.org/10.1016/j.cmpb.2020.105568) (Qin et al., 2020).
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

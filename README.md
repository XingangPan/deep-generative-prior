## Deep Generative Prior (DGP)

### Paper

Xingang Pan, Xiaohang Zhan, Bo Dai, Dahua Lin, Chen Change Loy, Ping Luo, "[Exploiting Deep Generative Prior for Versatile Image Restoration and Manipulation](https://arxiv.org/abs/2003.13659)"([project page](https://xingangpan.github.io/projects/DGP.html))

### Demos

DGP exploits the image prior of an off-the-shelf GAN for various image restoration and manipulation.

**Image restoration**:

<p align="center">
    <img src="data/restoration.gif", width="600">
</p>

**Image manipulation**:

<p align="center">
    <img src="data/manipulation.gif", width="500">
</p>

### Requirements

* python>=3.6
* pytorch>=1.0.1
* others

    ```sh
    pip install -r requirements.txt
    ```

### Get Started

Before start, please download the pretrained BigGAN at [Google drive](https://drive.google.com/drive/folders/1buQ2BtbnUhkh4PEPXOgdPuVo2iRK7gvI?usp=sharing) or [Baidu cloud](https://pan.baidu.com/s/10GKkWt7kSClvhnEGQU4ckA)(password: uqtw), and put them to `pretrained` folder.

Example1: run image colorization example:
    
    sh experiments/examples/run_colorization.sh   

The results will be saved in `experiments/examples/images` and `experiments/examples/image_sheet`.

Example2: process images with an image list:
    
    sh experiments/examples/run_inpainting_list.sh   

Example3: evaluate on 1k ImageNet validation images via distributed training based on [slurm](https://slurm.schedmd.com/):

    # need to specifiy the root path of imagenet validate set in --root_dir
    sh experiments/imagenet1k_128/colorization/train_slurm.sh   

Note:  
\- BigGAN needs a class condition as input. If no class condition is provided, it would be chosen from a set of random samples.  
\- The hyperparameters provided may not be optimal, feel free to tune them.  

### Acknowledgement

The code of BigGAN is borrowed from [https://github.com/ajbrock/BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch).

### Citation

```  
@article{pan2020dgp,
  author = {Pan, Xingang and Zhan, Xiaohang and Dai, Bo and Lin, Dahua and Loy, Chen Change and Luo, Ping},
  title = {Exploiting Deep Generative Prior for Versatile Image Restoration and Manipulation},
  journal = {arxiv preprint arxiv:2003.13659},
  year = {2020}
}
```  

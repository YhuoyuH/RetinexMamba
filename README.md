# RetinexMamba

This is the official code for "RetinexMamba: Retinex-based Mamba for Low-light Image Enhancement." For the paper, please see {[Arxiv]()}

## Abstract

In the field of low-light image enhancement, traditional Retinex methods and deep learning-based techniques such as Retinexformer each display unique advantages and limitations. Traditional Retinex methods, by simulating the human eye's perception of brightness and color, decompose images into illumination and reflection components. However, they often fail to adequately consider noise and detail loss introduced by low lighting. Retinexformer, while optimizing illumination estimation through deep learning, suffers from high computational complexity and requires multi-stage training. Addressing these challenges, this paper introduces the RetinexMamba architecture, which not only inherits the physical intuitiveness of traditional Retinex methods but also incorporates the deep learning framework of Retinexformer, further integrating the efficient computational capabilities of State Space Models (SSMs). RetinexMamba enhances image illumination and repairs damages during the enhancement process through innovative illumination estimators and damage repairers, while substituting the IG-MSA in Retinexformer with a Fused-Attention mechanism to enhance model interpretability. Experimental results on the LOL dataset demonstrate that RetinexMamba surpasses existing deep learning methods based on Retinex theory in both quantitative and qualitative metrics, proving its effectiveness and superiority in low-light image enhancement tasks.

For the Chinese link, please see[CN](https://github.com/YhuoyuH/RetinexMamba-CN)



### 1.Download the project.

Please run the following command to ensure that you deploy our project locally.

```python
git clone https://github.com/YhuoyuH/RetinexMamba.git
```

### 2.Create environment.

Please note that since the "causal_conv1d" package is only available on Linux systems, ensure that your operating environment is Linux.

### 2.1Create Conda environment.

To prevent any discrepancies between your environment and ours, we recommend that you choose the same virtual environment as us. You can directly install the environment we have packaged for you, or choose to follow our tutorial to install it step by step.

#### 2.1.1Directly use the compressed package.

Please download the RetinexMamba.tar.zip compressed package from [Baidu Netdisk]() and use the following command to unzip it.

```python
cd /path/to/folder/containing/RetinexMamba.tar.zip
tar -xzf RetinexMamba.tar.zip -C /your/anaconda/envs/directory/
```

If you use this command, you do not need to follow the subsequent steps in "Create Environment." You can directly proceed to Part 3.

#### 2.1.2Use package installation.

```python
conda create -n RetinexMamba python=3.8
conda activate RetinexMamba
```

### 2.2Install dependencies.

```python
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy
pip install opencv-python joblib natsort tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips
```

Please note that you may encounter network issues during the installation of "causal_conv1d" and "mamba_ssm," which could cause the process to continuously hang at "Building wheel for mamba ssm (setup.py)." Therefore, please download the .whl files from [Baidu Netdisk]() and copy them locally. Then, run the following command for manual installation.

```python
pip install path/filename.whl
```

#### 2.3Install BasicSR.

```python
cd /RetinexMamba/
python setup.py develop --no_cuda_ext
```

### 3.Prepare the dataset.

Please download the dataset from [Baidu Netdisk]() and place the data file in the RetinexMamba folder.

The final placement should be as follows:


```
|--RetinexMamba  	
|  	 |--data   
|    |    |--LOLv1
|    |    |    |--Train
|    |    |    |    |--input
|    |    |    |    |--target
|    |    |    |--Test
|    |    |    |    |--input
|    |    |    |    |--target
|    |    |--LOLv2
|    |    |    |--Real_captured
|    |    |    |    |--Train
|    |    |    |    |    |--Low
|    |    |    |    |    |--Normal
|    |    |    |    |--Test
|    |    |    |    |    |--Low
|    |    |    |    |    |--Normal
|    |    |    |--Synthetic
|    |    |    |    |--Train
|    |    |    |    |    |--Low
|    |    |    |    |    |--Normal
|    |    |    |    |--Test
|    |    |    |    |    |--Low
|    |    |    |    |    |--Normal
```

### 4.Test

Please ensure that the pretrained_weights folder contains our pre-trained weights. If your weight files are missing, please download them from [Baidu Netdisk]().

```
# activate the environment
conda activate RetinexMamba

# LOL-v1
python3 Enhancement/test_from_dataset.py --opt Options/RetinexMamba_LOL_v1.yml --weights pretrained_weights/LOL_v1.pth --dataset LOL_v1

# LOL-v2-real
python3 Enhancement/test_from_dataset.py --opt Options/RetinexMamba_LOL_v2_real.yml --weights pretrained_weights/LOL_v2_real.pth --dataset LOL_v2_real

# LOL-v2-synthetic
python3 Enhancement/test_from_dataset.py --opt Options/RetinexMamba_LOL_v2_synthetic.yml --weights pretrained_weights/LOL_v2_synthetic.pth --dataset LOL_v2_synthetic
```

### 5.Model parameters and FLOPS evaluation.

If you want to view the model's parameter count and floating points, please directly run `ReinexMamba_arch` located in `basicsr/models/archs`. If the importation above the code is:

```
from .vmamba_arch import SS2D
from .fuse_block_arch import TransformerBlock
```

change it into:

```
from vmamba_arch import SS2D
from fuse_block_arch import TransformerBlock
```

If you want to check the parameter count of the ablation study models, please move them from the `Ablation_Model` folder to the `archs` folder and repeat the above operation.

### 6.Train

Please ensure that you have fully completed the environment setup and can correctly infer the parameters and floating points.

```
# activate the enviroment
conda activate RetinexMamba

# LOL-v1
python3 basicsr/train.py --opt Options/RetinexMamba_LOL_v1.yml

# LOL-v2-real
python3 basicsr/train.py --opt Options/RetinexMamba_LOL_v2_real.yml

# LOL-v2-synthetic
python3 basicsr/train.py --opt Options/RetinexMamba_LOL_v2_synthetic.yml
```


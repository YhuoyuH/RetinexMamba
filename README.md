# RetinexMamba

这是"RetinexMamba:Retinex-based Mamba for Low-light Image Enhancement"的官方代码。论文请见{[Arxiv]()}

## 摘要

In the field of low-light image enhancement, traditional Retinex methods and deep learning-based techniques such as Retinexformer each display unique advantages and limitations. Traditional Retinex methods, by simulating the human eye's perception of brightness and color, decompose images into illumination and reflection components. However, they often fail to adequately consider noise and detail loss introduced by low lighting. Retinexformer, while optimizing illumination estimation through deep learning, suffers from high computational complexity and requires multi-stage training. Addressing these challenges, this paper introduces the RetinexMamba architecture, which not only inherits the physical intuitiveness of traditional Retinex methods but also incorporates the deep learning framework of Retinexformer, further integrating the efficient computational capabilities of State Space Models (SSMs). RetinexMamba enhances image illumination and repairs damages during the enhancement process through innovative illumination estimators and damage repairers, while substituting the IG-MSA in Retinexformer with a Fused-Attention mechanism to enhance model interpretability. Experimental results on the LOL dataset demonstrate that RetinexMamba surpasses existing deep learning methods based on Retinex theory in both quantitative and qualitative metrics, proving its effectiveness and superiority in low-light image enhancement tasks.

中文链接请详见[CN]()



### 1.下载项目

请您运行下面的命令以确保您将我们的项目部署到本地

```python
git clone https://github.com/YhuoyuH/RetinexMamba.git
```

### 2.创建环境

注意，由于后续的"causal_conv1d"包仅存在于Linux系统上，因此请确保您所使用的操作环境是Linux

### 2.1创建Conda环境

为了防止您和我们所使用的环境版本不一致的问题，我们建议您选择和我们相同的虚拟环境，您可直接安装我们已经为您打包好的环境，也可选择跟随我们的教程一步步安装。

#### 2.1.1直接使用压缩包

请将[百度网盘]()的RetinexMamba.tar.zip压缩包下载下来，并利用下面的命令进行解压

```python
cd /RetinexMamba.tar.zip所在的文件夹/
tar -xzf RetinexMamba.tar.zip -C /你的anaconda的env文件路径/
```

若您采取这条命令，则不需要进行 "Create Environment"中的后续操作，直接跳到第二部分即可

#### 2.1.2利用包安装

```python
conda create -n RetinexMamba python=3.8
conda activate RetinexMamba
```

### 2.2安装依赖项

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

注意，您可能在安装"causal_conv1d"以及"mamba_ssm"会出现网络异常而导致一直卡在"Building wheel for mamba ssm(setup.py)"的问题。因此请您下载[百度网盘]()中的.whl文件拷贝在本地，并运行下面的命令进行手动安装

```python
pip install 路径/文件名.whl
```

#### 2.3安装BasicSR

```python
cd /RetinexMamba/
python setup.py develop --no_cuda_ext
```

### 3.准备数据集

请从[百度网盘]()中下载数据集，并将data文件放置在RetinexMamba的文件夹下。

最终放置的结果如下：


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

### 4.测试

请确保pretrained_weights文件夹下有我们预训练的权重，如果您的权重文件丢失，请从[百度网盘]()下载。

```
# activate the environment
conda activate RetinexMamba

# LOL-v1
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v1.yml --weights pretrained_weights/LOL_v1.pth --dataset LOL_v1

# LOL-v2-real
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_real.yml --weights pretrained_weights/LOL_v2_real.pth --dataset LOL_v2_real

# LOL-v2-synthetic
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_synthetic.yml --weights pretrained_weights/LOL_v2_synthetic.pth --dataset LOL_v2_synthetic
```

### 5.模型参数以及浮点数评估

若您想要查看模型的参数量以及浮点数，请您直接运行basicsr/models/archs中的ReinexMamba_arch即可，请注意将代码上方的

```
from .vmamba_arch import SS2D
from .fuse_block_arch import TransformerBlock
```

改成

```
from vmamba_arch import SS2D
from fuse_block_arch import TransformerBlock
```

若您想查看消融实验模型的参数量，请将他们从Ablation_Model中移到archs文件夹下并重复上述操作即可

### 6.训练

请确保您已完整完成环境配置并且可以正常的推理出参数和浮点数

```
# activate the enviroment
conda activate RetinexMamba

# LOL-v1
python3 basicsr/train.py --opt Options/RetinexFormer_LOL_v1.yml

# LOL-v2-real
python3 basicsr/train.py --opt Options/RetinexFormer_LOL_v2_real.yml

# LOL-v2-synthetic
python3 basicsr/train.py --opt Options/RetinexFormer_LOL_v2_synthetic.yml
```


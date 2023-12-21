# 2023/10/31 整理 生成noise（perturbation）的代码
1. Phantom Sponge 代码（依赖于BDD100k-val的数据，生成UAP，可调控lambda超参）
    a. Git设置代理（即clash用的port） https://blog.csdn.net/panc_guizaijianchi/article/details/122968009
    b. 创建适用与跑这个文件夹内的conda环境 （delete，还是使用py39base）
        i. py39; cuda 10.2; 
        ii. `conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch`
        iii. pip 安装 albumentations； pickleshare；pillow （maybe 版本有偏差）
2. 
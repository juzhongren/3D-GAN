# 3D-GAN
3维生成对抗网络

网络模型在ecbm6040中的model中，另一个metric是评价指标包含PSNR、MSE等
原始代码写的是wgan，后来我在这个基础上改成了WGAN-GP

MySequentialSampler.py是按照自己设置的indeices进行测试的
read_datalist.py是一个读取numpy保存的序号的

数据集按./data/data1/这样设置，image和label是按..._in.nii和..._out.nii设置的

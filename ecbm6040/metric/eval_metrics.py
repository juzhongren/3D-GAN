# import skimage.measure as measure
import torch
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import normalized_root_mse as compare_nrmse
from skimage.metrics import mean_squared_error as compare_mse

# 常见的客观评价标准
# 常见的评价量主要有：
#
# MSE：MSE是mean square error的缩写，也就是均方误差。这是最简单粗暴的一个量。它首先计算失真图片与原图每在个像素上差的平方，然后取均值。
# PSNR：这个量全称为峰值信噪比（peak signal-to-noise ration），这个量很容易计算（粗略地说是对图像上的最大值和MSE进行比较），而且有明确的物理意义，优化也方便，在很多地方都使用。

# SSIM是感知模型，即更符合人眼的直观感受。
# SSIM的取值范围[-1, 1], 具有对成性，边界性，唯一最大性（当且仅当x=y时SSIM=1），是一种距离公式。
# SSIM 主要考量图片的三个关键特征：亮度（Luminance）, 对比度（Contrast）, 结构 (Structure)
# 结构相似性（structual similarity, SSIM
# 结构相似性指数（Structural Similarity Index measure，SSIM）用作度量两个给定图像之间的相似性。

# 结构相似性SSIM
# 结构相似度指数从图像组成的角度将结构信息定义为独立于亮度、对比度的，反映场景中物体结构的属性，
# 并将失真建模为亮度、对比度和结构三个不同因素的组合。用均值作为亮度的估计，标准差作为对比度的估计，协方差作为结构相似程度的度量。

#计算均方误差
def mse(img_true, img_test):
    img_true = img_true.float()
    img_true = img_true.numpy()

    img_test = img_test.numpy()
    # print("输出真实数据img_true")
    # print(img_true.shape)
    img_true = img_true.squeeze()
    img_test = img_test.squeeze()
    # print(img_true.shape)
    mse = []



    # print("img_true.shape[0]")
    # print(img_true.shape[0])
    # print(img_true.shape)
    # print(compare_ssim(img_true, img_test,multichannel=True))
    #开启multichannel=True这个，
    for i in range(img_true.shape[0]):
        # print(i)
        mse = np.append(mse, compare_mse(img_true[i], img_test[i]))
    # print(ssim)
    return mse
def ssim(img_true, img_test):
    '''
    This function input two batches of true images and the fake images. Use skimage.measure.compare_ssim function to compute the mean structural similarity index between two images.
    该函数输入两批真图像和假图像。 使用 skimage.measure.compare_ssim 函数计算两个图像之间的平均结构相似性指数。
    (Input) img_true: the input should be derived from dataloader, it's in torch.ShortTensor (B,z,x,y). By default, it should be HR images.
                        输入应该来自数据加载器，它在 torch.ShortTensor (B,z,x,y) 中。 默认情况下，它应该是 HR 图像。
    (Input) img_test: the input should be derived from depatching function, it's in torch.float (B,z,x,y). By default, it should be SR images.
                        输入应该来自 depatching 函数，它在 torch.float (B,z,x,y) 中。 默认情况下，它应该是 SR 图像。
    (Output) ssim: an ndarray with length (B,1), which contains the ssim value for each image in the batch.
                    一个长度为 (B,1) 的 ndarray，其中包含批次中每个图像的 ssim 值。
    '''
    #batch_size.channel,64,64,64
    #tensor转numpy
    img_true = img_true.float()
    img_true = img_true.numpy()

    img_test = img_test.numpy()
    # print("输出真实数据img_true")
    # print(img_true.shape)
    img_true = img_true.squeeze()
    img_test = img_test.squeeze()
    # print(img_true.shape)
    ssim = []
    # print("img_true.shape[0]")
    # print(img_true.shape[0])
    # print(img_true.shape)
    # print(compare_ssim(img_true, img_test,multichannel=True))
    #开启multichannel=True这个，
    for i in range(img_true.shape[0]):
        # print(i)
        data_range = np.max(img_true[i])
        # data_range = 16.0
        ssim = np.append(ssim, compare_ssim(img_true[i], img_test[i],data_range = data_range))
    # print(ssim)
    return ssim
    #单个图片
    # #类型都是【1,1,64,64,64,64】
    # # 需要压缩一个维度，变成【1,64,64,64,64】
    # img_true = img_true.squeeze()
    # img_test = img_test.squeeze()
    # return compare_ssim(img_true, img_test, multichannel=True)


# PSNR（Peak Signal to Noise Ratio），峰值信噪比，即峰值信号的能量与噪声的平均能量之比，通常表示的时候取log 变成分贝（dB），
# 由于 MSE 为真实图像与含噪图像之差的能量均值，而两者的差即为噪声，因此 PSNR 即峰值信号能量与 MSE之比。
def psnr(img_true, img_test):
    '''
    This function input two batches of true images and the fake images. Use skimage.measure.compare_psnr function to compute the peak signal to noise ratio (PSNR) between two images.
    该函数输入两批真图像和假图像。 使用 skimage.measure.compare_psnr 函数计算两个图像之间的峰值信噪比 (PSNR)。
    (Input) img_true: the input should be derived from dataloader, it's in torch.ShortTensor (B,z,x,y). By default, it should be HR images.
    (Input) img_test: the input should be derived from depatching function, it's in torch.float (B,z,x,y). By default, it should be SR images.
    (Output) psnr: an ndarray with length (B,1), which contains the psnr value for each image in the batch.
    '''
    img_true = img_true.float()
    img_true = img_true.numpy()

    img_test = img_test.numpy()
    #类型都是【1,1,64,64,64,64】
    # 需要压缩一个维度，变成【1,64,64,64,64】
    img_true = img_true.squeeze()
    img_test = img_test.squeeze()
    psnr = []
    for i in range(img_true.shape[0]):
        data_max = img_true[i].max()
        err = compare_mse(img_true[i], img_test[i])
        data_range = np.max(img_true[i])
        # data_range = 16.0
        psnr_result = 10 * np.log10((data_range ** 2) / err)
        psnr = np.append(psnr, psnr_result)
        # psnr = np.append(psnr, compare_psnr(img_true[i], img_test[i]))
    # print(psnr)
    return psnr

    # return compare_psnr(img_true, img_test)
# 均方根误差(Root Mean Square Error)是一个翻译空间细节信息的评价指标
# 归一化均方根误差（normalized root mean square error）就是将RMSE的值变成(0,1)之间。
def nrmse(img_true, img_test):
    '''
    This function input two batches of true images and the fake images. Use skimage.measure.compare_nrmse function to compute the normalized root mean-squared error (NRMSE) between two images.
    (Input) img_true: the input should be derived from dataloader, it's in torch.ShortTensor (B,z,x,y). By default, it should be HR images.
    (Input) img_test: the input should be derived from depatching function, it's in torch.float (B,z,x,y). By default, it should be SR images.
    (Output) nrmse: an ndarray with length (B,1), which contains the psnr value for each image in the batch.
    '''
    img_true = img_true.float()
    img_true = img_true.numpy()

    img_test = img_test.numpy()
    #类型都是【1,1,64,64,64,64】
    # 需要压缩一个维度，变成【1,64,64,64,64】
    img_true = img_true.squeeze()
    img_test = img_test.squeeze()
    # return compare_nrmse(img_true, img_test)
    nrmse = []
    for i in range(img_true.shape[0]):
        nrmse = np.append(nrmse, compare_nrmse(img_true[i], img_test[i]))
    # print(nrmse)
    return nrmse

if __name__ == '__main__':
    #测试评价函数
    #输入是tensor
    x = 64
    y = 64
    z = 64
    # 创建形式有两种
    # 1 随机数形式
    data1 = np.random.random((8,1,x, y, z))
    data2 = np.random.random((8,1,x, y, z))
    print(data1.shape)
    data1 = torch.tensor(data1)
    data2 = torch.tensor(data2)
    ssim(data1,data2)
    psnr(data1, data2)
    nrmse(data1, data2)
    # 2 0或1形式
    # np.ones((x, y, z))
    # np.zeros((x, y, z))
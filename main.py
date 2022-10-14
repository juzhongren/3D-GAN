from __future__ import print_function

import os
import random
import math
import numpy as np
import torch.nn as nn
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

from data.code.ecbm6040.model.MySequentialSampler import MySequentialSampler


def main1(datasetname,index):
    print(datasetname)
    if not os.path.exists(datasetname +"test{}".format(index)):
        print(datasetname +"test{}".format(index))
        os.makedirs(datasetname +"test{}".format(index))
    print(datasetname + "test{}".format(index))
    #设置随机数种子
    # Set random seem for reproducibility
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Number of workers for dataloader
    workers = 4
    # Batch size. It controls the number of samples once download
    batch_size = 4
    # Learning rate for mDCSRN (G) pre-training optimizers (in paper: 1e-4)
    lr_pre = 1e-4
    # Learning rate for optimizers (in paper: 5e-6)
    lr = 1e-4
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # set GPU device
    os.environ['CUDA_VISIBLE_DEVICES']='0,1'
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print("gpu数量:", end="")
    print(torch.cuda.device_count())
    # Set percentage of data spliting
    train_split= 0.8
    validate_split = 0.2
    # test_split = 0.2

    # Set shuffle and stablize random_seed
    shuffle_dataset = True
    random_seed= 999

    from niidataset import NiiDataset
    dataset = NiiDataset(datasetname+"/data/")
    dataset_size = len(dataset)
    print("总共"+str(dataset_size)+"对数据")
    from read_datalist import read_list
    train_val_indices = read_list(datasetname+"train_{}.txt".format(index))
    print(train_val_indices)
    test_indices = read_list(datasetname+"test_{}.txt".format(index))
    print(test_indices)
    # 读取训练集
    # from read_datalist import read_list
    # train_indices = read_list("train_{}.txt".format(index))
    # get indices for spliting train, valiadate, evaluate, test sets
    train_size = math.ceil(train_split * len(train_val_indices))
    validate_size = int(len(train_val_indices) - train_size)
    # evaluate_size = int(evaluate_split * dataset_size)
    test_size = len(test_indices)

    # #把序号打乱
    # indices = list(range(dataset_size))
    # print("打乱前的{}".format(indices))
    # if shuffle_dataset :
    #     np.random.seed(random_seed)
    #     np.random.shuffle(indices)
    # print("打乱后的{}".format(indices))
    #训练
    train_indices = train_val_indices[:train_size]
    # To reduce computational cost in validation period, you can intentionally decrease the validation size.
    #测试
    # loc_val = train_size + validate_size
    val_indices = train_val_indices[train_size:]
    #验证
    test_indices = test_indices
    num_train = len(train_indices)
    num_val = len(val_indices)
    num_test = len(test_indices)
    print('training set number:{}'.format(len(train_indices)))
    print('validation set number:{}'.format(len(val_indices)))
    print('test set number:{}'.format(len(test_indices)))

    # The following cell create dataloaders used for pretraining.
    # Use for pre-training (we don't want too much validation)

    # test_sampler = SubsetRandomSampler(test_indices)
    test_sampler = MySequentialSampler(test_indices)
    # Batch size. It controls the number of samples once download
    def chunks(arr, m):
        '''
        This function split the list into m fold.
        '''
        n = int(np.floor(len(arr) / float(m)))#np.floor()返回不大于输入参数的最大整数。（向下取整）
        arr_split = [arr[i:i + n] for i in range(0, len(arr), n)]
        return arr_split
    #
    # Split indices
    train_indices_split = chunks(train_indices, 10)
    val_indices_split = chunks(val_indices, 10)
    #
    dataloaders={'train':[], 'val':[]}
    dataset_sizes={'train':[], 'val':[]}
    #
    for i in range(10):
        train_sampler = SubsetRandomSampler(train_indices_split[i])
        valid_sampler = SubsetRandomSampler(val_indices_split[i])
        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batch_size,
                                            sampler=train_sampler,
                                            shuffle=False,
                                            num_workers=workers)
        validation_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batch_size,
                                            sampler=valid_sampler,
                                            shuffle=False,
                                            num_workers=workers)
        dataloaders['train'].append(train_loader)
        dataloaders['val'].append(validation_loader)

    test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=1,
                                            sampler=test_sampler,
                                            shuffle=False,
                                            num_workers=workers)

    from ecbm6040.model.mDCSRN_WGAN import Generator

    # Create the generator
    # netG = Generator(ngpu).cuda(device)
    netG = Generator(ngpu).cuda(device)
    # Print the model
    # print(netG)
    # print('# generator parameters:', sum(param.numel() for param in netG.parameters()))

    # Initialize Loss functions
    supervised_criterion = nn.L1Loss()
    # supervised_criterion = nn.MSELoss()


    # Use for WGAN-training (we want to have more frequent validation)

    # #
    from ecbm6040.model.mDCSRN_WGAN import Discriminator
    # Create the Discriminator
    netD = Discriminator(ngpu).cuda(device)
    # netD = Discriminator(ngpu)
    # Print the model
    # print(netD)
    # print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    from WGAN_GP import WGAN_GP
    # We move wasserstein Loss into the training function.
    wgan_gp = WGAN_GP(netG = netG,netD = netD,supervised_criterion = supervised_criterion,device=device,ngpu =ngpu,lr=lr)
    # #训练gan网络
    #
    print("序号为：{}".format(index))
    # model_G, model_D = wgan_gp.training_epoch(dataloaders=dataloaders,max_epochs= 200,num_epochs_pre=0,pretrainedG=" ",pretrainedD=" ",datasetname= datasetname,index = index)

    from predictWithGanG import predict_G
    #
    predict_G(test_loader,ngpu,device,datasetname,index,test_indices,test_sampler)


def main2(datasetname,index):
    print(datasetname)
    if not os.path.exists(datasetname + "test{}".format(index)):
        print(datasetname + "test{}".format(index))
        os.makedirs(datasetname + "test{}".format(index))
    print(datasetname + "test{}".format(index))
    # 设置随机数种子
    # Set random seem for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Number of workers for dataloader
    workers = 4
    # Batch size. It controls the number of samples once download
    batch_size = 4
    # Learning rate for mDCSRN (G) pre-training optimizers (in paper: 1e-4)
    lr_pre = 1e-4
    # Learning rate for optimizers (in paper: 5e-6)
    lr = 1e-4
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print("gpu数量:", end="")
    print(torch.cuda.device_count())
    # Set percentage of data spliting
    train_split = 0.8
    validate_split = 0.2
    # test_split = 0.2

    # Set shuffle and stablize random_seed
    shuffle_dataset = True
    random_seed = 999

    from niidataset import NiiDataset
    dataset = NiiDataset(datasetname + "/data/")
    dataset_size = len(dataset)
    print("总共" + str(dataset_size) + "对数据")
    from read_datalist import read_list
    train_val_indices = read_list(datasetname + "train_{}.txt".format(index))
    print(train_val_indices)
    test_indices = read_list(datasetname + "test_{}.txt".format(index))
    print(test_indices)
    # 读取训练集
    # from read_datalist import read_list
    # train_indices = read_list("train_{}.txt".format(index))
    # get indices for spliting train, valiadate, evaluate, test sets
    train_size = math.ceil(train_split * len(train_val_indices))
    validate_size = int(len(train_val_indices) - train_size)
    # evaluate_size = int(evaluate_split * dataset_size)
    test_size = len(test_indices)

    # #把序号打乱
    # indices = list(range(dataset_size))
    # print("打乱前的{}".format(indices))
    # if shuffle_dataset :
    #     np.random.seed(random_seed)
    #     np.random.shuffle(indices)
    # print("打乱后的{}".format(indices))
    # 训练
    train_indices = train_val_indices[:train_size]
    # To reduce computational cost in validation period, you can intentionally decrease the validation size.
    # 测试
    # loc_val = train_size + validate_size
    val_indices = train_val_indices[train_size:]
    # 验证
    test_indices = test_indices
    print("test_indices的类型")
    print(type(test_indices))
    print(test_indices)
    print(test_indices.sort(reverse = False ))

    num_train = len(train_indices)
    num_val = len(val_indices)
    num_test = len(test_indices)
    print('training set number:{}'.format(len(train_indices)))
    print('validation set number:{}'.format(len(val_indices)))
    print('test set number:{}'.format(len(test_indices)))

    # The following cell create dataloaders used for pretraining.
    # Use for pre-training (we don't want too much validation)
    #zhelide
    # test_sampler = SubsetRandomSampler(test_indices)
    test_sampler = MySequentialSampler(test_indices)

    # Batch size. It controls the number of samples once download
    def chunks(arr, m):
        '''
        This function split the list into m fold.
        '''
        n = int(np.floor(len(arr) / float(m)))  # np.floor()返回不大于输入参数的最大整数。（向下取整）
        arr_split = [arr[i:i + n] for i in range(0, len(arr), n)]
        return arr_split

    #
    # Split indices
    train_indices_split = chunks(train_indices, 10)
    val_indices_split = chunks(val_indices, 10)
    #
    dataloaders = {'train': [], 'val': []}
    dataset_sizes = {'train': [], 'val': []}
    #
    for i in range(10):
        train_sampler = SubsetRandomSampler(train_indices_split[i])
        valid_sampler = SubsetRandomSampler(val_indices_split[i])
        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=batch_size,
                                                   sampler=train_sampler,
                                                   shuffle=False,
                                                   num_workers=workers)
        validation_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                        batch_size=batch_size,
                                                        sampler=valid_sampler,
                                                        shuffle=False,
                                                        num_workers=workers)
        dataloaders['train'].append(train_loader)
        dataloaders['val'].append(validation_loader)

    # for index1 in test_sampler:
    #     print("index: {}, data: {}".format(str(index1), str(test_indices[index1])))

    test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=1,
                                              sampler=test_sampler,
                                              shuffle=False,
                                              num_workers=0)

    # from ecbm6040.model.mDCSRN_WGAN2 import Generator
    #
    # # Create the generator
    # netG = Generator(ngpu).cuda(device)
    # # Print the model
    # print(netG)
    # print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    #
    # # Initialize Loss functions
    # # Supervised criterion
    # supervised_criterion = nn.L1Loss()
    # # We move wasserstein Loss into the training function.
    # from ecbm6040.model.mDCSRN_WGAN2 import Discriminator
    # # Create the Discriminator
    # netD = Discriminator(ngpu).cuda(device)
    #
    # # Print the model
    # print(netD)
    # print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    # from WGAN_GP2 import WGAN_GP
    # wgan_gp = WGAN_GP(netG, netD, supervised_criterion, device, ngpu, lr = lr)
    # #训练gan网络
    # # model_G, model_D = wgan_gp.training_epoch(dataloaders=dataloaders,max_epochs=200,num_epochs_pre=0,pretrainedG=" ",pretrainedD=" ",datasetname= datasetname,index = index)
    #

    # #测试代码
    from predictWithGanG2 import predict_G
    #
    print("测试集")
    print("index:{}".format(index))
    predict_G(test_loader,ngpu,device,datasetname,index,test_indices,test_sampler)


if __name__ == '__main__':
    import time

    start = time.time()
    # for i in range(1,2):
    #     main1("./data/data2/",i)

    for i in range(1,2):
        print("第{}轮".format(i))
        main2("./data/data2/",i)



    end = time.time()
    print(end - start)
    f = open("./time.txt", "w")
    value = end - start
    s = str(value)
    f.write(s)
    # 关闭打开的文件，必须关闭不然电脑能炸裂
    f.close()
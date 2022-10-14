from __future__ import print_function
import glob
import math
import os
import torch.utils.data
from torch.utils.data import SubsetRandomSampler

def inputdata(img):
    # b_size代表batchsize
    # 把数据变成二维的 ，batchsize*64;[batchsize,1,64,64,64]->[batchsize*64,1,64,64]
    b_size = img.size()[0]
    img = img.squeeze()
    img = img.view(b_size * 64, 64, 64)
    img = img.unsqueeze(1)
    return img


def outputdata(img):
    # 复原sr_patches
    b_size = img.size()[0]
    b_size = img.size()[0] / 64
    b_size = int(b_size)
    # print("b_size")
    # print(b_size)
    img = img.squeeze()
    img = img.view(b_size, 64, 64, 64)
    img = img.unsqueeze(1)
    return img

def predict_G(dataloader,ngpu,device,datasetname,index,test_indices,test_sampler):

    if not os.path.exists(datasetname +"test{}/test_2".format(index)):
        print(datasetname +"test{}/test_2".format(index))
        os.makedirs(datasetname +"test{}/test_2".format(index))
    print(datasetname + "test{}/test_2".format(index))

    #网络结构
    #定义生成器
    #整个网络，全部保存的
    # netG = torch.load("./models/pretrained_G_step20000")

    from ecbm6040.model.mDCSRN_WGAN2 import Generator
    # # Create the generator
    netG = Generator(ngpu).cuda(device)
    # # Print the model
    # print(netG)
    # print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    # netG.load_state_dict(torch.load("./models/final_model_G")).eval()
    # #可以判断device
    if device != "cpu":
        print("使用的是")
        print(device)
        print("使用GPU")
        # netG.load_state_dict(torch.load("./models2/models2/WGAN_G_step250000"))
        # netG.load_state_dict(torch.load("./models2/ngpu_pretrained_G_epoch20"))
        # netG.load_state_dict(torch.load("./models2/final_model_G"))
        netG.load_state_dict(torch.load(datasetname + "models/model2/model_{}/final_model_G".format(index)))
    else:
        print(device)
        # netG.load_state_dict(torch.load("./models/ngpu_param_pretrained_G_step250000", map_location = 'cpu'))
        # netG.load_state_dict(torch.load("./models/final_model_G"))
        netG.load_state_dict(torch.load(datasetname + "models/model2/model_{}/final_model_G".format(index)))

    # Set model to eval mode
    netG = netG.eval()

    nii_sample = glob.glob(datasetname+"data/*_in.nii" )[0]
    import nibabel as nib
    img_sample = nib.load(nii_sample)
    image_folder = datasetname+"test{}//test_2/".format(index)


    for i,(lr_data, hr_data) in enumerate(dataloader):
        # This time, validation period would be different
        # since they need to be merged again to measure the evaluation metrics.
        lr_patches = lr_data
        hr_patches = hr_data.cuda(device)
        lr_patches = inputdata(lr_patches)
        # hr_patches = inputdata(hr_patches)
        lr_patches = lr_patches.cuda(device)
        # hr_patches = hr_patches.cuda(device)

        with torch.set_grad_enabled(False):
            sr_patches = netG(lr_patches)
            # print(sr_patches)
            # print(sr_patches.size)
            # print(sr_patches.size())
            #
            # print(device)
            sr_patches = outputdata(sr_patches)
            #低复原
            lr_patches = outputdata(lr_patches)
            # print("复原后的")
            # print(sr_patches.size())
            if device != "cpu":
                # print("使用的是"+str(device))
                # print("这里是cuda")
                #反函数
                sr_patches = torch.special.expm1(sr_patches)
                sr_patches = sr_patches * 1e-8
                hr_patches = torch.special.expm1(hr_patches)
                hr_patches = hr_patches * 1e-8
                lr_patches = torch.special.expm1(lr_patches)
                lr_patches = lr_patches * 1e-8

                # print("维度")
                # print(sr_patches.size())
                # 保存真实的数据
                # 把cuda tensor转为cpu tensor，再转numpy
                sr_patches = sr_patches.detach().cpu().numpy()[0, 0, :, :, :]
                hr_patches = hr_patches.detach().cpu().numpy()[0, 0, :, :, :]
                lr_patches = lr_patches.detach().cpu().numpy()[0, 0, :, :, :]
                # hr_patches = hr_patches.detach().cpu().numpy()[0, 0, :, :, :]
            else:
                print("这里是cpu")
                print("使用的是"+str(device))
                sr_patches = sr_patches.detach().numpy()[0, 0, :, :, :]
                # 保存真实的数据
                hr_patches = hr_patches.detach().numpy()[0, 0, :, :, :]

            # print(sr_patches.shape)
            predict = sr_patches

            #反函数
            # predict = torch.special.expm1(predict)
            # predict = predict*1e-8
            # hr_patches = torch.special.expm1(hr_patches)
            # hr_patches = hr_patches*1e-8
            # for index1 in test_sampler:
            #     print("index: {}, data: {}".format(str(index1), str(test_indices[index1])))
            print("序号:{}--{}-----{}".format(i,test_indices[i],test_indices[i]))
            #原始高光子数据
            nii_name = image_folder + '%sreal_A.nii' % (test_indices[i])
            print("序号:{}".format(i))
            print('%sreal_A.nii' % (test_indices[i]))

            nii_img = nib.Nifti1Image(hr_patches, img_sample.affine, img_sample.header)
            nib.save(nii_img, nii_name)

            #生成的假高光子数据
            nii_name = image_folder + '%sfake_A.nii' % (test_indices[i])
            nii_img = nib.Nifti1Image(predict, img_sample.affine, img_sample.header)
            nib.save(nii_img, nii_name)

            #原始数据低光子数据
            nii_name = image_folder + '%sdi_A.nii' % (test_indices[i])
            nii_img = nib.Nifti1Image(lr_patches, img_sample.affine, img_sample.header)
            nib.save(nii_img, nii_name)
    #数据改为 2 3 1 就复原了

if __name__ == '__main__':

    # Number of workers for dataloader
    workers = 0
    # Batch size. It controls the number of samples once download
    batch_size = 1
    ngpu = 2
    # set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda")

    # 修改dataset和dataloader

    # 加载数据集
    from niidataset import NiiDataset
    dataset = NiiDataset("./data/data1/")
    # ----------------------------------------------
    dataset_size = len(dataset)
    # Set percentage of data spliting
    train_split = 0.7
    validate_split = 0.1
    evaluate_split = 0.1
    test_split = 0.1
    # Set shuffle and stablize random_seed
    shuffle_dataset = True
    random_seed = 999
    # get indices for spliting train, valiadate, evaluate, test sets
    train_size = math.ceil(train_split * dataset_size)
    validate_size = int(validate_split * dataset_size)
    evaluate_size = int(evaluate_split * dataset_size)
    test_size = int(dataset_size - train_size - validate_size - evaluate_size)

    indices = list(range(dataset_size))
    import numpy as np
    #每次划分的数据集一样
    if shuffle_dataset:
        np.random.seed(random_seed)#用于生成指定随机数。seed()被设置了之后，np,random.random()可以按顺序产生一组固定的数组，
        # 如果使用相同的seed()值，则每次生成的随机数都相同，如果不设置这个值，那么每次生成的随机数不同。但是，只在调用的时候seed()一下并不能使生成的随机数相同，需要每次调用都seed()一下，表示种子相同，从而生成的随机数相同。
        np.random.shuffle(indices)#np.random,shuffle作用就是重新排序返回一个随机序列作用类似洗牌
        print(indices)
    train_indices = indices[:train_size]
    # To reduce computational cost in validation period, you can intentionally decrease the validation size.
    loc_val = train_size + validate_size
    # loc_val_reduced = train_size + math.ceil(validate_size/3)
    # val_indices = indices[train_size:loc_val_reduced]
    val_indices = indices[train_size:loc_val]
    loc_eval = loc_val + evaluate_size
    eval_indices = indices[loc_val:loc_eval]
    test_indices = indices[loc_eval:]
    num_train = len(train_indices)
    num_val = len(val_indices)
    num_eval = len(eval_indices)
    num_test = len(test_indices)
    print('training set number:{}'.format(len(train_indices)))
    print('validation set number:{}'.format(len(val_indices)))
    print('evaluation set number:{}'.format(len(eval_indices)))
    print('test set number:{}'.format(len(test_indices)))
    # Use for pre-training (we don't want too much validation)
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    eval_sampler = SubsetRandomSampler(eval_indices)
    test_sampler = SubsetRandomSampler(test_indices)

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
    eval_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              sampler=eval_sampler,
                                              shuffle=False,
                                              num_workers=workers)
    test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              sampler=test_sampler,
                                              shuffle=False,
                                              num_workers=workers)

    dataloaders = {'train': train_loader, 'val': validation_loader}
    dataset_sizes = {'train': len(train_sampler), 'val': len(valid_sampler)}
    # ----------------------------------------------
    # set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    predict_G(test_loader)


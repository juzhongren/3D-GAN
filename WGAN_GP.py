import time
import numpy as np
import pickle
import re
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from ecbm6040.metric.eval_metrics import ssim, psnr, nrmse

# for gradient penalty
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

import os

class WGAN_GP(object): 
    """
    This class is for the mDCSRN+SRGAN network.
    Args:
    netG (model, or model.module) - Generator.
    netD (model, or model.module) - Discriminator.
    supervised_criterion (torch.nn.modules.loss) - the predefined loss function for the generator, in this project, we use nn.L1Loss().
    D_criterion (torch.nn.modules.loss) - the predefined loss function for the pretraining of discriminator.
    The idea is to let the discriminator first become a good classifier. So, we use nn.BCELoss().
    device (torch.device) - the device you set.
    ngpu (int) - how many GPU you use.
    lr (float) - the learning rate for pretraining. By default, the value is 5e-6.
    joint_opt_param (float) - the \lambda in the loss function. By default, the value is 0.001.
   """
    def __init__(self, netG, netD, 
             supervised_criterion,
             device, ngpu,
             lr=1e-8, joint_opt_param=0.001):
        self.netG = netG
        self.netD = netD
        self.supervised_criterion = supervised_criterion
        # self.D_criterion = D_criterion
        self.device = device
        self.ngpu = ngpu
        self.lr = lr
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr)
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr)
        self.optimizer_preD = optim.Adam(self.netD.parameters(),lr=self.lr)
        self.lmda = joint_opt_param

    def wasserstein_loss(self, D_fake, D_real=torch.Tensor([0.0])):
        '''
        This function calculate the Earth Mover (EM) distance for the wasserstein loss.
        此函数计算 wasserstein 损失的 Earth Mover (EM) 距离。

        (Input) D_fake: the Discriminator's output digit for SR images.
        (Input) D_real: the Discriminator's output digit for HR images. For Generator training, you don't input D_real. That time, we use the default setting: D_real = torch.Tensor([0.0]).
        (Output) G_loss: the Generator's loss only for WGAN's part.
        (Output) D_loss: the Discriminator's loss.
        '''
        D_real = D_real.cuda(self.device)
        D_loss = - (torch.mean(D_real) - torch.mean(D_fake))
        G_loss = - torch.mean(D_fake)
        return G_loss, D_loss

    def calc_gradient_penalty(self, real_data, generated_data):

        # gp weight
        gp_weight = 10

        b_size = real_data.size()[0]
        # print("batch_size是{}".format(b_size))
        # Calculate interpolation
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(b_size, 1, 64, 64, 64)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.cuda()

        # Get random interpolation between real and fake samples
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.netD(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]  ## only_inputs= True 수정

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(b_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return gp_weight * ((gradients_norm - 1) ** 2).mean()
    #更新判别器
    def updateD(self, lr_patches, hr_patches):
        '''
        This function completes the update of D network.
        升级判别器， LR是低分辨率图像，HR是高分辨率图像
        (Input) lr_patches: the LR patches.
        (Input) hr_patches: the HR patches.
        (Output) sr_patches: the SR patches.
        (Output) D_loss: the Discriminator's loss.
        (Output) G_loss: the Generator's loss only for WGAN's part.
        (Output) loss: the network's loss
        '''
        # forward
        #判别器和生成器的参数都要更
        # forward
        for p in self.netG.parameters():
            p.requires_grad = False
        for p in self.netD.parameters():
            p.requires_grad = True

        # hr_patches_ = Variable(hr_patches)
        # lr_patches_ = Variable(lr_patches)

        # input SR to D (fake)
        sr_patches = self.netG(lr_patches)
        D_fake = self.netD(sr_patches)
        # input HR to D (real)
        D_real = self.netD(hr_patches)
        # Supervised Loss
        # Calculate L1 Loss
        L1_loss = self.supervised_criterion(sr_patches, hr_patches)
        # WGAN's Loss
        # Calculate Wasserstein Loss
        G_loss, D_loss = self.wasserstein_loss(D_fake, D_real)

        gradient_penalty = self.calc_gradient_penalty(hr_patches, sr_patches)
        # gradient_penalty.backward(retain_graph=True)

        D_loss += gradient_penalty

        # Semi-supervised Loss (main loss)
        # L1_loss from generator and D_loss from Discriminator as wgan-gp
        loss = L1_loss + self.lmda * G_loss
        # backward + optimize only if in training phase
        D_loss.backward()
        self.optimizerD.step()

        return sr_patches, D_loss, G_loss, loss

    #更新生成器
    def updateG(self, lr_patches, hr_patches):
        '''
        This function completes the update of G network.
        升级生成器
        (Input) lr_patches: the LR patches.
        (Input) hr_patches: the HR patches.
        (Output) sr_patches: the SR patches.
        (Output) G_loss: the Generator's loss only for WGAN's part.
        (Output) loss: the network's loss
        '''

        for p in self.netG.parameters():
            p.requires_grad = True
        for p in self.netD.parameters():
            p.requires_grad = False # to avoid computation

        #hr_patches_ = Variable(hr_patches)
        #lr_patches_ = Variable(lr_patches)


        # input SR to D (fake)
        sr_patches = self.netG(lr_patches)
        D_fake = self.netD(sr_patches)
        # Supervised Loss
        # Calculate L1 Loss
        L1_loss = self.supervised_criterion(sr_patches, hr_patches)
        # WGAN's Loss
        # Calculate Wasserstein Loss
        G_loss,D_loss = self.wasserstein_loss(D_fake)
        # Semi-supervised Loss (main loss)
        loss = L1_loss + self.lmda * G_loss
        # backward + optimize only if in training phase
        loss.backward()
        self.optimizerG.step()

        return sr_patches, D_loss, G_loss, loss

    
    def forwardDG(self, lr_patches, hr_patches):
        '''
        This function only goes through the forward of the network. It's used in validation period.
        此功能仅通过网络的前向传播。 在验证期内使用。
        
        (Input) lr_patches: the LR patches.
        (Input) hr_patches: the HR patches.
        (Output) sr_patches: the SR patches.
        (Output) D_loss: the Discriminator's loss.
        (Output) G_loss: the Generator's loss only for WGAN's part.
        (Output) loss: the network's loss 
        '''
        # input SR to D (fake)
        sr_patches = self.netG(lr_patches)
        D_fake = self.netD(sr_patches)
        # input HR to D (real)
        D_real = self.netD(hr_patches)
        # Supervised Loss
        # Calculate L1 Loss
        L1_loss = self.supervised_criterion(sr_patches, hr_patches)
        # WGAN's Loss
        # Calculate Wasserstein Loss
        G_loss, D_loss = self.wasserstein_loss(D_fake, D_real)

        # gp
        # gradient_penalty = self.calc_gradient_penalty(hr_patches,sr_patches)
        # gradient_penalty.backward()
        # D_loss += gradient_penalty

        # Semi-supervised Loss (main loss)
        loss = L1_loss + self.lmda * G_loss
        return sr_patches, D_loss, G_loss, loss
        
    def training_epoch(self, dataloaders,
                 max_epochs=0, num_epochs_pre = 0,
                 pretrainedG = ' ',pretrainedD =' ',datasetname=" ",index = 0):
        """
        This function is the training of the network.
        Args:
            dataloaders (torch.utils.data.DataLoader) - the torch dataloader you defined. For a default setting, use a dictionary with interleaved phases with 'train' and 'val'. See in the main.ipynb.
            max_step (int) - the maximum step of the whole training (including pretraining). By default, we set the value to be 550000 (in the paper, it's 1050000).总共的训练步数
            整个训练的最大步长（包括预训练）。 默认情况下，我们将值设置为 550000（在论文中，它是 1050000）。
            first_steps (int) - # of steps of training of Discriminator alone at first. By default, we set the value to be 10000.单独训练判别器的步数
            首先单独训练鉴别器的步骤。 默认情况下，我们将值设置为 10000。
            num_steps_pre (int) - # of steps of pretraining of Generator. It should be equal to the actual pretrained steps (250000 here). 之前训练生成器的步数
            生成器预训练的步骤。 它应该等于实际的预训练步骤（此处为 250000）
            patch_size (int) - the number of patches once send into the model. By default, the value is 2.
            cube_size (int) - the size of one patch (eg. 64 means a cubic patch with size: 64x64x64), this is exact the size of the model input. By default, the value is 64.
            usage (float) - the percentage of usage of one cluster of patches. For example: usage= 0.5 means to randomly pick 50% patches from a cluster of 200 patches. This is only used in training period. By default, the value is 1.0.
            一组补丁的使用百分比。 例如：usage= 0.5 表示从 200 个补丁的集群中随机挑选 50% 的补丁。 这仅在训练期间使用。 默认情况下，该值为 1.0。
            pretrained_G (string) - the root of the saved pretrained Generator.
            pretrained_D (string) - the root of the saved pretrained Discriminator. 
        """
        since = time.time()

        print ("WGAN training...")


        #如果不为空，加载这个按这个步骤接着训练生成器
        if pretrainedG != ' ':
            self.netG.load_state_dict(torch.load(pretrainedG))
            pre_epoch = int(re.sub("\D", "", pretrainedG))  #start from the pretrained model's step
            print("已经训练了")
            print(int(re.sub("\D", "", pretrainedG)))
            train_loss=[]
            train_D_loss=[]
            val_loss=[]
            val_D_loss=[]
        else:
            # record loss function of the whole period
            pre_epoch = 0
            train_loss=[]
            train_D_loss=[]
            val_loss=[]
            val_D_loss=[]

        #是否接着训练判别器
        if pretrainedD != ' ':
            self.netD.load_state_dict(torch.load(pretrainedD))
            if not os.path.exists(datasetname + "loss_history/loss_history1/loss_history_{}".format(index)):
                os.makedirs(datasetname + "loss_history/loss_history1/loss_history_{}".format(index))
            root_dir = datasetname + "loss_history/loss_history1/loss_history_{}".format(index)
            # recall the loss history to continue
            f=open(root_dir+'/train_loss_epoch{}.txt'.format(pre_epoch),'rb')
            train_loss= pickle.load(f)
            f.close()
            f=open(root_dir+'/train_loss_D_epoch{}.txt'.format(pre_epoch),'rb')
            train_D_loss= pickle.load(f)
            f.close()
            f=open(root_dir+'/val_loss_epoch{}.txt'.format(pre_epoch),'rb')
            val_loss= pickle.load(f)
            f.close()
            f=open(root_dir+'/val_loss_D_epoch{}.txt'.format(pre_epoch),'rb')
            val_D_loss= pickle.load(f)
            f.close()

        # if transfer from a single gpu case, set multi-gpu here again.
        if (self.device.type == 'cuda') and (self.ngpu > 1):
            print("多GPU")
            self.netG = nn.DataParallel(self.netG, list(range(self.ngpu)))
            self.netD = nn.DataParallel(self.netD, list(range(self.ngpu)))
        # self.netG.cuda(device=self.device)
        # self.netD.cuda(device=self.device)

        #开始训练
        step = 0
        for epoch in range(max(num_epochs_pre+1,pre_epoch+1),max_epochs+1):
            print('epoch {}/{}'.format(epoch, max_epochs))
            print('-' * 20)
            mean_generator_content_loss = 0.0
            mean_discriminator_loss = 0.0
            # Each epoch has 10 training and validation phases
            #
            for fold in range(10):
                for phase in ['train', 'val']:
                    if phase == 'train':
                        self.netD.train()  # Set model to training mode
                        self.netG.train()
                    else:
                        self.netD.eval()   # Set model to training mode
                        self.netG.eval()
                    
                    batch_loss = []
                    batch_G_loss = []
                    batch_D_loss = []
                    val_ssim = []
                    val_psnr = []
                    val_nrmse = []
                    # print(dataloaders)
                    for lr_data, hr_data in dataloaders[phase][fold]:
                        # #这里原本是把大的分开成小的patch
                        # if phase == 'train':
                        #     pass
                        # else:
                        #     sr_data_cat = torch.Tensor([])  # for concatenation
                        # This time, validation period would be different 
                        # since they need to be merged again to measure the evaluation metrics.

                        lr_patches = lr_data.cuda(self.device)
                        hr_patches = hr_data.cuda(self.device)
                        # zero the parameter gradients
                        # self.optimizerG.zero_grad()
                        # self.optimizerD.zero_grad()

                        if phase == 'train':
                            # Training phase
                            #训练gan网络
                            with torch.set_grad_enabled(True):
                                #######
                                ##更新判别器5步，更新一步生成器
                                ##
                                ##
                                #######
                                # if step % 5 == 0:
                                #     # return sr_patches, D_loss, G_loss, loss
                                #     sr_patches, D_loss, G_loss, loss = self.updateG(lr_patches, hr_patches)
                                # else:
                                #     # return sr_patches, D_loss, G_loss, loss
                                #     sr_patches, D_loss, G_loss, loss = self.updateD(lr_patches, hr_patches)
                                self.optimizerG.zero_grad()
                                self.optimizerD.zero_grad()
                                sr_patches, D_loss, G_loss, loss = self.updateD(lr_patches, hr_patches)
                                # print('train epoch:{} Step: {}, loss= {:.4f}, D_loss= {:.4f}, G_loss= {:.4f}'.format(epoch,step,
                                #                                                                         loss.item(),
                                #                                                                         D_loss.item(),
                                #                                                                         G_loss.item()))
                                self.optimizerG.zero_grad()
                                self.optimizerD.zero_grad()
                                sr_patches, D_loss, G_loss, loss = self.updateG(lr_patches, hr_patches)

                                # lr_patches = lr_patches.cuda(self.device)
                                # hr_patches = hr_patches.cuda(self.device)
                                # return sr_patches, D_loss, G_loss, loss
                                ##########################
                                # print(torch.cuda.mem_get_info(device=self.device))
                                # sr_patches = self.netG(lr_patches)
                                # # Train D
                                # # print("训练判别器")
                                # self.optimizerD.zero_grad()
                                # logits_real = self.netD(hr_patches).mean()
                                # logits_fake = self.netD(sr_patches).mean()
                                # # gradient_penalty = self.calc_gradient_penalty(hr_patches, sr_patches)
                                # # gradient_penalty = compute_gradient_penalty(netD, real_img_hr, fake_img_hr)
                                # D_loss = logits_fake - logits_real #+ gradient_penalty
                                # # D_loss.backward(retain_graph=True)
                                # D_loss.backward()
                                # self.optimizerD.step()
                                #
                                # sr_patches = self.netG(lr_patches)
                                # # Train G
                                # # print("训练生成器")
                                # self.optimizerG.zero_grad()
                                # image_loss = self.supervised_criterion(sr_patches, hr_patches)
                                # G_loss = -1 * self.netD(sr_patches).mean()
                                # loss = image_loss + self.lmda * G_loss
                                # loss.backward()
                                # self.optimizerG.step()
                                ###########################
                                step += 1
                                # print('train epoch:{} Step: {}, loss= {:.4f}, D_loss= {:.4f}, G_loss= {:.4f}'.format(epoch,step,
                                #                                                                         loss.item(),
                                #                                                                         D_loss.item(),
                                #                                                                         G_loss.item()))
                            # statistics
                            # 取loss值
                            batch_loss = np.append(batch_loss, loss.item())#为原始array添加一些values
                            batch_G_loss = np.append(batch_G_loss, G_loss.item())
                            batch_D_loss = np.append(batch_D_loss, D_loss.item())
                        else:
                            #这里也是if phase == 'val':
                            #验证的代码,使用上边写的forwardDG
                            # Validation phase
                            with torch.set_grad_enabled(False):
                                sr_patches, D_loss, G_loss, loss = self.forwardDG(lr_patches, hr_patches)

                                # print('val epoch:{} Step: {}, loss= {:.4f}, D_loss= {:.4f}, G_loss= {:.4f}'.format(
                                #                                                                             epoch, step,
                                #                                                                             loss.item(),
                                #                                                                             D_loss.item(),
                                #                                                                             G_loss.item()))
                            # statistics
                            batch_loss = np.append(batch_loss, loss.item())
                            batch_G_loss = np.append(batch_G_loss, G_loss.item())
                            batch_D_loss = np.append(batch_D_loss, D_loss.item())
                            # concatenate patches, send patches to cpu to save GPU memory
                            sr_data_cat = sr_patches.to("cpu")
                            # sr_data_cat = torch.cat([sr_data_cat, sr_patches.to("cpu")], 0)
                            #为了计算验证集的loss 和 评价参数
                            # if phase == 'val':
                            # calculate the evaluation metric

                            # sr_patches代表假数据
                            sr_data = sr_data_cat
                            batch_ssim = ssim(hr_data, sr_data)
                            batch_psnr = psnr(hr_data, sr_data)
                            batch_nrmse = nrmse(hr_data, sr_data)
                            val_ssim = np.append(val_ssim, batch_ssim)
                            val_psnr = np.append(val_psnr, batch_psnr)
                            val_nrmse = np.append(val_nrmse, batch_nrmse)
                    #生成器的loss
                    mean_generator_content_loss = np.mean(batch_loss)
                    #判别器的loss
                    mean_discriminator_loss = np.mean(batch_D_loss)
                    if phase == 'val':
                        mean_ssim = np.mean(val_ssim)
                        std_ssim = np.std(val_ssim)
                        mean_psnr = np.mean(val_psnr)
                        std_psnr = np.std(val_psnr)
                        mean_nrmse = np.mean(val_nrmse)
                        std_nrmse = np.std(val_nrmse)
                        val_loss = np.append(val_loss, batch_loss)
                        val_D_loss = np.append(val_D_loss, batch_D_loss)
                        print('epoch:{} fold No. {} {} period. Mean main loss: {:.4f}. Mean discriminator loss: {:.4f}.'.format(epoch,fold+1, phase, mean_generator_content_loss, mean_discriminator_loss))
                        print('Metrics: subject-wise mean SSIM = {:.4f}, std = {:.4f}; mean PSNR = {:.4f}, std = {:.4f}; mean NRMSE = {:.4f}, std = {:.4f}.'.format(mean_ssim, std_ssim, mean_psnr, std_psnr, mean_nrmse, std_nrmse))
                    else:
                        train_loss = np.append(train_loss, batch_loss)
                        train_D_loss = np.append(train_D_loss, batch_D_loss)
                        # print('epoch:{} fold No.{} {} period. Mean main loss: {:.4f}. Mean discriminator loss: {:.4f}'.format(epoch,fold+1, phase, mean_generator_content_loss, mean_discriminator_loss))
                #每一个flod打印一次消息
                time_elapsed = time.time() - since
                print('Now the training uses {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                print()
            if (epoch % 10) == 0:
                if not os.path.exists(datasetname+"models/model1/model_{}".format(index)):
                    os.makedirs(datasetname+"models/model1/model_{}".format(index))
                root_dir = datasetname+"models/model1/model_{}".format(index)
                # save intermediate models for singal GPU and multi GPU
                if self.ngpu > 1:
                    torch.save(self.netG.module.state_dict(), root_dir+'/WGAN_G_epoch{}'.format(epoch))
                    torch.save(self.netD.module.state_dict(), root_dir+'/WGAN_D_epoch{}'.format(epoch))
                else:
                    torch.save(self.netG.state_dict(), root_dir+'/WGAN_G_epoch{}'.format(epoch))
                    torch.save(self.netD.state_dict(), root_dir+'/WGAN_D_epoch{}'.format(epoch))
                    # record instant loss
                # 保存训练的信息
                train_loss = np.append(train_loss, batch_loss)
                train_D_loss = np.append(train_D_loss, batch_D_loss)
                if not os.path.exists(datasetname+"loss_history/loss_history1/loss_history_{}".format(index)):
                    os.makedirs(datasetname+"loss_history/loss_history1/loss_history_{}".format(index))
                root_dir = datasetname+"loss_history/loss_history1/loss_history_{}".format(index)

                f = open(root_dir+'/train_loss_epoch{}.txt'.format(epoch), 'wb')  # wb二进制写，文件存储同样被清空
                pickle.dump(train_loss, f)
                f.close()
                f = open(root_dir+'/train_loss_D_epoch{}.txt'.format(epoch), 'wb')
                pickle.dump(train_D_loss, f)
                f.close()
                f = open(root_dir+'/val_loss_epoch{}.txt'.format(epoch), 'wb')
                pickle.dump(val_loss, f)
                f.close()
                f = open(root_dir+'/val_loss_D_epoch{}.txt'.format(epoch), 'wb')
                pickle.dump(val_D_loss, f)
                f.close()
        #for_epoch循环结束
        # 训练结束
        print("训练结束")
        root_dir = datasetname + "loss_history/loss_history1/loss_history_{}".format(index)
        # record instant loss
        train_loss = np.append(train_loss, batch_loss)
        train_D_loss = np.append(train_D_loss, batch_D_loss)
        f = open(root_dir+'/train_loss_history.txt', 'wb')
        pickle.dump(train_loss, f)
        f.close()
        f = open(root_dir+'/train_loss_D_history.txt', 'wb')
        pickle.dump(train_D_loss, f)
        f.close()
        f = open(root_dir+'/val_loss_history.txt', 'wb')
        pickle.dump(val_loss, f)
        f.close()
        f = open(root_dir+'/val_loss_D_history.txt', 'wb')
        pickle.dump(val_D_loss, f)
        f.close()
        # 保存单独的模型，没有保存结构
        # save for single GPU and multi GPU
        root_dir = datasetname + "models/model1/model_{}".format(index)
        if self.ngpu > 1:
            torch.save(self.netG.module.state_dict(), root_dir+'/final_model_G')
            torch.save(self.netD.module.state_dict(), root_dir+'/final_model_D')
        else:
            torch.save(self.netG.state_dict(), root_dir+'/final_model_G')
            torch.save(self.netD.state_dict(), root_dir+'/final_model_D')
        return self.netG, self.netD

    def test(self, dataloader,
             pretrainedG = ' ',pretrainedD =' '):
        """
        This function is the test of the network. It can be applied on evaluation set and test set.
        该功能是对网络的测试。 它可以应用于评估集和测试集。
        Args:
            dataloader (torch.utils.data.DataLoader) - the torch dataloader you defined. For a default setting, it could be either evaluation dataloader or test dataloader. See in the main.ipynb.
            patch_size (int) - the number of patches once send into the model. By default, the value is 2.
            cube_size (int) - the size of one patch (eg. 64 means a cubic patch with size: 64x64x64), this is exact the size of the model input. By default, the value is 64.
            pretrained_G (string) - the root of the saved pretrained Generator. 
            pretrained_D (string) - the root of the saved pretrained Discriminator. 
        """
        since = time.time()

        print ("WGAN testing...")
        if pretrainedD != ' ':
            self.netD.load_state_dict(torch.load(pretrainedD))
        self.netG.load_state_dict(torch.load(pretrainedG))
        test_loss=[]
        test_D_loss=[]
        self.netD.eval()   # Set model to eval mode
        self.netG.eval()
        test_ssim = []
        test_psnr = []
        test_nrmse = []

        for lr_data, hr_data in dataloader:
         # This time, validation period would be different 
         # since they need to be merged again to measure the evaluation metrics.
            lr_patches = lr_data.cuda(self.device)
            hr_patches = hr_data.cuda(self.device)
            sr_data_cat = torch.Tensor([]) # for concatenation
        # for lr_patches, hr_patches in patch_loader:
        #     lr_patches=lr_patches.cuda(self.device)
        #     hr_patches=hr_patches.cuda(self.device)
            # zero the parameter gradients
            self.optimizerG.zero_grad()
            self.optimizerD.zero_grad()
            with torch.set_grad_enabled(False):
                sr_patches, _, _, _ = self.forwardDG(lr_patches, hr_patches)
                # statistics
                # concatenate patches, send patches to cpu to save GPU memory
                sr_data_cat = torch.cat([sr_data_cat, sr_patches.to("cpu")],0)
            # calculate the evaluation metric
            sr_data = sr_data_cat
            # sr_data = depatching(sr_data_cat, lr_data.size(0))
            batch_ssim = ssim(hr_data, sr_data)
            batch_psnr = psnr(hr_data, sr_data)
            batch_nrmse = nrmse(hr_data, sr_data)
            test_ssim = np.append(test_ssim, batch_ssim)
            test_psnr = np.append(test_psnr, batch_psnr)
            test_nrmse = np.append(test_nrmse, batch_nrmse)

        mean_ssim = np.mean(test_ssim)
        std_ssim = np.std(test_ssim)
        mean_psnr = np.mean(test_psnr)
        std_psnr = np.std(test_psnr)
        mean_nrmse = np.mean(test_nrmse)
        std_nrmse = np.std(test_nrmse)
        
        f=open('example_images/image_lr.txt','wb')
        pickle.dump(lr_data[0].cpu().numpy() ,f)
        f.close()
        f=open('example_images/image_sr.txt','wb')
        pickle.dump(sr_data[0].cpu().numpy() ,f)
        f.close()
        f=open('example_images/image_hr.txt','wb')
        pickle.dump(hr_data[0].cpu().numpy() ,f)
        f.close()
        print('Metrics: subject-wise mean SSIM = {:.4f}, std = {:.4f}; mean PSNR = {:.4f}, std = {:.4f}; mean NRMSE = {:.4f}, std = {:.4f}.'.format(mean_ssim, std_ssim, mean_psnr, std_psnr, mean_nrmse, std_nrmse))
        return
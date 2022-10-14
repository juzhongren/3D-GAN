import torch
from torch.utils.data import Dataset
import glob
import nibabel as nib
from numpy.random import randint

class NiiDataset(Dataset):#所谓数据集，其实就是一个负责处理索引(index)到样本(sample)映射的一个类(class)。
    def __init__(self, datapath):
        # self.datapath = datapath
        # print([x.split('.') for x in glob.glob(r'./data/guang/*/*.nii')])
        # print('../..'+x.split('.')[4] for x in glob.glob(self.datapath + '/*.nii'))
        # self.samples = ['../..'+x.split('.')[4] for x in glob.glob(self.datapath + '/*.im')]#查找符合特定规则的文件路径名

        self.datapath = datapath
        self.samples = [x[0:-7] for x in glob.glob(self.datapath + '/*_in.nii')]
        self.is_transform = True
        # print(self.samples)
        # self.is_transform = True

    def transform(self, image):
        '''
        This function transforms the 3D image of np.ndarray (z,x,y) to a torch.ShortTensor (B,z,x,y).
        '''
        image_torch = torch.FloatTensor(image)
        return image_torch

    def norm(self,v):
        # v = 1e8*v
        return torch.log1p(1e8*v)

    # 定义旋转
    def crop_volume(self,vol, crop_pos, crop_size):
        crop_slice = (slice(1),) + tuple(
            slice(crop_pos[i], crop_pos[i] + crop_size[i - 1]) for i in range(len(crop_pos)))
        return vol[crop_slice]

    def random_crop(self,x, y, crop_size):
        t_shape = x.shape
        crop_pos = tuple(randint(0, t_shape[i + 1] - crop_size[0]) for i in range(len(t_shape) - 1))
        return self.crop_volume(x, crop_pos, crop_size), self.crop_volume(y, crop_pos, crop_size)

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        nii_in = self.samples[idx] + '_in.nii'
        nii_out = self.samples[idx] + '_out.nii'
        # print(nii_in, nii_out)
        #读取nii数据
        # mcx生成的数据是4维，压缩成3维
        # image = nib.load(nii_in).get_data()[:,:,:,0]
        # mask = nib.load(nii_out).get_data()[:,:,:,0]
        image = nib.load(nii_in).get_data()
        mask = nib.load(nii_out).get_data()

        #输出维度信息

        # (64, 64, 64, 1)变成(64, 64, 64)
        # print(image.shape)
        image = image.squeeze(-1)
        # print(image.shape)
        mask = mask.squeeze(-1)
        if(self.is_transform):
            image = self.transform(image)
            mask = self.transform(mask)
            # print("transform")
            # print(image)
            image = torch.unsqueeze(image, dim=0)
            # print(image.size)
            mask = torch.unsqueeze(mask, dim=0)

        image = self.norm(image)
        mask = self.norm(mask)

        return image,mask
        # return {"A": image, "B": mask}


if __name__ == '__main__':
    import torchvision.transforms as transforms
    # Configure dataloaders
    transforms_ = transforms.Compose([
        # transforms.Resize((opt.img_height, opt.img_width, opt.img_depth), Image.BICUBIC),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    data = NiiDataset("./data/data1/")
    data.__getitem__(0)






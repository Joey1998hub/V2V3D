from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import torch.nn.functional as F
import tifffile as tf
import utils as ut
import numpy as np
import os
import torch
    
class SyntheticData(Dataset):
    def __init__(self,lf_dir,psf_dir,device,Nnum=13,input_size=256,test_all=False):
        super().__init__()

        psfs = ut.load_psfs(psf_dir,Nnum)
        warp_psfs = ut.genWarpPSFs(psfs)
        self.psfs = psfs.to(device)
        self.warp_psfs = warp_psfs.to(device)
        psf_mean = torch.mean(self.psfs,dim=0).mean(-1).mean(-1)
        self.energy_rate = psf_mean/psf_mean.mean()
        
        self.Nnum,z_res,psf_res,_ = psfs.shape
        self.input_size = input_size

        self.lf_dir = lf_dir
        self.lf_names = sorted(os.listdir(lf_dir))
        self.postfix = self.lf_names[0].split('fp')[-1]
        self.test_lf_imgs = self.getTestLF(test_all)

        self.lf_imgs = []
        for lf_name in self.lf_names:
            lf_path = os.path.join(lf_dir,lf_name)
            self.lf_img = torch.from_numpy(tf.imread(lf_path).astype(np.float32)).squeeze()
            if torch.max(self.lf_img)>32767: self.lf_img = self.lf_img-32767 
            self.lf_img = F.relu(self.lf_img)
            amp = 0.2/torch.mean(self.lf_img)
            self.lf_img = amp*self.lf_img
            self.lf_imgs.append(self.lf_img)
            print('Loading LF: %s'%lf_name,' Amplify:',amp.item())
        self.lf_imgs = torch.stack(self.lf_imgs,dim=0)
        dbsize,angles,self.fp_res,_ = self.lf_imgs.shape
        print('Dataset size:',dbsize)

    def getTestLF(self,select_all):

        self.test_lf_names = [
            'Immune_cells_Neutrophil_highNA_zoom2_avg10_dz1um_slices39_energy10%_gain1-0.6-234-0.7_00002_00001_019_000_00_00_fp',
            ]
        
        if not select_all: self.test_lf_names = [self.test_lf_names[3]]
        
        self.test_lf_imgs = []; self.amp = torch.zeros((len(self.test_lf_names)))
        for i,test_lf_name in enumerate(self.test_lf_names):
            test_lf_name = test_lf_name+self.postfix
            lf_path = os.path.join(self.lf_dir,test_lf_name)
            self.test_lf_img = torch.from_numpy(tf.imread(lf_path).astype(np.float32)).squeeze()
            if torch.max(self.test_lf_img)>32767: self.test_lf_img = self.test_lf_img-32767 
            self.test_lf_img = F.relu(self.test_lf_img)
            self.amp[i] = 0.2/torch.mean(self.test_lf_img)
            self.test_lf_img = self.amp[i]*self.test_lf_img
            self.test_lf_imgs.append(self.test_lf_img)
            print('Loading Test LF: %s'%test_lf_name,' Amplify:',self.amp[i].item())
        self.test_lf_imgs = torch.stack(self.test_lf_imgs,dim=0)
        print('Test Dataset size:',self.test_lf_imgs.shape[0])

        return self.test_lf_imgs
    
    def getPSF(self):
        return self.psfs, self.warp_psfs, self.energy_rate
    
    def __len__(self):
        return self.lf_imgs.shape[0]

    def __getitem__(self,index):
        lf_img = self.lf_imgs[index,...]

        #random crop
        idx_start = np.random.randint(0, self.fp_res-self.input_size)
        idy_start = np.random.randint(0, self.fp_res-self.input_size)
        lf_img = lf_img[:,idx_start:idx_start+self.input_size,idy_start:idy_start+self.input_size]

        #transform
        Vertical_P = np.random.choice([0,1])
        Horizontal_P = np.random.choice([0,1])

        transform = transforms.Compose([transforms.RandomVerticalFlip(Vertical_P),transforms.RandomHorizontalFlip(Horizontal_P)])
        lf_img = transform(lf_img)

        return lf_img

        
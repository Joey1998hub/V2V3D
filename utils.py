import numpy as np;import torch
import torch.nn.functional as F
from torch.fft import fft2,ifft2
import tifffile as tf
import os;import math

#MicroNeRF utils
def generate_fp(psf,sample,fp_res=None):
    z,ra,ca = sample.shape
    _,rb,cb =psf.shape

    r = ra+rb-1
    p1 = (r-ra)/2

    a1 = torch.zeros(z,r,r).to(sample.device)
    b1 = torch.zeros(z,r,r).to(sample.device)

    a1[:,0:ra,0:ca] = sample
    b1[:,0:rb,0:cb] = psf
    conv1 = ifft2(fft2(a1)*fft2(b1))
    projections = torch.abs(conv1[:,int(p1):int(r-p1),int(p1):int(r-p1)])

    if fp_res is not None and projections.shape[-1]!=fp_res: projections = upsample(projections,size=fp_res)
    return projections

def generate_fps(psf,sample,fp_res=None):
    z,ra,ca = sample.shape
    u,_,rb,cb =psf.shape

    r = ra+rb-1
    p1 = (r-ra)/2

    a1 = torch.zeros(1,z,r,r).to(sample.device)
    b1 = torch.zeros(u,z,r,r).to(sample.device)

    a1[:,:,0:ra,0:ca] = torch.unsqueeze(sample,dim=0)
    b1[:,:,0:rb,0:cb] = psf
    conv1 = ifft2(fft2(a1)*fft2(b1))
    projections = torch.real(conv1[:,:,int(p1):int(r-p1),int(p1):int(r-p1)])

    if fp_res is not None and projections.shape[-1]!=fp_res: projections = upsample(projections,dims=2,size=fp_res)
    return projections


def warp_feats(psf,feats):
    u,ch,ra,ca = feats.shape
    u,z,rb,cb =psf.shape

    r = ra+rb-1
    p1 = (r-ra)/2

    a1 = torch.zeros(u,ch,1,r,r).to(feats.device)
    b1 = torch.zeros(u,1,z,r,r).to(feats.device)

    a1[:,:,:,0:ra,0:ca] = torch.unsqueeze(feats,dim=2)
    b1[:,:,:,0:rb,0:cb] = torch.unsqueeze(psf,dim=1)

    projections = ifft2(fft2(a1)*fft2(b1))
    projections = torch.real(projections[:,:,:,int(p1):int(r-p1),int(p1):int(r-p1)])
    return projections

def genWarpPSFs(psfs):
    psfs_flip = torch.flip(psfs,dims=[-2,-1])
    u_res,depth,height,width = psfs_flip.shape
    psfs_warp = []
    for i in range(u_res):
        psf_flip = psfs_flip[i,...]
        masks = []
        for z in range(depth):
            slice_tensor = psf_flip[z,...] 
            M = torch.sum(slice_tensor)  

            x_coords = torch.arange(width).view(1, -1)  # 1 x W
            y_coords = torch.arange(height).view(-1, 1)  # H x 1

            C_x = torch.sum(x_coords * slice_tensor) / M
            C_y = torch.sum(y_coords * slice_tensor) / M

            mask = torch.zeros((height, width), dtype=torch.float32) 
            C_x_int = round(C_x.item()) 
            C_y_int = round(C_y.item())
            mask[C_y_int, C_x_int] = 1
            masks.append(mask)
        psf_warp = torch.stack(masks,dim=0)
        psfs_warp.append(psf_warp)
    psfs_warp = torch.stack(psfs_warp,dim=0)

    psfs_warp_sum = psfs_warp.sum(dim=0).sum(dim=0)
    coords = torch.nonzero(psfs_warp_sum>0)
    coords_min = coords.min(dim=0)[0].min()
    coords_max = coords.max(dim=0)[0].max()+1

    psfs_warp = psfs_warp[:,:,coords_min-1:coords_max+1,coords_min-1:coords_max+1]
    return psfs_warp

# Utils for data processing
def normal(input):
    out = (input-torch.min(input))/(torch.max(input)-torch.min(input))
    return out

def load_uint16(path):
    input = tf.imread(path)
    input = input.astype(np.float32)
    out = torch.from_numpy(input)
    return out

def denoise(img):
    base_noise = torch.min(torch.mean(img,dim=0))
    img = img-base_noise
    print('Base Noise:',base_noise.item())
    img = torch.where(img>=0,img,torch.zeros_like(img))
    return img

def upsample(img,dims=2,factor=None,size=None,mode=None):
    if dims==2 and mode==None: mode = 'bicubic'
    elif dims==3 and mode==None: mode = 'trilinear'
    if len(img.shape) == dims:
        img = torch.unsqueeze(torch.unsqueeze(img,dim=0),dim=0)
        if size == None: img = F.interpolate(img,scale_factor=factor, mode=mode)
        else: img = F.interpolate(img,size=size, mode=mode)
        img = img[0,0,:,:]
    elif len(img.shape) == (dims+1):
        img = torch.unsqueeze(img,dim=0)
        if size == None: img = F.interpolate(img,scale_factor=factor, mode=mode)
        else: img = F.interpolate(img,size=size, mode=mode)
        img = img[0,:,:,:]
    elif len(img.shape) == (dims+2):
        if size == None: img = F.interpolate(img,scale_factor=factor, mode=mode)
        else: img = F.interpolate(img,size=size, mode=mode)
    else: raise ValueError('Invalid dims')

    return img

def calBacknoise(img,bins=256):
    hist_max = torch.mean(img).item()
    hist_min = torch.min(img).item()
    hist = torch.histc(img,bins,hist_min,hist_max)
    backnoise_index = torch.max(hist,0)[1]
    backnoise = (backnoise_index/bins)*(hist_max-hist_min)+hist_min

    return backnoise

# Utils for data loading
def load_psfs(PSF_dir,Nnum):

    psfs = []
    postfix = os.listdir(PSF_dir)[0].split('.')[1]
    for index in range(Nnum):
        psf_path = os.path.join(PSF_dir,'psf_'+str(index+1)+'.'+postfix)
        psf = torch.from_numpy(tf.imread(psf_path).astype(np.float32))
        psfs.append(psf)
        print('Load PSF:',psf_path,'PSF max:',torch.max(psf).item())

    psfs = torch.stack(psfs,dim=0).squeeze()

    return psfs

# Utils for training
def adjust_lr(lr_init,lr_decay,epoch,decay_init,decay_every,optimizer):
    if epoch<decay_init: lr = lr_init
    else: lr = lr_init * (lr_decay ** (((epoch-decay_init)//decay_every)+1))
    print('Epoch %d Learning Rate %f' %(epoch,lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def cal_psnr(original, reconstructed, max_value=1):

    mse = torch.mean((original - reconstructed) ** 2)
    psnr_value = 10 * torch.log10((max_value ** 2) / mse)

    return psnr_value

def test1b(lf,model,epoch,result_dir,log):
    if not os.path.exists(result_dir): os.makedirs(result_dir)
    with torch.no_grad():
        xguess = torch.squeeze(model(lf))
        xguess = xguess.cpu().numpy().astype(np.float32)
        save_name = 'V2V3D_1b_%s_Epoch%.2d.tif'%(log,epoch+1)
        save_path = os.path.join(result_dir,save_name)
        tf.imwrite(save_path,xguess)
        torch.cuda.empty_cache()

def test2b(lf_all,model,epoch,result_dir,log):

    if not os.path.exists(result_dir): os.makedirs(result_dir)
    if lf_all.ndim == 3: lf_all = torch.unsqueeze(lf_all,dim=0)
    with torch.no_grad():
        all_xguess = []
        for i in range(lf_all.shape[0]):
            _,_,xguess = model(lf_all[i,...])
            all_xguess.append(torch.squeeze(xguess))
        all_xguess = torch.stack(all_xguess,dim=0)

        all_xguess = all_xguess.cpu().numpy().astype(np.float16)
        save_name = 'V2V3D_Epoch%.2d.tif'%(epoch+1)
        save_path = os.path.join(result_dir,save_name)
        tf.imwrite(save_path,all_xguess)
        torch.cuda.empty_cache()

def test_patch(lf_all,model,test_size,z_res,epoch,result_dir,log):

    if not os.path.exists(result_dir): os.makedirs(result_dir)
    with torch.no_grad():
        xguess = torch.zeros((z_res,lf_all.shape[-1],lf_all.shape[-1]))
        for i in range(int(lf_all.shape[-1]/test_size)):
            for j in range(int(lf_all.shape[-2]/test_size)):
                _,_,xguess_patch = model(lf_all[:,i*test_size:(i+1)*test_size,j*test_size:(j+1)*test_size])
                xguess[:,i*test_size:(i+1)*test_size,j*test_size:(j+1)*test_size] = xguess_patch.cpu()
        xguess = xguess.numpy().astype(np.float16)
        save_name = 'V2V3D_%s_Epoch%.2d.tif'%(log,epoch+1)
        save_path = os.path.join(result_dir,save_name)
        tf.imwrite(save_path,xguess)
        torch.cuda.empty_cache()

def save_ckpt(model,ckpt_dir,epoch):
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    device = next(model.parameters()).device
    model = model.cpu()
    save_name = 'epoch_%d.mdl' %epoch
    torch.save(model.state_dict(),os.path.join(ckpt_dir,save_name))
    model = model.to(device)

def save_results(xguess,save_dir,save_name):
    save_path = os.path.join(save_dir,save_name)
    tf.imwrite(save_path,xguess.detach().cpu().numpy().astype(np.float16))
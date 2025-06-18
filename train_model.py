import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os
import configargparse
from utils import test2b, adjust_lr, generate_fps, save_ckpt, normal
from loss import fftloss, deCrosstalk_loss, posloss, zloss
from dataset import SyntheticData
from model import V2V3D

def parse_args():
    parser = configargparse.ArgumentParser(
        description='V2V3D Training',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        default_config_files=['Config/train.yaml'] 
    )
    parser.add('-c', '--config', required=False, is_config_file=True)
    
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--psf_dir', type=str)
    parser.add_argument('--lf_dir', type=str)
    
    parser.add_argument('--Nnum', type=int, default=13, help='Num of all views')
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--feat_ch', type=int, default=4)
    parser.add_argument('--dc_weight', type=float, default=0.1)
    parser.add_argument('--tv_weight', type=float, default=1e-3)
    
    parser.add_argument('--lr_init', type=float, default=8e-5)
    parser.add_argument('--lr_decay', type=float, default=0.3)
    parser.add_argument('--decay_init', type=int, default=60)
    parser.add_argument('--decay_every', type=int, default=20)
    
    parser.add_argument('--log', type=str, default='', help='specific log')
    
    return parser.parse_args()

def train(args):
    device = torch.device('cuda', args.gpu_id)
    
    curtime = time.strftime('%m_%d_%H_%M', time.localtime(time.time()))
    projection_name = args.lf_dir.split('/')[-1]
    notes = '_V2V3D_F%d_'%(args.feat_ch)+args.log+'_' if args.log != '' else '_V2V3D_F%d_'%(args.feat_ch)
    project_name = curtime+notes+projection_name.split('.tif')[0] 

    ckpt_root = './Checkpoints'
    result_root = './Results'
    tb_root = './Log'
    ckpt_dir = os.path.join(ckpt_root, project_name)
    result_dir = os.path.join(result_root, project_name)
    tb_dir = os.path.join(tb_root, project_name)
    
    for dir_path in [ckpt_dir, result_dir, tb_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    with open(os.path.join(ckpt_dir, 'config.txt'), 'w') as f:
        f.write(str(args))
    
    train_db = SyntheticData(args.lf_dir, args.psf_dir, device, args.Nnum, args.input_size)
    train_loader = DataLoader(train_db, batch_size=1, shuffle=True)

    test_lfs = train_db.test_lf_imgs.to(device)
    psfs, warp_psfs, energy_rate = train_db.getPSF()
    psf_energy_mean = torch.mean(psfs[0,...].sum(-1).sum(-1))

    u_res, z_res, psf_res, _ = psfs.shape
    select_v = np.arange(0, u_res, 2)
    remain_v = np.arange(1, u_res, 2)

    model = V2V3D(warp_psfs, z_res, select_v, remain_v, u_res, args.feat_ch).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_init)
    epochs = args.decay_init + args.decay_every
    db_size = len(train_db)
    loss_fn = nn.MSELoss()

    writer = SummaryWriter(tb_dir)

    loops = round(2000/db_size) if db_size<2000 else 1
    iters = loops*db_size
    
    for epoch in range(epochs):
        loss_total_mse = 0; iter = 0
        optimizer = adjust_lr(args.lr_init, args.lr_decay, epoch, args.decay_init, args.decay_every, optimizer)
        model.train()
        for loop in range(loops):
            for step, lf_all in enumerate(train_loader):
                lf_all = torch.squeeze(lf_all, 0).to(device)
                select_lfs = lf_all[select_v,...]
                remain_lfs = lf_all[remain_v,...]
                cv = lf_all.mean(dim=0)
                xguess1, xguess2, xguess = model(lf_all)

                gen_remain_fps = generate_fps(psfs[remain_v,...], torch.squeeze(xguess1))
                gen_select_fps = generate_fps(psfs[select_v,...], torch.squeeze(xguess2))
                gen_remain_lfs = gen_remain_fps.mean(dim=1)
                gen_select_lfs = gen_select_fps.mean(dim=1)

                loss_mse = loss_fn(gen_remain_lfs, remain_lfs) + loss_fn(gen_select_lfs, select_lfs)
                loss_fft = fftloss(gen_remain_lfs, remain_lfs) + fftloss(gen_select_lfs, select_lfs)
                loss_pos = posloss(xguess1) + posloss(xguess2)
                loss = loss_mse + loss_fft * 0.5 + loss_pos * 1e-3
                if args.dc_weight > 0: loss += deCrosstalk_loss(cv, xguess, psf_energy_mean) * args.dc_weight
                if args.tv_weight > 0: loss += zloss(xguess) * args.tv_weight
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iter += 1

                loss_total_mse += loss_mse.cpu().item()
                lfmean = torch.mean(lf_all).item()
                print('Epoch:[%d/%d]'%(epoch+1,epochs),'Iters:[%d/%d]'%(iter,iters),'MSE:',loss_mse.item(),'LFMean:',lfmean)

        loss_avg_mse = loss_total_mse/(iters)
        print('Epoch:[%d/%d]'%(epoch+1,epochs),' AVG MSE:',loss_avg_mse)
        
        writer.add_scalar('MSE Loss', loss_avg_mse, epoch)
        writer.add_image('CV Gen', normal(gen_remain_lfs[0,...].unsqueeze(0)), epoch)
        writer.add_image('CV GT', normal(remain_lfs[0,...].unsqueeze(0)), epoch)
        
        if epoch%int(epochs/10)==0 or epoch+1 == epochs: test2b(test_lfs, model, epoch, result_dir, args.log)
        if (epoch+1)%int(epochs/5)==0 or epoch+1 == epochs: save_ckpt(model, ckpt_dir, epoch)

if __name__ == '__main__':
    args = parse_args()
    train(args)

import tifffile as tf
import numpy as np
import torch
import os
import configargparse
from tqdm import tqdm
from dataset import SyntheticData
from model import V2V3D

def parse_args():
    parser = configargparse.ArgumentParser(
        description='V2V3D Testing',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        default_config_files=['Config/test.yaml'] 
    )
    parser.add('-c', '--config', required=False, is_config_file=True)
    
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--psf_dir', type=str)
    parser.add_argument('--lf_dir', type=str)
    parser.add_argument('--model_path', type=str, required=True)

    parser.add_argument('--Nnum', type=int, default=13)
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--feat_ch', type=int, default=4)

    parser.add_argument('--result_dir', type=str, default=None)
    parser.add_argument('--use_amp', action='store_true')
    
    return parser.parse_args()

def test(args):
    device = torch.device('cuda', args.gpu_id)

    projection_name = args.model_path.split('/')[-2]
    result_dir = os.path.join('./Results', projection_name + '_test')
    
    os.makedirs(result_dir, exist_ok=True)
    
    with open(os.path.join(result_dir, 'test_config.txt'), 'w') as f:
        f.write(str(args))
    
    train_db = SyntheticData(args.lf_dir, args.psf_dir, device, args.Nnum, args.input_size, test_all=True)
    
    test_lfs = train_db.test_lf_imgs.to(device)
    test_lf_names = train_db.test_lf_names
    
    if test_lfs.ndim == 3: test_lfs = torch.unsqueeze(test_lfs, dim=0)
    
    psfs, warp_psfs, energy_rate = train_db.getPSF()
    
    u_res, z_res, psf_res, _ = psfs.shape
    select_v = np.arange(0, u_res, 2)
    remain_v = np.arange(1, u_res, 2)
    
    print(f"Loading: {args.model_path}")
    model = V2V3D(warp_psfs, z_res, select_v, remain_v, u_res, args.feat_ch).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(test_lfs.shape[0]), desc="Test Bar"):
            test_lf_img = test_lfs[i, ...]
            _, _, xguess = model(test_lf_img)
            base_name = test_lf_names[i]
            recon_save_path = os.path.join(result_dir, base_name + '_recon.tif')
            if args.use_amp:
                recon_data = torch.squeeze(xguess / train_db.amp[i]).cpu().numpy().astype(np.float32)
            else:
                recon_data = torch.squeeze(xguess).cpu().numpy().astype(np.float32)
            tf.imwrite(recon_save_path, recon_data)
    
    print(f"Saved: {result_dir}")

if __name__ == '__main__':
    args = parse_args()
    test(args)
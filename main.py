import argparse
import yaml
import os
from glob import glob
from tqdm.autonotebook import tqdm
import numpy as np

from fastmri import fft2c, ifft2c
from fastmri.data import subsample
from fastmri.losses import SSIMLoss
from fastmri.data.transforms import VarNetDataTransform, center_crop, complex_center_crop
from fastmri.models import VarNet

import torch
import torch.optim as optim
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data.distributed import DistributedSampler
from utils.utils import logging, natural_sort, flatten
from utils.data import SliceDataset, UnetDataTransform, load_dataset, slice_df
from models import NormUnet, VisionTransformer, ReconNet
import time

import warnings


MODEL_NAMES = {
    'u-net': ['unet-xsmall', 'unet-small', 'unet-large'],
    'vit': ['vit-small', 'vit-large'],
    'varnet' : ['varnet-small', 'varnet-large'],
    }

# ---------------------------------------------------------HELPERS----------------------------------------------------------------------#
def setup_world(rank, world_size, master_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def create_folder_name(combination):
    folder_name = ''
    for distribution in combination:
        for v in distribution.values():
            folder_name += v + '-'

    folder_name = folder_name[:-1]
    
    return folder_name

def get_testset_filename(test_attr):
    testset_filenames = []
    for q in test_attr:
        filename = ''
        for v in q.values():
            filename += v + '-'
        filename = filename[:-1] + '.txt'
        testset_filenames.append(filename)

    return testset_filenames

def init_model(model_name, device, seed=0):
    """Init fixed U-net model"""
    torch.manual_seed(seed)
    
    if model_name == 'unet-xsmall':
        model = NormUnet(chans=10, num_pools=4)

    elif model_name == 'unet-small':
        model = NormUnet(chans=32, num_pools=4)

    elif model_name == 'unet-large':
        model = NormUnet(chans=128, num_pools=4)

    elif model_name == 'vit-small':
        net = VisionTransformer(
            avrg_img_size=320,
            patch_size=(10,10),
            in_chans=1, embed_dim=44, 
            depth=4, num_heads=9,
            )
        model = ReconNet(net)

    elif model_name == 'vit-large':
        net = VisionTransformer(
            avrg_img_size=320,
            patch_size=(10,10),
            in_chans=1, embed_dim=64, 
            depth=10, num_heads=16,
            )
        model = ReconNet(net)

    elif model_name == 'varnet-small':
        model = VarNet(num_cascades=8, chans=12)

    elif model_name == 'varnet-large':
        model = VarNet(num_cascades=12, chans=18, sens_chans=8)
    
    return model.to(device)

def get_dataloader(files, transform, num_workers, augment_data=False, slice_range=None, batch_size=1, rank=0, world_size=1):
    """Get dataloader from list of fastmri files and transform"""
    dataset = SliceDataset(files=files, transform=transform, augment_data=augment_data, slice_range=slice_range, batch_size=batch_size)
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False) 
        shuffle = False
    else:
        sampler = None
        shuffle = True
    loader = DataLoader(dataset, batch_size=1, shuffle=shuffle, pin_memory=False, num_workers=num_workers, sampler=sampler)
    return loader, sampler

def traintools(model, dataloader, num_epochs, max_lr):
    """Get optimizer and schduler"""
    optimizer = optim.Adam(model.parameters(), lr=0.0)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, 
        max_lr=max_lr,
        steps_per_epoch=len(dataloader),
        epochs=num_epochs,
        pct_start=0.01,
        anneal_strategy='linear',
        cycle_momentum=False,
        base_momentum=0., 
        max_momentum=0.,
        div_factor = 25.,
        final_div_factor=1.,
    )
    return optimizer, scheduler

def mask_func_params(accl_factors):
    center_fractions = []
    accelerations = []
    for accl_factor in accl_factors:
        if accl_factor == 2:
            center_fractions.append(0.16)
            accelerations.append(2)
        elif accl_factor == 3:        
            center_fractions.append(0.12)
            accelerations.append(3)
        elif accl_factor == 4:        
            center_fractions.append(0.08)
            accelerations.append(4)
        elif accl_factor == 8:
            center_fractions.append(0.04)
            accelerations.append(8)
        elif accl_factor == 16:
            center_fractions.append(0.02)
            accelerations.append(16)
    return center_fractions, accelerations

def setup(mode, model_name, accl_factors):
    center_fractions, accelerations = mask_func_params(accl_factors)

    mask_func = subsample.EquiSpacedMaskFunc(
        center_fractions=center_fractions,
        accelerations=accelerations,
        )
    
    if mode == 'train':
        use_seed = False
    elif mode == 'test':
        use_seed = True
    
    if model_name in MODEL_NAMES['u-net'] + MODEL_NAMES['vit']:
        data_transform = UnetDataTransform('multicoil', mask_func, use_seed=use_seed)

    elif model_name in MODEL_NAMES['varnet']:
        data_transform = VarNetDataTransform(mask_func, use_seed=use_seed)

    return data_transform

def normalized_eval(output, target):
    mt = target.mean(dim=(-2, -1), keepdim=True)
    st = target.std(dim=(-2, -1), keepdim=True)
    mo = output.mean(dim=(-2, -1), keepdim=True)
    so = output.std(dim=(-2, -1), keepdim=True)
    return (output - mo) / so   * st + mt

def unet_forward(model, sample, criterion, device, normalize_output=False):
    inputs, targets, maxval = sample.image[0], sample.target[0], sample.max_value
    inputs = inputs.unsqueeze(-3).to(device)
    targets = targets.unsqueeze(-3).to(device)
    outputs = model(inputs)
    if normalize_output:
        outputs = normalized_eval(outputs, targets)
    return outputs

def varnet_forward(model, sample, criterion, device, adjust_img_size_factor=1, normalize_output=False):
    inputs, targets, maxval, mask = sample.masked_kspace[0], sample.target[0], sample.max_value, sample.mask[0]
    targets = targets.unsqueeze(-3).to(device)
    crop_size = (targets.shape[-2], targets.shape[-1])
    inputs = inputs.to(device)
    # image size
    if adjust_img_size_factor > 1:
        zero_filled = ifft2c(inputs)
        m = zero_filled.mean(dim=(-3, -2, -1), keepdim=True)
        s = zero_filled.std(dim=(-3, -2, -1), keepdim=True)
        inputs = inputs.repeat_interleave(adjust_img_size_factor,-3).repeat_interleave(adjust_img_size_factor,-2)
        inputs = ifft2c(inputs)
        center = complex_center_crop(inputs, crop_size)
        mc = center.mean(dim=(-3, -2, -1), keepdim=True)
        sc = center.std(dim=(-3, -2, -1), keepdim=True)
        inputs = (inputs - mc) / sc * s + m
        inputs = fft2c(inputs)
        mask = mask.repeat_interleave(adjust_img_size_factor,3)

    outputs = model(inputs.to(device), mask.to(device))
    outputs = center_crop(outputs, crop_size).unsqueeze(-3)
    if normalize_output:
        outputs = normalized_eval(outputs, targets)

    return outputs
    
def train_model(
    model_type, model, trainloader, num_epochs, criterion, 
    optimizer, scheduler, save_path, checkpoints, 
    disable=False, first_epoch=1, 
    sampler=None, dist_training=False, rank=0
):
    path_to_checkpoints = os.path.join(save_path, 'checkpoints')
    if rank == 0:
        if not os.path.exists(path_to_checkpoints):
            os.makedirs(path_to_checkpoints)
            print(f"Created {path_to_checkpoints}")
        else:
            print(f"{path_to_checkpoints} already exist. Continue.")
        
    model.train()
    device = next(model.parameters()).device if not dist_training else rank
    criterion = criterion

    for epoch in tqdm(range(first_epoch, first_epoch+num_epochs), disable=disable):
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_loss = 0.0
        with tqdm(total=len(trainloader), disable=disable) as pbar:
            for sample in trainloader:
                optimizer.zero_grad(set_to_none=True)
                if model_type == 'unet':
                    outputs = unet_forward(model, sample, criterion, device)
                elif model_type == 'varnet':
                    outputs = varnet_forward(model, sample, criterion, device)
                targets, maxval = sample.target[0], sample.max_value
                targets = targets.unsqueeze(-3).to(device)
                loss = criterion(outputs, targets, maxval.to(device))    
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2.)
                optimizer.step()
                try:
                    scheduler.step()
                except ValueError:
                    warnings.warn("Scheduler total steps reached.")
                pbar.update(1)
                train_loss += loss.item()
        train_loss = train_loss/len(trainloader)
        if dist_training:
            dist.barrier()
        if rank == 0:
            with open(os.path.join(save_path,'train_loss.txt'), 'a') as f:
                f.write(str(train_loss)+'\n')
            if epoch in checkpoints:
                checkpoint = { 
                    'epoch': epoch,
                    'model_state_dict': model.state_dict() if not dist_training else model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()}
                torch.save(checkpoint, os.path.join(path_to_checkpoints, 'checkpoint_'+str(epoch)+'.pth'))
            if not disable:
                print('Epoch {}, Train loss.: {:0.4f}'.format(epoch, train_loss))
                
def cut_out_pathology(sample, df_pathology, outputs, targets):
        fname, slice_num = sample.fname[0], sample.slice_num.item()
        df = df_pathology
        file = df[(df['filename'] == fname) & (df['start_slice'] == slice_num)]
        x = int(file.x.item())
        y = int(file.y.item())
        w = int(file.width.item())
        h = int(file.height.item())
        a = (h * w) / (targets.shape[-1]*targets.shape[-2])
        targets = torch.flip(targets, dims=(-2, ))
        targets = targets[..., y:y+h, x:x+w]
        outputs = torch.flip(outputs, dims=(-2, ))
        outputs = outputs[..., y:y+h, x:x+w]

        m = min(targets.shape[-1], targets.shape[-2])
        
        if m < 7:
            s = 7/m
            targets = interpolate(targets, scale_factor = s)
            outputs = interpolate(outputs, scale_factor = s)
        
        return outputs, targets, a

def test_model(model_type, model, testloader, criterion, save_path, filename=None, disable=False, normalize_output=False, results_folder=None, adjust_img_size_factor=1, 
               eval_pathology=False, df_pathology=None, min_area=None, max_area=None):
    if filename is None:
        filename = 'test_loss.txt'
    else:
        assert isinstance(filename, str), "filename must be a string"
        
    assert isinstance(results_folder, str), "result_folder must be a string"
    save_path = os.path.join(save_path, results_folder) if results_folder is not None else save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created {save_path}")
    else:
        print(f"{save_path} already exist. Continue.")


    model.eval()
    device = next(model.parameters()).device
    test_loss = 0
    count = 0
    with torch.no_grad():
        with tqdm(total=len(testloader), disable=disable) as pbar:
            for sample in testloader:
                if model_type == 'unet':
                    outputs = unet_forward(model, sample, criterion, device, normalize_output=normalize_output)
                elif model_type == 'varnet':
                    outputs = varnet_forward(model, sample, criterion, device, normalize_output=normalize_output, adjust_img_size_factor=adjust_img_size_factor)
                targets, maxval = sample.target[0], sample.max_value
                targets = targets.unsqueeze(-3).to(device)
                if eval_pathology:
                    outputs, targets, area = cut_out_pathology(sample, df_pathology, outputs, targets)
                    if min_area < area <= max_area:
                        loss = criterion(outputs, targets, maxval.to(device))
                        count += 1
                    else:
                        loss = torch.tensor([0.])
                else:
                    loss = criterion(outputs, targets, maxval.to(device))    
                    count += 1

                test_loss += loss.item()
                pbar.update(1)
        test_loss = test_loss/count
        with open(os.path.join(save_path, filename), 'a') as f:
            f.write(str(test_loss)+'\n')
        if not disable:
            print('Test loss.: {:0.4f}'.format(test_loss))
    return test_loss

def test_checkpoints(
    save_dir, checkpoints, test_func, 
    model_type, model, testloader, 
    criterion, testset_filenames=None, 
    disable = False,
    normalize_output = False,
    results_folder = None,
    adjust_img_size_factor=1,
    eval_pathology=False, df_pathology=None, min_area=None, max_area=None,
    ):
    test_loss = []
    device = next(model.parameters()).device
    for file in natural_sort(checkpoints):
        print(f"Evaluating: {file}")
        checkpoint = torch.load(file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        loss = test_func(model_type, model, testloader, criterion, save_dir, testset_filenames, disable=disable, normalize_output=normalize_output, results_folder=results_folder, adjust_img_size_factor=adjust_img_size_factor, 
                         eval_pathology=eval_pathology, df_pathology=df_pathology, min_area=min_area, max_area=max_area)
        test_loss.append(loss)

    return test_loss

# ----------------------------------------------------------TRAIN/TEST-----------------------------------------------------------------#
def train(
    rank,
    master_port,
    world_size,
    device,
    model_name,
    save_dir,
    trainset_csv,
    train_attr,
    finetune_cp,
    num_workers,
    accl_factors,
    num_epochs,
    lr,
    checkpoint_epochs,
    batch_size,
    disable_tqdm,
    augment_data,
    resume_training,
    model_seed,
):  
    if world_size > 1:
        dist_training  = True
        setup_world(rank, world_size, master_port) if world_size > 1 else None
        device = rank        
        map_location = 'cuda:%d' % rank 
        batch_size = batch_size//world_size
    else: 
        dist_training = False
        map_location = device

    # Load data        
    df = load_dataset(trainset_csv, attr=train_attr)

    lr = lr
    if model_name in MODEL_NAMES['u-net'] + MODEL_NAMES['vit']:
        model_type = 'unet'
    elif model_name in MODEL_NAMES['varnet']:
        model_type = 'varnet'

    files = df['folderDirectory'] + df['filename']
    slice_range = list(zip(df.start_slice, df.end_slice))
    model = init_model(model_name, device, seed=model_seed)
    if finetune_cp is not None:
        checkpoint = torch.load(finetune_cp, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    data_transform = setup('train', model_name, accl_factors)
    dataloader, sampler = get_dataloader(files, data_transform, num_workers, augment_data, slice_range, batch_size=batch_size, rank=rank, world_size=world_size)
    criterion = SSIMLoss().to(device)
    optimizer, scheduler = traintools(model, dataloader, num_epochs, lr)

    # Training
    if not resume_training:
        logging(save_dir, 'Start training')
        first_epoch = 1
    else:
        assert glob(os.path.join(save_dir, 'checkpoints/*.pth')), 'Resume training failed: No checkpoints saved.'
        logging(save_dir, 'Resume training.')
        checkpoint = natural_sort(glob(os.path.join(save_dir, 'checkpoints/*.pth')))[-1]
        checkpoint = torch.load(checkpoint, map_location=map_location)
        checkpoint_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        logging(save_dir, 'Model from epoch {:d} loaded.'.format(checkpoint_epoch))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging(save_dir, 'Optimizer loaded. Current learning rate: {:f}'.format(optimizer.param_groups[0]['lr']))
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logging(save_dir, 'Scheduler loaded. Current learning rate: {:f}'.format(scheduler.get_last_lr()[0]))
        first_epoch = checkpoint_epoch+1
        assert first_epoch <= num_epochs, 'Cannot resume. Training is already finished.'
        num_epochs = num_epochs-checkpoint_epoch
        print('Resume training at epoch {:d}'.format(first_epoch))
        logging(save_dir, 'Resume training at epoch {:d}'.format(first_epoch))

    if dist_training:
        model = DDP(model, device_ids=[rank])

    train_model(
        model_type, model, dataloader, num_epochs, criterion, 
        optimizer, scheduler, save_dir, checkpoint_epochs, disable = disable_tqdm, 
        first_epoch=first_epoch, sampler=sampler, dist_training=dist_training, rank=rank
    )
    logging(save_dir, 'Finish training')

    if dist_training:
        cleanup()

def test(
    model_name,
    save_dir,
    testset_csv,
    test_attr,
    device,
    num_workers,
    accl_factors,
    disable_tqdm,
    normalize_output,
    results_folder,
    adjust_img_size_factor,
    eval_pathology, min_area, max_area,
):

    # Setup
    data_transform = setup('test', model_name, accl_factors)
    criterion = SSIMLoss().to(device)
    testset_filenames = get_testset_filename(test_attr)
    # Load data
    df = load_dataset(testset_csv)
    # Testing
    logging(save_dir, 'Start testing')
    if model_name in MODEL_NAMES['u-net'] + MODEL_NAMES['vit']:
        model_type = 'unet'
    elif model_name in MODEL_NAMES['varnet']:
        model_type = 'varnet'
    cp_path = os.path.join(save_dir, 'checkpoints')
    assert glob(cp_path + '/*.pth'), f'Cannot run evaluation. {cp_path} is empty.'
    checkpoints = natural_sort(glob(cp_path + '/*.pth'))
    model = init_model(model_name, device)
    logging(save_dir, "Testing on filtered attributes")
    for attr, fname in zip(test_attr, testset_filenames):
        logging(save_dir, "Testing {:s}, {:s}".format(str(attr), fname))
        df_sliced = slice_df(df, attributes=attr) if attr is not None else df
        files = df_sliced['folderDirectory'] + df_sliced['filename']
        slice_range = list(zip(df_sliced.start_slice, df_sliced.end_slice))
        dataloader, _ = get_dataloader(files, data_transform, num_workers, slice_range=slice_range)
        loss_cps = test_checkpoints(save_dir, checkpoints, test_model, model_type, model, dataloader, criterion, fname, disable = disable_tqdm, normalize_output=normalize_output, results_folder=results_folder, adjust_img_size_factor=adjust_img_size_factor, 
                                    eval_pathology=eval_pathology, df_pathology=df_sliced, min_area=min_area, max_area=max_area)
        best_idx = np.argmin(np.array(loss_cps))
        logging(save_dir, "Best loss: {:f}, {:s}".format(loss_cps[best_idx], os.path.basename(checkpoints[best_idx])))
    logging(save_dir, 'Finish testing')

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str,
        help="configuration of the experiment as yaml file"
    )
    parser.add_argument("--finetune", type=str, default=None,
        help="path to checkpoint for finetuning"
    )
    parser.add_argument("--device", type=str,  default=None,
        help="E.g., cpu, cuda:0, cuda:1, etc. Only used for single GPU training."
    )
    parser.add_argument("--num_workers", type=int, default=4,
        help="Number of workers for pytorch dataloader"
    )
    parser.add_argument("--resume_training", action='store_true',
        help="Whether to resume training from last checkpoints"
    )
    parser.add_argument("--only_test", action='store_true',
        help="Whether to skip training and only run model testing"
    )
    parser.add_argument("--world_size", type=int, default=1,
        help="Set larger than 1 to use distributed data parallel training."
    )
    parser.add_argument("--disable_tqdm", action='store_true',
        help="Whether to display tqdm progress bar"
    )
    parser.add_argument("--model_seed", type=int, default=0,
        help="Random seed for init the model"
    )
    parser.add_argument("--eval_pathology", action='store_true',
        help="Eval pathology"
    )
    parser.add_argument("--min_area", type=float, default=None,
        help="min area of pathology"
    )
    parser.add_argument("--max_area", type=float, default=None,
        help="max area of pathology"
    )
    args = parser.parse_args()
    
    return args
    
def main():
    args = build_args()

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
        
    save_dir = config['save_dir']
    trainset_csv = config['trainset_csv']
    testset_csv = config['testset_csv']
    model_name = config['model']
    accl_factors = config['accl_factors']
    num_epochs = config['num_epochs']
    lr = config['lr']
    train_attr = config['train_attr']
    test_attr = config['test_attr']
    checkpoint_epochs = config['checkpoint_epochs']
    augment_data = config['augment_data']
    batch_size = config['batch_size']
    if 'normalize_output_during_testing' in config:
        normalize_output = config['normalize_output_during_testing'] 
    else:
        normalize_output = False
    if 'results_folder' in config:
        results_folder = config['results_folder']
    else:
        results_folder = ''
    if 'adjust_img_size_factor' in config:
        adjust_img_size_factor = config['adjust_img_size_factor'] 
    else:
        adjust_img_size_factor = 1


    if None in test_attr:
        test_attr = train_attr
    train_attr = tuple(train_attr)

    # TODO: ASSERT only viable string input is all
    if checkpoint_epochs == 'all':
        checkpoint_epochs = list(range(1, num_epochs+1))

    finetune_cp = args.finetune
    if args.device is None :
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device 

    num_workers = args.num_workers
    resume_training = args.resume_training
    disable_tqdm = args.disable_tqdm
    only_test = args.only_test
    world_size = args.world_size
    model_seed = args.model_seed
    eval_pathology = args.eval_pathology
    min_area = args.min_area
    max_area = args.max_area

    # Create folder for saving checkpoints and results
    save_dir = os.path.join(save_dir, create_folder_name(train_attr))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Create a file with input arguments
    with open(os.path.join(save_dir, 'arguments.txt'), 'a') as f:
        f.write(time.strftime('%X %x %Z')+', New run:\n')
        f.write(f'model: {model_name}\n')
        f.write(f'finetune: {finetune_cp}\n')
        f.write(f'save_dir: {save_dir}\n')
        f.write(f'traineset file: {trainset_csv}\n')
        f.write(f'testset file: {testset_csv}\n')
        f.write(f'train attributes: {train_attr}\n')
        f.write(f'test attributes: {test_attr}\n')
        f.write(f'accleration factors: {accl_factors}\n')
        f.write(f'training epochs: {num_epochs}\n')
        f.write(f'checkpoint epochs: {checkpoint_epochs}\n')
        f.write(f'model_seed: {model_seed}\n')
        f.write(f'lr: {lr}\n')

    assert batch_size % world_size == 0, 'batch_size must be divisible by world_size'
    assert model_name in flatten(MODEL_NAMES.values()), f'\'{model_name}\' is not implemented.' 

    train_args=(
        world_size,
        device,
        model_name,
        save_dir,
        trainset_csv,
        train_attr,
        finetune_cp,
        num_workers,
        accl_factors,
        num_epochs,
        lr,
        checkpoint_epochs,
        batch_size,
        disable_tqdm,
        augment_data,
        resume_training,
        model_seed,
    )

    if not only_test:
        if world_size == 1:
            train(0, None, *train_args) 

        elif world_size > 1:
            master_port = str(np.random.randint(15000,65535)); print(f"master port: {master_port}")
            args = (master_port, ) + train_args
            mp.spawn(train, args=args, nprocs=world_size, join=True)
            
    test(
        model_name,
        save_dir,
        testset_csv,
        test_attr,
        device = device,
        num_workers = num_workers,
        accl_factors = accl_factors,
        disable_tqdm = disable_tqdm,
        normalize_output = normalize_output,
        results_folder = results_folder,
        adjust_img_size_factor= adjust_img_size_factor,
        eval_pathology = eval_pathology,
        min_area = min_area,
        max_area = max_area,
    )

if __name__ == '__main__':
    main()
    


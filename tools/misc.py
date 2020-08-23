
#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)
# modified Yanxin Hu
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch 
import torch.nn as nn
import numpy as np
import os 
import sys
import soundfile as sf
def read_and_config_file(wave_list, decode=0):
    processed_list = []
    
    with open(wave_list) as fid:
        spk2id_dict = {}
        spk_idx = 0
        if decode:
            for line in fid:
                tmp = line.strip().split()
                sample = {'inputs': tmp[0]}
                processed_list.append(sample)

        else:
            for line in fid:
                tmp = line.strip().split()
                # this is for wsj0-2mix
                name = os.path.basename(tmp[0]).split('_')[0]
                one_sample_spk_idx = [] 

                for idx in range(len(name)//4):
                    spkname = name[idx*4:idx*4+3]
                    if spkname not in spk2id_dict.keys():
                        spk2id_dict[spkname] = spk_idx
                        spk_idx += 1
                    one_sample_spk_idx.append(spk2id_dict[spkname])
                data, fs = sf.read(tmp[0])
                sample = {'inputs': tmp[0], 'labels':tmp[1:], 'duration': data.shape[0], 'spkid': one_sample_spk_idx}
                processed_list.append(sample)
    if spk_idx != 0:
        return processed_list, spk2id_dict, spk_idx
    return processed_list

def load_checkpoint(checkpoint_path, use_cuda):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint

def get_learning_rate(optimizer):
    """Get learning rate"""
    return optimizer.param_groups[0]["lr"]


def reload_for_eval(model, checkpoint_dir, use_cuda):
    best_name = os.path.join(checkpoint_dir, 'best_model')
    ckpt_name = os.path.join(checkpoint_dir, 'checkpoint')
    if os.path.isfile(best_name):
        name = best_name 
    elif os.path.isfile(ckpt_name):
        name = ckpt_name
    else:
        print('Warning: There is no exited checkpoint or best_model!!!!!!!!!!!!')
        return
    with open(name, 'r') as f:
        model_name = f.readline().strip()
    checkpoint_path = os.path.join(checkpoint_dir, model_name)
    checkpoint = load_checkpoint(checkpoint_path, use_cuda)
    model.load_state_dict(checkpoint['model'], strict=False)
    print('=> Reload well-trained model {} for decoding.'.format(
            model_name))



def reload_model(model, optimizer, checkpoint_dir, use_cuda=True, strict=True):
    ckpt_name = os.path.join(checkpoint_dir, 'checkpoint')
    if os.path.isfile(ckpt_name):
        with open(ckpt_name, 'r') as f:
            model_name = f.readline().strip()
        checkpoint_path = os.path.join(checkpoint_dir, model_name)
        checkpoint = load_checkpoint(checkpoint_path, use_cuda)
        model.load_state_dict(checkpoint['model'], strict=strict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        step = checkpoint['step']
        print('=> Reload previous model and optimizer.')
    else:
        print('[!] checkpoint directory is empty. Train a new model ...')
        epoch = 0
        step = 0
    return epoch, step

def save_checkpoint(model, optimizer, epoch, step, checkpoint_dir, mode='checkpoint'):
    checkpoint_path = os.path.join(
        checkpoint_dir, 'model.ckpt-{}.pt'.format(epoch))
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'step': step}, checkpoint_path)

    with open(os.path.join(checkpoint_dir, mode), 'w') as f:
        f.write('model.ckpt-{}.pt'.format(epoch))
    print("=> Save checkpoint:", checkpoint_path)

def setup_lr(opt, lr):
    for param_group in opt.param_groups:
        param_group['lr'] = lr

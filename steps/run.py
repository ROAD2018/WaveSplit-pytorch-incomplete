
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sys
import os
import argparse
import torch.nn.parallel.data_parallel as data_parallel
import numpy as np
import scipy
import scipy.io as sio
import torch.optim as optim
import time
import multiprocessing
sys.path.append(os.path.dirname(sys.path[0]))
sys.path.append(
    os.path.dirname(sys.path[0]) + '/tools/speech_processing_toolbox')

from model.wavesplit import WaveSplit as Model 

from tools.misc import get_learning_rate, save_checkpoint, reload_for_eval, reload_model
from tools.time_dataset import make_loader, Processer, DataReader

import soundfile as sf
import warnings
warnings.filterwarnings("ignore")

class Moniter(object):

    def __init__(self, name,overall_step, print_freq):
        self.name = name
        self.print_freq = print_freq
        self.total_metric = 0.0
        self.recent_metric = 0.0
        self.overall = overall_step 

    def __call__(self, data):
        if isinstance(data, float):
            self.total_metric += data 
            self.recent_metric += data 
        elif isinstance(data, torch.Tensor):
            self.total_metric += data.data.cpu().item()
            self.recent_metric += data.data.cpu().item()
    
    def recent(self): 
        out = '{:s}: {:2.4f}'.format(self.name, self.recent_metric/self.print_freq)
        return out 

    def average(self):
        out = '{:s}: {:2.4f}'.format(self.name, self.total_metric/self.overall)
        return out 
    
    def reset_print(self):
        self.recent_metric = 0.
    
    def ave_float(self):
        return self.total_metric/self.overall
    
    def rec_float(self):
        return self.recent_metric/self.print_freq
    
    def reset(self):
        self.recent_metric = 0.

def train(model, args, device, writer):
    print('preparing data...')
    dataloader, dataset = make_loader(
        args.tr_list,
        args.batch_size,
        num_workers=args.num_threads,
        segment_length=1.5,
        sample_rate=args.sample_rate,
        processer=Processer(
            sample_rate=args.sample_rate,
            ))
    print_freq = 100
    num_batch = len(dataloader)
    params = model.get_params(args.weight_decay)
    optimizer = optim.Adam(params, lr=args.learn_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=1, verbose=True)
    
    if args.retrain:
        start_epoch, step = reload_model(model, optimizer, args.exp_dir,
                                         args.use_cuda)
    else:
        start_epoch, step = 0, 0
    print('---------PRERUN-----------')
    lr = get_learning_rate(optimizer)
    print('(Initialization)')
    val_loss, val_sisnr = 30,30. #validation(model, args, lr, -1, device)
    writer.add_scalar('Loss/Train', val_loss, step)
    writer.add_scalar('Loss/Cross-Validation', val_loss, step)
    
    writer.add_scalar('SISNR/Train', -val_sisnr, step)
    writer.add_scalar('SISNR/Cross-Validation', -val_sisnr, step)

    for epoch in range(start_epoch, args.max_epoch):
        torch.manual_seed(args.seed + epoch)
        if args.use_cuda:
            torch.cuda.manual_seed(args.seed + epoch)
        model.train()
        
        all_moniter = Moniter('All', num_batch, print_freq)
        sdr_moniter = Moniter('SDR', num_batch, print_freq)
        speaker_moniter = Moniter('Speaker', num_batch, print_freq)
        reg_moniter = Moniter('Reg', num_batch, print_freq) 

        stime = time.time()
        lr = get_learning_rate(optimizer)
        for idx, data in enumerate(dataloader):
            inputs, labels, spkid = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            spkid = spkid.to(device)
            model.zero_grad()
            est_wav, speaker_loss, reg_loss = data_parallel(model, (inputs,spkid))
            speaker_loss = torch.mean(speaker_loss)
            reg_loss = torch.mean(reg_loss)
            
            sdr = model.loss(est_wav, labels, loss_mode='SDR')
            all = sdr+2*speaker_loss + 0.3*reg_loss
            all.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            all_moniter(all)
            sdr_moniter(sdr)
            speaker_moniter(speaker_loss)
            reg_moniter(reg_loss)

            step += 1


            if (idx+1) % 3000 == 0:
                save_checkpoint(model, optimizer, -1, step, args.exp_dir)
                #val_loss, val_sisnr= validation(model, args, lr, epoch, device)
                #scheduler.step(val_loss)
                #lr = get_learning_rate(optimizer)
            if (idx + 1) % print_freq == 0:
                eplashed = time.time() - stime
                speed_avg = eplashed / (idx+1)
                log_str = 'Epoch {:3d}/{:3d} | batches {:5d}/{:5d} | lr {:1.4e} |'\
                      '{:2.3f}s/batches |' \
                      ' {:s} |'\
                      ' {:s} |'\
                      ' {:s} |'\
                      ' {:s} |'\
                      ''.format(
                          epoch, args.max_epoch, idx + 1, num_batch, lr,
                          speed_avg, 
                            all_moniter.recent(),
                            sdr_moniter.recent(),
                            speaker_moniter.recent(),
                            reg_moniter.recent()
                          )
                writer.add_scalar('Loss/Train', all_moniter.rec_float(), step)
                writer.add_scalar('SISNR/Train', -sdr_moniter.rec_float(), step)
                print(log_str)
                all_moniter.reset()
                sdr_moniter.reset()
                speaker_moniter.reset()
                reg_moniter.reset()
                sys.stdout.flush()
        
        eplashed = time.time() - stime
        log_str = 'Training AVG.LOSS |' \
            ' Epoch {:3d}/{:3d} | lr {:1.4e} |' \
            ' {:2.3f}s/batch | time {:3.2f}mins |' \
            ' {:s} |'\
            ' {:s} |'\
            ' {:s} |'\
            ' {:s} |'\
            ''.format(
                                    epoch + 1,
                                    args.max_epoch,
                                    lr,
                                    eplashed/num_batch,
                                    eplashed/60.0,
                            all_moniter.average(),
                            sdr_moniter.average(),
                            speaker_moniter.average(),
                            reg_moniter.average()
                        )
        print(log_str)
        val_loss, val_sisnr= validation(model, args, lr, epoch, device)
        writer.add_scalar('Loss/Cross-Validation', val_loss, step)
        writer.add_scalar('SISNR/Cross-Validation', -val_sisnr, step)
        writer.add_scalar('learn_rate', lr, step) 
        if val_loss > scheduler.best:
            print('Rejected !!! The best is {:2.6f}'.format(scheduler.best))
        else:
            save_checkpoint(model, optimizer, epoch + 1, step, args.exp_dir, mode='best_model')
        scheduler.step(val_loss)
        sys.stdout.flush()
        stime = time.time()


def validation(model, args, lr, epoch, device):
    dataloader, dataset = make_loader(
        args.cv_list,
        args.batch_size,
        num_workers=args.num_threads,
        sample_rate=args.sample_rate,
        segment_length=2,
        processer=Processer(
            sample_rate=args.sample_rate,
            ))
    model.eval()
    num_batch = len(dataloader)
    stime = time.time()
    all_moniter = Moniter('All', num_batch, 1)
    sdr_moniter = Moniter('SDR', num_batch, 1)
    speaker_moniter = Moniter('Speaker', num_batch, 1)
    reg_moniter = Moniter('Reg', num_batch, 1) 
    num_batch = len(dataloader)
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            inputs, labels,spkid = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            spkid = spkid.to(device)
            est_wav,speaker_loss,reg_loss = data_parallel(model, (inputs,spkid))
            speaker_loss = torch.mean(speaker_loss)
            reg_loss = torch.mean(reg_loss)
            #gth_spec = data_parallel(model.stft, (labels))[0]
            #loss = model.loss(est_spec, gth_spec, loss_mode='MSE')
            sdr = model.loss(est_wav, labels, loss_mode='SDR')
            all = sdr+2*speaker_loss + 0.3*reg_loss
            all_moniter(all)
            sdr_moniter(sdr)
            speaker_moniter(speaker_loss)
            reg_moniter(reg_loss)

        etime = time.time()
    eplashed = time.time() - stime

    log_str = 'CROSSVAL AVG.LOSS | Epoch {:3d}/{:3d} ' \
          '| lr {:.4e} | {:2.3f}s/batch| time {:2.1f}mins |' \
            ' {:s} |'\
            ' {:s} |'\
            ' {:s} |'\
            ' {:s} |'\
           ''.format(
                        epoch + 1,
                        args.max_epoch,
                        lr,
                        eplashed,
                        (etime - stime)/60.0,
                            all_moniter.average(),
                            sdr_moniter.average(),
                            speaker_moniter.average(),
                            reg_moniter.average()
               )
    print(log_str)
    sys.stdout.flush()
    return all_moniter.ave_float(), sdr_moniter.ave_float()


def decode(model, args, device):
    model.eval()
    with torch.no_grad():
        
        data_reader = DataReader(
            args.tt_list,
            sample_rate=args.sample_rate)
        output_wave_dir = os.path.join(args.exp_dir, 'rec_wav/')
        if not os.path.isdir(output_wave_dir):
            os.mkdir(output_wave_dir)
        num_samples = len(data_reader)
        print('Decoding...')
        for idx in range(num_samples):
            inputs, utt_id, nsamples = data_reader[idx]
            
            inputs = torch.from_numpy(inputs)
            inputs = inputs.to(device)
            
            outputs = model(inputs, training=False)
            if outputs.dim() != 1:
                outputs = torch.squeeze(outputs).cpu().data.numpy()
            else:
                outputs = outputs.cpu().data.numpy()
            if len(outputs.shape) > 1:
                outputs = outputs.T
            sf.write(os.path.join(output_wave_dir, utt_id), outputs, args.sample_rate) 

        print('Decode Done!!!')


def main(args):
    device = torch.device('cuda' if args.use_cuda else 'cpu')
    args.sample_rate = {
        '8k':8000,
        '16k':16000,
        '24k':24000,
        '48k':48000,
    }[args.sample_rate]
    model = Model(
        overallspks=105,
    )
    if not args.log_dir:
        writer = SummaryWriter(os.path.join(args.exp_dir, 'tensorboard'))
    else:
        writer = SummaryWriter(args.log_dir)
    model.to(device)
    if not args.decode:
        train(model, FLAGS, device, writer)
    reload_for_eval(model, FLAGS.exp_dir, FLAGS.use_cuda)
    decode(model, args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('PyTorch Version ENhancement')
    parser.add_argument('--decode', type=int, default=0, help='if decode')
    parser.add_argument(
        '--exp-dir',
        dest='exp_dir',
        type=str,
        default='exp/cldnn',
        help='the exp dir')
    parser.add_argument(
        '--tr-list', dest='tr_list', type=str, help='the train data list')
    parser.add_argument(
        '--cv-list',
        dest='cv_list',
        type=str,
        help='the cross-validation data list')
    parser.add_argument(
        '--tt-list', dest='tt_list', type=str, help='the test data list')
    parser.add_argument(
        '--rnn-layers',
        dest='rnn_layers',
        type=int,
        default=2,
        help='the num hidden rnn layers')
    parser.add_argument(
        '--rnn-units',
        dest='rnn_units',
        type=int,
        default=512,
        help='the num hidden rnn units')
    parser.add_argument(
        '--learn-rate',
        dest='learn_rate',
        type=float,
        default=0.001,
        help='the learning rate in training')
    parser.add_argument(
        '--max-epoch',
        dest='max_epoch',
        type=int,
        default=20,
        help='the max epochs')

    parser.add_argument(
        '--dropout',
        dest='dropout',
        type=float,
        default=0.2,
        help='the probility of dropout')
    parser.add_argument(
        '--left-context',
        dest='left_context',
        type=int,
        default=1,
        help='the left context to add')
    parser.add_argument(
        '--right-context',
        dest='right_context',
        type=int,
        default=1,
        help='the right context to add')
    parser.add_argument(
        '--input-dim',
        dest='input_dim',
        type=int,
        default=257,
        help='the input dim')
    parser.add_argument(
        '--output-dim',
        dest='output_dim',
        type=int,
        default=257,
        help='the output dim')
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        type=int,
        help='the batch size in train')
    parser.add_argument(
        '--use-cuda', dest='use_cuda', default=1, type=int, help='use cuda')
    parser.add_argument(
        '--seed', dest='seed', type=int, default=20, help='the random seed')
    parser.add_argument(
        '--log-dir',
        dest='log_dir',
        type=str,
        default=None,
        help='the random seed')
    parser.add_argument(
        '--num-threads', dest='num_threads', type=int, default=10)
    parser.add_argument(
        '--window-len',
        dest='win_len',
        type=int,
        default=400,
        help='the window-len in enframe')
    parser.add_argument(
        '--window-inc',
        dest='win_inc',
        type=int,
        default=100,
        help='the window include in enframe')
    parser.add_argument(
        '--fft-len',
        dest='fft_len',
        type=int,
        default=512,
        help='the fft length when in extract feature')
    parser.add_argument(
        '--window-type',
        dest='win_type',
        type=str,
        default='hamming',
        help='the window type in enframe, include hamming and None')
    parser.add_argument(
        '--kernel-size',
        dest='kernel_size',
        type=int,
        default=6,
        help='the kernel_size')
    parser.add_argument(
        '--kernel-num',
        dest='kernel_num',
        type=int,
        default=9,
        help='the kernel_num')
    parser.add_argument(
        '--num-gpu',
        dest='num_gpu',
        type=int,
        default=1,
        help='the num gpus to use')
    parser.add_argument(
        '--target-mode',
        dest='target_mode',
        type=str,
        default='MSA',
        help='the type of target, MSA, PSA, PSM, IBM, IRM...')
    
    parser.add_argument(
        '--weight-decay', dest='weight_decay', type=float, default=0.00001)
    parser.add_argument(
        '--clip-grad-norm', dest='clip_grad_norm', type=float, default=5.)
    parser.add_argument(
        '--sample-rate', dest='sample_rate', type=str, default='16k')
    parser.add_argument('--retrain', dest='retrain', type=int, default=0)
    FLAGS, _ = parser.parse_known_args()
    FLAGS.use_cuda = FLAGS.use_cuda and torch.cuda.is_available()
    os.makedirs(FLAGS.exp_dir, exist_ok=True)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    #torch.backends.cudnn.benchmark = True
    if FLAGS.use_cuda:
        torch.cuda.manual_seed(FLAGS.seed)
    import pprint
    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS.__dict__)
    print(FLAGS.win_type)
    main(FLAGS)

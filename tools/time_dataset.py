#!/usr/bin/env python
# coding=utf-8
import numpy as np
import scipy
import torch 
import random
import torch.nn as nn
from torch.utils.data import Dataset
import torch.utils.data as tud
import os 
import sys
sys.path.append(
    os.path.dirname(__file__))
from misc import read_and_config_file
import multiprocessing as mp
from utils import audioread, audiowrite

class DataReader(object):
    def __init__(self, file_name, sample_rate=16000):
        self.file_list = read_and_config_file(file_name, decode=True)
        self.sample_rate = sample_rate 

    def extract_feature(self, path):
        path = path['inputs']
        utt_id = path.split('/')[-1]
        data, fs= audioread(path) 
        if fs != self.sample_rate:
            raise Warning("file {:s}'s sample rate is not match {:d}!".format(path, self.sample_rate)) 
        inputs = np.reshape(data, [1, data.shape[0]]).astype(np.float32)
        
        return inputs, utt_id, data.shape[0]
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        return self.extract_feature(self.file_list[index])

def clip_data(data, start, segment_length):
    data_len = data.shape[0]
    shape = list(data.shape)
    shape[0] = segment_length
    tgt = np.zeros(shape)
    if start == -2:
        # this means  segment_length//4 < data_len < segment_length//2
        # padding to A_A_A
        if data_len < segment_length//3:
            data = np.pad(data, [0,segment_length//3-data_len])
            tgt[:segment_length//3] += data 
            st = segment_length//3
            tgt[st:st+data.shape[0]] += data
            st = segment_length//3*2
            tgt[st:st+data.shape[0]] = data

        else:
            st = (segment_length//2-data_len)%101
            tgt[st:st+data_len] += data
            st = segment_length//2+(segment_length//2-data_len)%173
            tgt[st:st+data_len] += data

    elif start == -1:
        # this means  segment_length < data_len*2
        # padding to A_A 
        if data_len %4 == 0:
            tgt[:data_len] += data
            
            tgt[data_len:] += data[:segment_length-data_len]
        elif data_len %4 == 1:
            tgt[:data_len] += data
        elif data_len %4 == 2:
            tgt[-data_len:] += data
        elif data_len %4 == 3:
            tgt[(segment_length-data_len)//2:(segment_length-data_len)//2+data_len] += data

    else:
        # this means  segment_length < data_len
        if tgt.shape[0] != data[start:start+segment_length].shape[0]:
            start = data.shape[0] - segment_length
        tgt += data[start:start+segment_length]
    return tgt

class Processer(object):
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def process(self, path, start_time, segment_length):
        inputs_path = path['inputs']
        spkid = path['spkid']
        labels_path = path['labels']
        data,fs = audioread(inputs_path)
        assert(fs == self.sample_rate)
        inputs = clip_data(data, start_time, segment_length)
        labels = []
        for item in labels_path:
            data,fs = audioread(item)
            labels.append( clip_data(data, start_time, segment_length))
        return inputs, np.stack(labels,0), spkid 


class TFDataset(Dataset):

    def __init__(
            self,
            scp_file_name,
            segment_length=0.75,
            sample_rate=16000,
            processer=Processer(),
            gender2spk=None
        ):
        '''
            scp_file_name: the list include:[input_wave_path, output_wave_path, duration]
            spk_emb_scp: a speaker embedding ark's scp 
            segment_length: to clip data in a fix length segment, default: 4s
            sample_rate: the sample rate of wav, default: 16000
            processer: a processer class to handle wave data 
            gender2spk: a list include gender2spk, default: None
        '''
        self.wav_list, self.spk2id, self.overallspks = read_and_config_file(scp_file_name)
        print(self.overallspks)
        self.processer = processer
        mgr = mp.Manager()
        self.index =mgr.list()#[d for b in buckets for d in b]
        self.segment_length = int(segment_length * sample_rate)
        self._dochunk(SAMPLE_RATE=sample_rate)


    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_info, start_time = self.index[index]
        inputs, labels, spkid = self.processer.process(data_info, start_time, self.segment_length)
        return inputs, labels, spkid

    def _dochunk(self, SAMPLE_RATE=16000, num_threads=12):
        # mutliproccesing
        def worker(target_list, result_list, start, end, segment_length, SAMPLE_RATE):
            for item in target_list[start:end]:
                duration = item['duration']
                length = duration
                if length < segment_length:
                    if length * 2 < segment_length:
                        continue
                    result_list.append([item, -1])
                else:
                    sample_index = 0
                    while sample_index + segment_length < length:
                        result_list.append(
                            [item, sample_index])
                        sample_index += segment_length
                    if sample_index != length - 1:
                        result_list.append([
                            item,
                            int(length - segment_length),
                        ])
        pc_list = []
        stride = len(self.wav_list) // num_threads
        if stride < 100:
            p = mp.Process(
                            target=worker,
                            args=(
                                    self.wav_list,
                                    self.index,
                                    0,
                                    len(self.wav_list),
                                    self.segment_length,
                                    SAMPLE_RATE,
                                )
                        )
            p.start()
            pc_list.append(p)
        else: 
            for idx in range(num_threads):
                if idx == num_threads-1:
                    end = len(self.wav_list)
                else:
                    end = (idx+1)*stride
                p = mp.Process(
                                target=worker,
                                args=(
                                    self.wav_list,
                                    self.index,
                                    idx*stride,
                                    end,
                                    self.segment_length,
                                    SAMPLE_RATE,
                                )
                            )
                p.start()
                pc_list.append(p)
        for p in pc_list:
            p.join()
            p.terminate()

class Sampler(tud.sampler.Sampler):
    '''
     
    '''
    def __init__(self, data_source, batch_size):
        it_end = len(data_source) - batch_size + 1
        self.batches = [range(i, i+batch_size)
                        for i in range(0, it_end, batch_size)]
        self.data_source = data_source
        
    def __iter__(self):
        random.shuffle(self.batches)
        return (i for b in self.batches for i in b)

    def __len__(self):
        return len(self.data_source)



def collate_fn(data):
    inputs, labels, spkid = zip(*data)
    inputs = np.array(inputs, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    spkid = np.array(spkid, dtype=np.int32)
    return torch.from_numpy(inputs), torch.from_numpy(labels), torch.from_numpy(spkid).long()

def make_loader(scp_file_name, batch_size, segment_length=0.75,num_workers=12, sample_rate=8000, processer=Processer(sample_rate=8000)):
    dataset = TFDataset(scp_file_name, segment_length, sample_rate=sample_rate,processer=processer)
    sampler = Sampler(dataset, batch_size)
    loader = tud.DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=collate_fn,
                            sampler=sampler,
                            drop_last=False
                        )
                            #shuffle=True,
    return loader, None #, Dataset
if __name__ == '__main__':
    #laoder,_ = make_loader('../data/tt.lst', 32, num_workers=16)
    #laoder,_ = make_loader('../data/cv.lst', 32, num_workers=16)
    laoder,_ = make_loader('../debug/tr.lst', 32, num_workers=16)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    import time 
    import soundfile as sf
    stime = time.time()

    for epoch in range(10):
        for idx, data in enumerate(laoder):
            inputs, labels, spkid= data 
            #inputs.cuda()
            #labels.cuda()
            if idx%100 == 0:
                etime = time.time()
                #print(epoch, idx, inputs.size(), labels.size(), spkid.size(), (etime-stime)/100)
                sf.write('labels{:d}.wav'.format(idx),labels[0].numpy().T,8000)
                stime = etime

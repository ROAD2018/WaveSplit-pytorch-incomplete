'''

for eval the model, pesq, stoi, si-sdr

need to install pypesq: 
https://github.com/vBaiCai/python-pesq

pystoi:
https://github.com/mpariente/pystoi
si-sdr:
kewang
'''

import soundfile as sf
from pypesq import pesq
import multiprocessing as mp
import argparse
from pystoi.stoi import stoi
import numpy as np 
import os 
import itertools 

os.environ['OMP_NUM_THREADS'] = '2'

def audioread(path, mono=False, fs=16000):
    wave_data, sr = sf.read(path)
    assert fs == sr
    if mono and len(wave_data.shape) > 2:
        if wave_data.shape[1] == 1:
            wave_data = wave_data[0]
        else:
            wave_data = np.mean(wave_data, axis=-1)
    return wave_data, fs

def remove_dc(signal):
    """Normalized to zero mean"""
    mean = np.mean(signal)
    signal -= mean
    return signal


def pow_np_norm(signal):
    """Compute 2 Norm"""
    return np.square(np.linalg.norm(signal, ord=2, axis=-1,keepdims=True))


def pow_norm(s1, s2):
    return np.sum(s1 * s2,axis=-1,keepdims=True)


def si_sdr(estimated, original):
    estimated = remove_dc(estimated)
    original = remove_dc(original)
    target = pow_norm(estimated, original) * original / pow_np_norm(original)
    noise = estimated - target
    return 10 * np.log10(pow_np_norm(target) / pow_np_norm(noise))


def sdr(estimated, original):
    estimated = remove_dc(estimated)
    original = remove_dc(original)

    noise = estimated - original
    return 10 * np.log10(pow_np_norm(original) / pow_np_norm(noise))


def pit(func, est, ref):
    
    assert ref.shape == est.shape
    ref = ref[:,None]
    est = est[None,:]
    nums = ref.shape[0]
    score = func(est,ref)
    tmp = itertools.permutations([ x for x in range(nums)])
    perm = np.array([ x for x in tmp] )
    index = np.eye(nums)[perm]
    pit_score = np.einsum("pnm,nm->p", index, score[...,0])
    score = np.max(pit_score)/nums
     
    return score


def test():
    ones = np.ones(16000)
    zeros = np.zeros(16000)
    ref = np.stack([ones,zeros],-1)
    est = np.stack([zeros,ones],-1)+1e-3
    
    loss = pit(sdr, est, ref)
    print(loss)


def eval(ref, enh, nsy, utt_id,results):
    try:
        fs=8000
        s1, sr = audioread(os.path.join(ref,'s1',utt_id),fs=fs)
        s2, sr = audioread(os.path.join(ref,'s2',utt_id),fs=fs)
        ref = np.stack([s1,s2])
        enh, sr = audioread(os.path.join(enh,utt_id),fs=fs)
        nsy, sr = audioread(os.path.join(nsy,utt_id),fs=fs)
        nsy = np.stack([nsy,nsy])
        enh = enh.T
        enh_len = enh.shape[1]
        ref_len = ref.shape[1]
        if enh_len > ref_len:
            enh = enh[:,:ref_len]
        else:
            ref = ref[:,:enh_len]
            nsy = nsy[:,:enh_len]
        ref_score = 0.#pesq(ref, nsy, sr)
        enh_score = 0.#pesq(ref, enh, sr)
        ref_stoi = 0.#stoi(ref, nsy, sr, extended=False)
        enh_stoi = 0.#stoi(ref, enh, sr, extended=False)
        ref_sisdr = pit(si_sdr, nsy, ref)
        enh_sisdr = pit(si_sdr, enh, ref)
        ref_sdr = pit(sdr, nsy, ref)
        enh_sdr = pit(sdr, enh, ref)

    except Exception as e:
        print(e)
    
    results.append([utt_id, 
                    {'pesq':[ref_score, enh_score],
                     'stoi':[ref_stoi,enh_stoi],
                     'si_sdr':[ref_sisdr, enh_sisdr],
                     'sdr':[ref_sdr, enh_sdr],
                    }])

def main(args):
    pathe=args.pathe#'/home/work_nfs3/yxhu/workspace/se-cldnn-torch/exp/cldnn_2_1_1_0.0005_16k_6_9/rec_wav/'
    pathc=args.pathc#'/home/work_nfs2/yxhu/data/test3000_new_data_noisy/clean/'
    pathn=args.pathn#'/home/work_nfs2/yxhu/data/test3000_new_data_noisy/wav/'
    
    pool = mp.Pool(args.num_threads)
    mgr = mp.Manager()
    results = mgr.list()
    with open(args.result_list, 'w') as wfid:
        with open(args.wav_list) as fid:
            for line in fid:
                name = line.strip().split()[0].split('/')[-1]
                pool.apply_async(
                    eval,
                    args=(
                        pathc,
                        pathe,
                        pathn,
                        name,
                        results,
                    )
                    )
        pool.close()
        pool.join()
        for eval_score in results:
            utt_id, score = eval_score
            pesq = score['pesq']
            stoi = score['stoi']
            si_sdr = score['si_sdr']
            sdr = score['sdr']
            wfid.writelines(
                    '{:s},{:.3f},{:.3f}, '.format(utt_id, pesq[0],pesq[1])
                )
            wfid.writelines(
                    '{:.3f},{:.3f}, '.format(stoi[0],stoi[1])
                )
            wfid.writelines(
                    '{:.3f},{:.3f}, '.format(si_sdr[0],si_sdr[1])
                )
            wfid.writelines(
                    '{:.3f},{:.3f}\n '.format(sdr[0],sdr[1])
                )


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--wav_list',
        type=str,
        default='../data/tt.lst'
        ) 
    
    parser.add_argument(
        '--result_list',
        type=str,
        default='result_list'
        ) 
    
    parser.add_argument(
        '--num_threads',
        type=int,
        default=6
        )
    parser.add_argument(
        '--pathe',
        type=str,
        default='../exp/debug_2_WaveSplit/tmp_test/'
        )
    parser.add_argument(
        '--pathc',
        type=str,
        default='/dockerdata/yanxinhu_data/2speakers/wav8k/min/tt/'
        )
    parser.add_argument(
        '--pathn',
        type=str,
        default='/dockerdata/yanxinhu_data/2speakers/wav8k/min/tt/mix/'
        )
    args = parser.parse_args()
    main(args)
    #test()

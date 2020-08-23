#!/bin/bash 
input_dim=257
output_dim=257
left_context=0
right_context=0
lr=0.001




win_len=320
win_inc=160
fft_len=320

win_len=400
win_inc=100
fft_len=512

sample_rate=16k
win_type=hanning
batch_size=4
max_epoch=45
rnn_units=512
rnn_layers=2
tt_list=data/tt.lst
tr_list=data/tr.lst
cv_list=data/cv.lst


tt_list=data/t
tt_list='data/test_828.lst'
tr_list='data/tr_28.lst'
cv_list='data/cv_28.lst'
tr_list='data/tr_56.lst'
cv_list='data/cv_56.lst'

tt_list='data/test_wsj0_0.lst'

tt_list='data/test_sogou_1spk.lst'
tt_list='data/test_sogou_max.lst'

tt_list=data/t

cv_list='data/cv_wsj0_-5~20.lst'
tr_list='data/tr_wsj0_-5~20.lst'




tr_list=debug/tr.lst
cv_list=debug/cv.lst

tr_list=data/tr.lst
cv_list=data/cv.lst
tt_list='data/tt.lst'
tt_list=debug/t

dropout=0.2
kernel_size=6
kernel_num=9
nropout=0.2
retrain=1
sample_rate=8k
num_gpu=1
batch_size=$[num_gpu*batch_size]

save_name='debug_2_WaveSplit'

exp_dir=exp/${save_name}
if [ ! -d ${exp_dir} ] ; then
    mkdir -p ${exp_dir}
fi

stage=2

if [ $stage -le 1 ] ; then
    CUDA_VISIBLE_DEVICES='6' nohup python -u ./steps/run.py \
    --decode=0 \
    --fft-len=${fft_len} \
    --input-dim=${input_dim} \
    --output-dim=${output_dim} \
    --window-len=${win_len} \
    --window-inc=${win_inc} \
    --exp-dir=${exp_dir} \
    --tr-list=${tr_list} \
    --cv-list=${cv_list} \
    --tt-list=${tt_list} \
    --retrain=${retrain} \
    --rnn-layers=${rnn_layers} \
    --rnn-units=${rnn_units} \
    --learn-rate=${lr} \
    --max-epoch=${max_epoch} \
    --dropout=${dropout} \
    --input-dim=${input_dim} \
    --output-dim=${output_dim} \
    --left-context=${left_context} \
    --right-context=${right_context} \
    --batch-size=${batch_size} \
    --kernel-size=${kernel_size} \
    --kernel-num=${kernel_num} \
    --sample-rate=${sample_rate} \
    --target-mode=${target_mode} \
    --weight-decay=0. \
    --window-type=${win_type} > ${exp_dir}/train.log &
    exit 0
fi

if [ $stage -le 2 ] ; then 
    CUDA_VISIBLE_DEVICES='' python -u ./steps/run.py \
    --decode=1 \
    --fft-len=${fft_len} \
    --input-dim=${input_dim} \
    --output-dim=${output_dim} \
    --window-len=${win_len} \
    --window-inc=${win_inc} \
    --exp-dir=${exp_dir} \
    --tr-list=${tr_list} \
    --cv-list=${cv_list} \
    --tt-list=${tt_list} \
    --retrain=${retrain} \
    --rnn-layers=${rnn_layers} \
    --rnn-units=${rnn_units} \
    --learn-rate=${lr} \
    --max-epoch=${max_epoch} \
    --dropout=${dropout} \
    --input-dim=${input_dim} \
    --output-dim=${output_dim} \
    --left-context=${left_context} \
    --right-context=${right_context} \
    --batch-size=${batch_size} \
    --kernel-size=${kernel_size} \
    --kernel-num=${kernel_num} \
    --sample-rate=${sample_rate} \
    --target-mode=${target_mode} \
    --window-type=${win_type} 
    exit 0
fi

if [ $stage -le 3 ] ; then

for snr in -5 0 5 10 15 20 ; do 
    #dataset_name=aishell
    dataset_name=wsj0
    tgt=crn_${target_mode}_${dataset_name}_${snr}db
    clean_wav_path="/search/odin/huyanxin/workspace/se-resnet/data/wavs/test_wsj0_clean_${snr}/"
    noisy_wav_path="/search/odin/huyanxin/workspace/se-resnet/data/wavs/test_wsj0_noisy_${snr}/"
    #clean_wav_path="/search/odin/huyanxin/workspace/se-cldnn-torch/data/wavs/test_${dataset_name}_clean_${snr}/"
    #noisy_wav_path="/search/odin/huyanxin/workspace/se-cldnn-torch/data/wavs/test_${dataset_name}_noisy_${snr}/"

    enh_wav_path=${exp_dir}/test_${dataset_name}_noisy_${snr}/
    find ${noisy_wav_path} -iname "*.wav" > wav.lst
    CUDA_VISIBLE_DEVICES='4' python -u ./steps/run_crn.py \
    --decode=1 \
    --fft-len=${fft_len} \
    --input-dim=${input_dim} \
    --output-dim=${output_dim} \
   --window-len=${win_len} \
    --window-inc=${win_inc} \
    --exp-dir=${exp_dir} \
    --tr-noise=${tr_noise_list} \
    --tr-clean=${tr_clean_list} \
    --cv-noise=${cv_noise_list} \
    --cv-clean=${cv_clean_list} \
    --tt-list=wav.lst \
    --retrain=${retrain} \
    --rnn-layers=${rnn_layers} \
    --rnn-units=${rnn_units} \
    --learn-rate=${lr} \
    --max-epoch=${max_epoch} \
    --dropout=${dropout} \
    --input-dim=${input_dim} \
    --output-dim=${output_dim} \
    --left-context=${left_context} \
    --right-context=${right_context} \
    --target-mode=${target_mode} \
    --batch-size=${batch_size} \
    --kernel-size=${kernel_size} \
    --sample-rate=${sample_rate} \
    --window-type=${win_type}  || exit 1 # > ${exp_dir}/train.log &
    mv ${exp_dir}/rec_wav ${enh_wav_path}
    
    ls $noisy_wav_path > t
    python ./tools/eval_objective.py --wav_list=t --result_list=${tgt}.csv --pathe=${enh_wav_path}\
    --pathc=${clean_wav_path} --pathn=${noisy_wav_path} ||exit 1
done

fi

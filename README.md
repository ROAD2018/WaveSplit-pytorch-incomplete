# WaveSplit-pytorch-incomplete
need more time and computing resource for construction 

# Attention !
This is a _incomplete_ pytorch version WaveSplit implemention 

The whole model's param is ~65M, and the original version can not training one sample even in P40...
So I added a encoder (conv1d layer) and a decoder to reduce the feature's lenght...
Moreover, the mapping model was replaced by masking...
I haved trained a model on wsj0-2mix, but the training was too slowly and 13 epoch model's sdr is ~10 dB...
Finally, I gave up it ... 
Maybe I will continue construction in future ...

If you have any questions or advices, please issue or mail arrowhyx@foxmail.com

# List
- [x] main model 
- [x] Speaker loss 
- [x] kmeans 
- [x] Gauss Layer but not used now
- [x] Dropout Layer but not used now 
- [ ] Mixup Layer 
- [ ] Dynamic mixing 



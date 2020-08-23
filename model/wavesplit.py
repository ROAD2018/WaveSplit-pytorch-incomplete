
''' 
Written by yxhu@NPU-ASLP in Tencent AiLAB on 2020.8
arrowhyx@foxmail.com
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations
import os
import sys
sys.path.append(os.path.dirname(__file__))
from kmeans import KMeans
from show import show_params 
import numpy as np

class LayerNorm1d(nn.LayerNorm):
        
    def __init__(self, *args, **kwargs):
        super(LayerNorm1d,self).__init__(*args, **kwargs)

    
    def forward(self, inputs):
        if inputs.dim() != 3:
            raise RuntimeError("Expecting the inputs is a 3D tensor")
        inputs = torch.transpose(inputs,1,2)
        out = super().forward(inputs)
        out = torch.transpose(out,1,2)
        return out

class DConv1d(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, dilation=1,norm='LN',causal=False,residual=True):
    
        super(DConv1d,self).__init__()
        
        if causal:
            self.pad = nn.ConstantPad1d([(kernel_size-1)*dilation,0],0)
        else:
            self.pad = nn.ConstantPad1d([(kernel_size-1)//2*dilation, (kernel_size-1)//2*dilation],0)
             
        self.conv = nn.Conv1d(in_channels, out_channels,kernel_size,dilation=dilation)
        self.residual = residual 
        
        if norm == 'LN':
            self.norm = LayerNorm1d(out_channels)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(out_channels)
        
        self.nl = nn.PReLU()

    def forward(self, inputs):
        
        out = self.pad(inputs)
        out = self.conv(out)
        out = self.nl(out)
        out = self.norm(out)
        if self.residual: 
            out = out+inputs
        return out
    
class ConditionDConv1d(nn.Module):
    '''

    '''
    def __init__(self, condition_dim=512, in_channels=1, out_channels=1, kernel_size=3, dilation=1,norm='LN',causal=False,residual=True):
        super(ConditionDConv1d, self).__init__()
        
        self.residual = residual
        self.in_channels = in_channels
        self.lin1 = nn.Sequential(
                nn.Conv1d(condition_dim, self.in_channels,kernel_size=1),
                nn.Tanh(),
            )
        self.lin2 = nn.Conv1d(condition_dim, self.in_channels,kernel_size=1)
        self.conv = nn.Conv1d(in_channels, out_channels,kernel_size,dilation=dilation)
        if causal:
            self.pad = nn.ConstantPad1d([(kernel_size-1)*dilation,0],0)
        else:
            self.pad = nn.ConstantPad1d([(kernel_size-1)//2*dilation, (kernel_size-1)//2*dilation],0)
        if norm == 'LN':
            self.norm = LayerNorm1d(out_channels)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(out_channels)
        self.nl = nn.PReLU()

    
    def forward(self, inputs, cond):
        if cond.dim() != 3 and cond.dim() != 2:
            raise RuntimeError("Expecting the inputs is a 2D/3D tensor")
        if cond.dim() == 2:
            cond = torch.unsqueeze(cond, -1)
        out = self.pad(inputs)
        out = self.conv(out)
        gamma = self.lin1(cond)
        beta = self.lin2(cond)
        out = gamma*out+beta
        out = self.nl(out)
        out = self.norm(out)
        
        if self.residual:
            out += inputs
        return out 

class SpeakerStack(nn.Module):
    
    def __init__(self, in_dim, out_dim, num_layers=14, latent_dim=512, kernel_size=3, numspks=2):
        super(SpeakerStack, self).__init__()
        
        self.layers = nn.ModuleList()
        self.numspks = numspks
        for idx in range(num_layers):
            self.layers.append(
                DConv1d(in_dim if idx == 0 else latent_dim,
                        latent_dim if idx != num_layers-1 else numspks*out_dim,
                        kernel_size=kernel_size,
                        dilation=(2**idx),
                        norm='LN',
                        residual=True if idx != num_layers-1 else False,
                       )
                )

    def forward(self, inputs):
        out = inputs
        for net in self.layers:
            out = net(out)
        b,d,t = out.size()
        out = torch.reshape(out,[b, self.numspks, d//self.numspks, t])
        norm = torch.norm(out,dim=(1,2,3), p=2,keepdim=True) 
        return out/norm

class GaussLayer(nn.Module):

    def __init__(self, mean=0., std=0.2):
        super(GaussLayer, self).__init__()

        self.std = std
        self.mean = mean
    
    def forward(self, inputs):
        
        if self.training:
            noise = torch.randn(inputs.size(), device=inputs.device, dtype=torch.float32)*self.std+self.mean
            inputs = inputs + noise
        return inputs

class DropLayer(nn.Module):

    def __init__(self, droprate=0.4):
        super(DropLayer, self).__init__()
        self.drop = droprate

    def forward(self, inputs): 
        if inputs.dim() != 3:
            raise RuntimeError("Expecting the inputs is a 3D tensor")
        b,n,d = inputs.size()
        
        if self.training:  
            prob = np.random.uniform(size=[b,1])
            prob_mask = torch.tensor(np.array(prob<self.drop), dtype=torch.float32)
            neg_idx = (torch.ones([b,n], dtype=torch.float32)*(1-prob_mask)).to(inputs.device)
            idx = F.one_hot(torch.tensor(np.random.choice(n,size=b)), n).float().to(inputs.device)
            mask = torch.where(prob_mask>0, idx, neg_idx).unsqueeze(-1)
            inputs = inputs*mask
        return inputs


class MixupLayer(nn.Module):

    def __init__(self, mixrate=0.5):
        super(MixupLayer, self).__init__()
        self.mixrate = 0.5
        
        self.beta = 1.#torch.

    def forward(self, inputs):
        pass


class SeparationStack(nn.Module):
    
    def __init__(self, in_dim,out_dim,num_stack=4, stack_size=10, latent_dim=512,kernel_size=3):
        
        super(SeparationStack, self).__init__()
        self.out_dim = out_dim 
        self.latent_dim = latent_dim 
        self.layers = nn.ModuleList()
        for idx in range(stack_size*num_stack):
            self.layers.append(
                ConditionDConv1d(
                        in_channels=in_dim if idx == 0 else latent_dim,
                        out_channels=latent_dim if idx != stack_size*num_stack-1 else out_dim,
                        kernel_size=kernel_size,
                        dilation=int(2**(idx%stack_size)),
                        norm='LN',
                        residual=True if idx != stack_size*num_stack-1 else False,
                       )
                )
        
    def forward(self, inputs, cond):
        out = inputs
        for net in self.layers:
            out = net(out, cond)
        return out 



class WaveSplit(nn.Module):

    def __init__(self, spk_layers=14, sep_stack_size=10, sep_stack_num=4, latent_dim=512, numspks=2, overallspks=3, with_filter=True):
        super(WaveSplit, self).__init__() 
        self.with_filter = with_filter 
        if with_filter:
            self.encoder = nn.Conv1d(1,latent_dim,kernel_size=16,stride=8,
                                        bias=False,
                                         )
            nn.init.normal_(self.encoder.weight) 
            self.decoder = nn.ConvTranspose1d(
                    latent_dim,1,
                    kernel_size=16,stride=8,
                    bias=False,
                )
            nn.init.normal_(self.decoder.weight) 

            self.speaker =  SpeakerStack(in_dim=latent_dim,out_dim=latent_dim, num_layers=spk_layers,numspks=numspks,latent_dim=latent_dim)
            self.separation = SeparationStack(in_dim=latent_dim, out_dim=latent_dim, num_stack=sep_stack_num, stack_size=sep_stack_size, latent_dim=latent_dim)
        else:
            self.speaker =  SpeakerStack(in_dim=1,out_dim=latent_dim,num_layers=spk_layers,numspks=numspks,latent_dim=latent_dim)
            self.separation = SeparationStack(in_dim=1,out_dim=1,num_stack=sep_stack_num, stack_size=sep_stack_size, latent_dim=latent_dim) 

        self.kmean = KMeans(1,numspks, latent_dim, iter_nums=80)
        self.numspks = numspks
        self.loss_func = SpeakerLoss(latent_dim, numspks=numspks, overallspks=overallspks)
        self.latent_dim = latent_dim
        show_params(self)

    def forward(self, inputs, spkid=None, training=True):
        if inputs.dim() == 2:
            inputs = torch.unsqueeze(inputs,1)
        if self.with_filter:
            inputs = self.encoder(inputs)
        #'''
        emb = self.speaker(inputs)
        spk_loss = None 
        reg = None

        if spkid is not None:
            spk_loss, center, reg = self.loss_func(emb, spkid)
         
        if not training:
            center = self.kmean(emb)

        #'''
        b,_,_ = inputs.size()
        outs = []
        #out = F.sigmoid(out)
        #out_c = torch.chunk(out, self.numspks, 1)
        for idx in range(self.numspks):
            out = self.separation(inputs, center[:,idx])
            mask = F.relu(out)

            if self.with_filter:
                out = self.decoder(inputs*mask)
            out = torch.clamp(out,-1,1)
            outs.append(out)
        outs = torch.cat(outs,1)

        if spkid is None:
            return outs
        else:
            return outs, spk_loss, reg 

    def get_params(self, weight_decay=0.0):
            # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
                     'params': weights,
                     'weight_decay': weight_decay,
                 }, {
                     'params': biases,
                     'weight_decay': 0.0,
                 }]
        
        return params
    
    def loss(self, est, labels, loss_mode='SDR'):
        '''
        mode == 'SiSNR'
            est: [B, N, T]
            labels: [B,N, T]
        mode == 'SDR'
            est: [B, N, T]
            labels: [B,N, T]
        '''
        if loss_mode == 'SiSNR':
            '''
            with pit

            '''
            return -torch.mean(pit(est, labels))

        if loss_mode == 'SDR':
            return -torch.mean(sdr(est, labels))
        if loss_mode == 'SNR':
            return -torch.mean(sdr(est, labels))


def l2_norm(s1, s2):
    
    norm = torch.sum(s1*s2, -1, keepdim=True)
    return norm 

def si_snr(s1, s2, eps=1e-8):
    #s1 = remove_dc(s1)
    #s2 = remove_dc(s2)
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target =  s1_s2_norm/(s2_s2_norm+eps)*s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10*torch.log10((target_norm)/(noise_norm+eps)+eps)

    return snr

def pit(inputs, labels):
    numspks = inputs.size(1)
    
    perm = permutations([x for x in range(numspks)])
    
    inputs = torch.unsqueeze(inputs,2)
    labels = torch.unsqueeze(labels,1)
    
    # [B,N,N]
    snr = si_snr(inputs, labels).squeeze(-1)
    
    # [N!,N,N]
    index = F.one_hot(torch.tensor(list(perm)), numspks).float().to(inputs.device)
    loss = torch.einsum('bmn,pmn->bp',snr, index) 
    loss,_ = torch.max(loss,-1)
    return torch.mean(loss/numspks)

def sdr(s1, s2, eps=1e-8):
    #s1 = remove_dc(s1)
    #s2 = remove_dc(s2)
    e_nosie = s1 - s2
    target_norm = l2_norm(s2, s2)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10*torch.log10((target_norm)/(noise_norm+eps)+eps)
    return snr



class SpeakerLoss(nn.Module):
    
    def __init__(self, embedding_dim=512, numspks=2, overallspks=50):
        super(SpeakerLoss, self).__init__()
        
        self.numspks = numspks 
        self.embedding_dim = embedding_dim
        self.M = torch.tensor(overallspks )
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1), requires_grad=True)
        emb = torch.ones(overallspks, embedding_dim)
        emb = torch.nn.init.orthogonal_(emb)
        norm = 1. #torch.norm(emb,p=2,dim=-1,keepdim=True)

        self.emb_table = nn.Parameter(emb/norm, requires_grad=True)
        perm = permutations([x for x in range(numspks)])
        self.register_buffer('perm',torch.tensor(list(perm)))
    
    def forward(self, inputs, spkid):
        B, N, D, T = inputs.size() 
        assert N == self.numspks 

        # [B, N, 1, D,T]
        feature = torch.unsqueeze(inputs, 2)
        # [1, M, D]
        embedding = torch.unsqueeze(self.emb_table, 0)
        # [1, M, D, 1]
        embedding = torch.unsqueeze(embedding, 1)
        # [1, 1, M, D, 1]
        embedding = torch.unsqueeze(embedding, -1)
        # [B, N, M, T]
        reg = self.emb_table.unsqueeze(1) - self.emb_table.unsqueeze(0)
        
        #reg = torch.einsum('md,nd->mn', self.emb_table,-self.emb_table)
        reg = torch.sum(torch.abs(reg),-1)
        max_val= torch.max(reg)
        max_val = max_val.detach()
        mask = (torch.eye(self.M)*max_val).to(inputs.device)
        min_reg, _ = torch.min(reg+mask,-1)
        min_entropy = torch.log(min_reg+1e-8)
        # negative log entropy
        reg = -torch.mean(min_entropy)
        # to [B,N,M,T]
        f_loss = torch.sum(torch.abs(feature-embedding)**2,3)
        distance = (torch.abs(self.alpha)+1e-5)*f_loss + self.beta
        
        t = []
        for i in range(len(self.perm)):
            t.append(spkid[:,self.perm[i]])

        # to [B,P,N]
        index = torch.stack(t,1)
        # to [B,P,N,M]
        index = F.one_hot(index, self.M).float().to(inputs.device)
        #print(spkid) 
        # to [B,N,1,T]
        others = torch.log(torch.mean(torch.exp(-distance),2,keepdim=True)+1e-8) 
        # to [B,P,N,T]
        s_loss = torch.einsum('bnmt, bpnm->bpnt', distance+others, index)  

        # to [B,P,T]
        s_loss = torch.mean(s_loss, 2)
        # to [B, T]
        loss, idx = torch.min(s_loss,1)
        #loss = torch.mean(torch.abs(inputs[:,self.per] - embedding[:,self.perm])**2) 
        #print(idx) 
        # Get center
        order = self.perm[idx]
        order = torch.transpose(order,1,2).unsqueeze(2).repeat(1,1,self.embedding_dim,1,)
        
        h = torch.gather(inputs, index=order, dim=1)
        loss = torch.mean(loss)
        center = torch.mean(h,-1)
        return loss, center, reg


def test_net():
    import soundfile as sf 
    import numpy as np
    net = WaveSplit(overallspks=50).cuda()
    #inputs = torch.randn(2,6000).cuda()
    #data = sf.read('../../../data/2speakers/wav8k/min/tt/mix/050a0501_1.7783_442o030z_-1.7783.wav')[0].T
    #inputs = torch.from_numpy(data[None,:].astype(np.float32)).cuda()

    inputs = torch.randn(3,6000).cuda()
    spkid = torch.tensor([[x+25,x+1] for x in range(3)]).cuda()
    net.zero_grad()
    out,loss,reg = net(inputs,spkid)
    inputs = inputs[:,:out.size(-1)]
    #sdr = net.loss(out,inputs,'SiSNR')
    loss = loss+reg#+sdr
    loss.backward()
    #net.eval()
    #out = net(inputs, training=False)
    #sf.write('a.wav',out[0].detach().cpu().numpy().T,8000)

def test_loss():
    func = SpeakerLoss()
    inputs_feature = torch.randn(10,2,512,100)

    spkid = torch.tensor([[x,x+1] for x in range(10)]) 
    loss, center, reg= func(inputs_feature, spkid)
    print(loss.shape, center.shape, reg.shape)


if __name__ == "__main__":
    torch.set_printoptions(profile="full")
    test_net()
#    test_loss()    
    inputs = torch.randn(10,2,1000)
    labels = torch.randn(10,2,1000)

    #pit(inputs, labels)

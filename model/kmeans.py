#!/usr/bin/env python
# coding=utf-8

import torch 
import torch.nn as nn

def initialize(feature):
    '''
        feature:[B, nspk, D, T]
    return:
        centers: [B,nspk,D]
    ''' 
    centers = torch.zeros(*feature.size()[:3]).to(feature.device)
    # [B, N, D]
    centers[:,0] = feature[:,0,:,0]
    k = centers.size(1)
    for idx in range(1,k):
        # [ B, idx, N, T]
        dist = torch.sum(torch.abs(feature[:,None]-centers[:,0:idx+1,None,:,None])**2,-2)
        # [ B, N, T]
        dist, index = torch.min(dist,1)
        # [ B, N, T]
        dist = dist/torch.sum(dist, [1,2], keepdim=True)
        B, N, T = dist.size()
        dist = torch.reshape(dist, [B,-1])
        prob = torch.cumsum(dist, 1)
        r = torch.rand(B)

        for b in range(B):
            for t in range(T*N):
                if r[b] >= prob[b,t]:
                    centers[b,idx] = feature[b,t//T,:,t%T]
    return centers


class KMeans(nn.Module):

    def __init__(self, batch_size,num_centers, feature_dim=512, iter_nums=100, error=1e-5):
        super(KMeans,self).__init__()       
        #self.register_buffer('center',torch.zeros([batch_size, num_centers, feature_dim]))
        self.iter_nums = iter_nums
        self.error = error
        self.num_centers = num_centers

    def forward(self, inputs):
        '''
            inputs : [batch_size, n_spks, dimention, time_step]
        '''
        self.centers = initialize(inputs)
        # to [ B, 1, N, D, 1]
        centers = torch.unsqueeze(self.centers, 1)
        centers = torch.unsqueeze(centers, -1)
        # to [ B, N, 1, D, T]
        inputs = torch.unsqueeze(inputs,2)
        iter = 0
        error= self.error*2+1.
        while iter < self.iter_nums:# and error > self.error:
            error, centers = self.update(inputs, centers)
            iter += 1
        centers = centers.squeeze(1)
        centers = centers.squeeze(-1)
        print(error, iter, torch.mean(torch.abs(centers[:,0]-centers[:,1])))
        return centers

    def update(self, inputs, centers):
        #inputs: [ B, 1, N, D, T]

        # [ B, 1, N, T]
        distance = torch.mean(torch.abs(inputs-centers)**2, 3) 
        # [ B, N, T]
        idx = torch.argmin(distance, 2)
        c = [] 
        for i in range(self.num_centers):
            # [ B, N, T]
            index = torch.eq(idx,i)
            # [ B, 1, 1, 1]
            nums = torch.sum(index, [1,2],keepdim=True).unsqueeze(1)
            # [ B, 1, 1, 1, 1]
            nums = torch.unsqueeze(nums, 1) 
            # [ B, N, 1, T]
            mask = torch.unsqueeze(index,2)
            # [ B, N, 1, 1, T]
            mask = torch.unsqueeze(mask,2)
            # [ B, 1, 1, D, 1]
            c_t = torch.sum(inputs*mask,[1,4], keepdim=True)
            c_t = c_t/(nums+1e-8)
            c.append(c_t)
        # [ B, 1, N, D,1]
        new_centers = torch.cat(c,2)
        error = torch.mean(torch.abs(new_centers-centers)).data.item()
        return error, new_centers 


def test():
    torch.manual_seed(10)
    net = KMeans(10,3,33)
    inputs_feature1 = torch.randn(10,1, 2,100)*0.1
    inputs_feature2 = torch.randn(10,1, 2,100)*0.1+1
    inputs_feature3 = torch.randn(10,1, 2,100)*0.1+2
    a=torch.cat([inputs_feature1,inputs_feature2,inputs_feature3],1)
    b = net(a)
    print(b[0,0])
    print(b[0,1])
    print(b[0,2])
#test()


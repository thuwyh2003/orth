# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict

import torch
import torch.nn as nn

from rlinf.utils.logging import get_logger

activation_dict = nn.ModuleDict(
    {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "mish": nn.Mish(),
        "identity": nn.Identity(),
        "softplus": nn.Softplus(),
        "silu": nn.SiLU(),
    }
)

class OrthNoiseNet(nn.Module):
    def __init__(
        self,
        in_dim:int,
        out_dim:int,
        hidden_dim:list[int],
        activation_type:str,
        delta:float=0.05,
        action_dim:int=32
    ):
        super().__init__()
        self.mlp=MLP(in_dim,hidden_dim,out_dim,activation_type)
        self.delta=delta
        self.d=action_dim
    def forward(self,noise_feature:torch.tensor):
        orth_gamma=self.mlp(noise_feature)
        # d=noise_feature.shape[-1]
        
        
        gamma_iso = 1.0 / self.d

        gamma = gamma_iso + self.delta * torch.tanh(orth_gamma)

        gamma = gamma.clamp(
            min=1e-4,
            max=1-1e-4
        )

        alpha = torch.sqrt(gamma)

        beta = torch.sqrt(
            (1-gamma) / (self.d-1)
        )
        return alpha, beta, gamma, orth_gamma
        
        
class MLP(nn.Module):
    def __init__(
        self,
        in_d:int,
        hid_d:list[int],
        o_dim:int,
        activation_type:str,
    ):
        super().__init__()
        dim_list=[in_d]+hid_d+[o_dim]
        self.moduleList = nn.ModuleList()
        layers=[]
        for idx in range(len(dim_list)):
            if idx<len(dim_list)-1:
                in_dim=dim_list[idx]
                out_dim=dim_list[idx+1]
                self.layer=nn.Linear(in_dim,out_dim)
                self.norm=nn.LayerNorm(out_dim)
                self.drop=nn.Dropout(0.0)
                layers.append(self.layer)
                layers.append(self.norm)
                layers.append(self.drop)
                
            self.act=(activation_dict[activation_type.lower()]
                      if idx<len(dim_list)-1
                      else activation_dict['identity']) 
            layers.append(self.act)
        self.moduleList=nn.Sequential(*layers)
        
    def forward(self,x):
        for idx,m in enumerate(self.moduleList):
            x=m(x)
        return x
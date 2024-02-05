from typing import Iterable, Tuple, Callable,Any
from FrEIA.modules.base import ShapeList
import FrEIA.modules as Fm
import torch
import torch.nn as nn
from torch import Tensor

#Modified GIN coupling block to allow variable jacobian determinant
class ModifiedGINCouplingBlock(Fm.GINCouplingBlock):
    def __init__(self, dims_in, dims_c=[], subnet_constructor: Callable[..., Any] = None, clamp: float = 2, clamp_activation: str | Callable[..., Any] = "ATAN", split_len: float | int = 0.5,normalize:bool = True):
        '''
        Additional parameters:
            normalize:      Return constant jacobian of on if true
        '''

        super().__init__(dims_in, dims_c, subnet_constructor, clamp, clamp_activation, split_len)
        self.normalize = normalize

    def _coupling1(self, x1, u2, rev=False):

        # notation (same for _coupling2):
        # x: inputs (i.e. 'x-side' when rev is False, 'z-side' when rev is True)
        # y: outputs (same scheme)
        # *_c: variables with condition appended
        # *1, *2: left half, right half
        # a: all affine coefficients
        # s, t: multiplicative and additive coefficients
        # j: log det Jacobian

        a2 = self.subnet2(u2)
        s2, t2 = a2[:, :self.split_len1], a2[:, self.split_len1:]
        s2 = self.clamp * self.f_clamp(s2)

        #Constant jacobian
        if self.normalize: 
            s2 = s2 - s2.mean(1, keepdim=True)
            jac = 0.0

        #Variable Jacobian
        else:
            jac = s2.sum(-1)

        if rev:
            y1 = (x1 - t2) * torch.exp(-s2)
            return y1, jac
        else:
            y1 = torch.exp(s2) * x1 + t2
            return y1, jac
        
    def _coupling2(self, x2, u1, rev=False):
        a1 = self.subnet1(u1)
        s1, t1 = a1[:, :self.split_len2], a1[:, self.split_len2:]
        s1 = self.clamp * self.f_clamp(s1)

        #Constant jacobian
        if self.normalize: 
            s1 = s1- s1.mean(1, keepdim=True)
            jac = 0.0

        #Variable Jacobian
        else:
            jac = s1.sum(-1)

        if rev:
            y2 = (x2 - t1) * torch.exp(-s1)
            return y2, -jac
        else:
            y2 = torch.exp(s1) * x2 + t1
            return y2, jac
        

#Global scaling block for INN with constant Jacobian determinant
class ScalingBlock(Fm.InvertibleModule):
    def __init__(self, dims_in: ShapeList, dims_c: ShapeList = None):
        super().__init__(dims_in, dims_c)

        #Learnable parameter
        self.a = nn.Parameter(torch.ones([1]))

    def output_dims(self, input_dims: ShapeList) -> ShapeList:
        return input_dims
    
    def forward(self, x_or_z: Iterable[Tensor], c: Iterable[Tensor] = None, rev: bool = False, jac: bool = True) -> Tuple[Tuple[Tensor], Tensor]:
        

        x = x_or_z[0]
        N= x.shape[0]
        d = x.shape[1]

        if rev:
            jac = - d * torch.log(self.a)
            x = x / self.a

        else:
            jac = + d * torch.log(self.a)
            x = x * self.a

        return ((x,),jac)     

#Construct the subnetworks for the normalizing flow
def get_subnet(c_in,c_out):

    d_hidden = 128
    layers = nn.Sequential(
        nn.Linear(c_in,d_hidden),
        nn.ReLU(),
        nn.Linear(d_hidden,d_hidden),
        nn.ReLU(),
        nn.Linear(d_hidden,d_hidden),
        nn.ReLU(),
        nn.Linear(d_hidden,c_out)
    )

    #Initialize the weights of the linear layers
    for layer in layers:
        if isinstance(layer,nn.Linear):
            nn.init.xavier_normal_(layer.weight)

    #Set the weights and the bias of the layer to zero
    layers[-1].weight.data.fill_(0.0)
    layers[-1].bias.data.fill_(0.0)

    return layers
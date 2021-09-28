import torch
import torch.nn as nn

from torch.nn.functional import softplus

import pickle, os

class cEncoder(nn.Module):
    """
    Encoder module for CVAE,
    i.e. it maps (Y, c) to the approximate posterior q(z)=N(mu_z, sigma_z)
    """
    # z_dim the number of dimensions
    # mapping is the NN coded in nn.Sequential(...)
    def __init__(self, z_dim, mapping):
        super().__init__()

        self.z_dim = z_dim

        # NN mapping from (Y, x) to z
        self.mapping = mapping

    def forward(self, Y, c):

#### they should return as many columns as LD 

        # z and c concatenated and pass through the network
        
        out = self.mapping(torch.cat([Y, c], dim=1))

        outfolder="ND_TOY_EXAMPLE"
        outsuffix="_debug_encoder"
        filename = os.path.join(outfolder, 'out'+ outsuffix +'.sav')
        pickle.dump(out, open(filename, 'wb'))
        print("... written: " + filename )
        
        mu = out[:, 0:self.z_dim]
        
                # SoftPlus is a smooth approximation to the ReLU function and can 
        # be used to constrain the output of a machine to always be positive.
        sigma = 1e-6 + softplus(out[:, self.z_dim:(2 * self.z_dim)])

        print("mu.shape")
        print(mu.shape)
        
        print("cEncoder - forward - mu.shape")
        print(mu.shape)
        
        print("cEncoder - forward - sigma.shape")
        print(sigma.shape)
        
        print("cEncoder - forward - out.shape")
        print(out.shape)



        return mu, sigma

import torch
import torch.nn as nn

from torch.nn.functional import softplus


class cEncoder(nn.Module):
    """
    Encoder module for CVAE,
    i.e. it maps (Y, c) to the approximate posterior q(z)=N(mu_z, sigma_z)
    """

    def __init__(self, z_dim, mapping):
        super().__init__()

        self.z_dim = z_dim

        # NN mapping from (Y, x) to z
        self.mapping = mapping

    def forward(self, Y, c):

        out = self.mapping(torch.cat([Y, c], dim=1))

        

        mu = out[:, 0:self.z_dim]
        
        # SoftPlus is a smooth approximation to the ReLU function and can 
        # be used to constrain the output of a machine to always be positive.
        sigma = 1e-6 + softplus(out[:, self.z_dim:(2 * self.z_dim)])

        print("cEncoder - forward - mu.shape")
        print(mu.shape)
        
        print("cEncoder - forward - sigma.shape")
        print(sigma.shape)
        
        print("cEncoder - forward - out.shape")
        print(out.shape)
        return mu, sigma


# =============================================================================
# mu.shape
# torch.Size([64, 1])
# sigma.shape
# torch.Size([64, 1])
# out.shape
# torch.Size([64, 2])        
# 
# 
# =============================================================================

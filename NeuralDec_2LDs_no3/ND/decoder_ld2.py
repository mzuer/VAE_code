import torch
import numpy as np

import torch.nn as nn
from torch.autograd import Variable

from torch.nn.functional import softplus

from .helpers import expand_grid, approximate_KLqp, rsample_RelaxedBernoulli

from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs

class Decoder(nn.Module):

    def __init__(self, output_dim,
                 grid_z1, grid_z2, grid_c, grid_cz1, grid_cz2, grid_z1z2, 
                 mapping_z1=None,  mapping_z2=None, mapping_c=None,
                 mapping_cz1=None, mapping_cz2=None, mapping_z1z2=None, 
                 has_feature_level_sparsity=True,
                 penalty_type="fixed", lambda0=1.0,
                 likelihood="Gaussian",
                 p1=0.2, p2=0.2, p3=0.2, p4=0.2, p5=0.2, p6=0.2, p7=0.2, device="cpu"):
        """
        NN mapping with constraints to be used as the decoder in a CVAE. Performs Neural Decomposition.
        :param output_dim: data dimensionality
        :param grid_z: grid for quadrature (estimation of integral for f(z))
        :param grid_c: grid for quadrature (estimation of integral for f(c))
        :param grid_cz: grid for quadrature (estimation of integral for f(z, c))
        :param mapping_z: neural net mapping z to data
        :param mapping_c: neural net mapping c to data
        :param mapping_cz: neural net mapping (z, c) to data
        :param has_feature_level_sparsity: whether to use (Relaxed) Bernoulli feature-level sparsity
        :param penalty_type: which penalty to apply
        :param lambda0: initialisation for both fixed penalty $c$ as well as $lambda$ values
        :param likelihood: Gaussian or Bernoulli
        :param p1: Bernoulli prior for sparsity on mapping_z1
        :param p2: Bernoulli prior for sparsity on mapping_z2
        :param p3: Bernoulli prior for sparsity on mapping_c
        :param p4: Bernoulli prior for sparsity on mapping_z1z2
        :param p5: Bernoulli prior for sparsity on mapping_cz1
        :param p6: Bernoulli prior for sparsity on mapping_cz2
        :param device: cpu or cuda
        """
        super().__init__()

        self.output_dim = output_dim
        self.likelihood = likelihood
        self.has_feature_level_sparsity = has_feature_level_sparsity
        self.penalty_type = penalty_type

        self.grid_z1 = grid_z1.to(device)
        self.grid_z2 = grid_z2.to(device)
        self.grid_c = grid_c.to(device)
        self.grid_cz1 = grid_cz1.to(device)
        self.grid_cz2 = grid_cz2.to(device)
        self.grid_z1z2 = grid_z1z2.to(device)


        self.n_grid_z1 = grid_z1.shape[0]
        self.n_grid_z2 = grid_z2.shape[0]
        self.n_grid_c = grid_c.shape[0]
        self.n_grid_cz1 = grid_cz1.shape[0]
        self.n_grid_cz2 = grid_cz2.shape[0]
        self.n_grid_z1z2 = grid_z1z2.shape[0]

        
        # input -> output
        self.mapping_z1 = mapping_z1
        self.mapping_z2 = mapping_z2
        self.mapping_c = mapping_c
        self.mapping_cz1 = mapping_cz1
        self.mapping_cz2 = mapping_cz2
        self.mapping_z1z2 = mapping_z1z2

        
        if self.likelihood == "Gaussian":
            # feature-specific variances (for Gaussian likelihood)
            # nn.Parameter: A kind of Tensor that is to be considered a module parameter.
            self.noise_sd = torch.nn.Parameter(-1.0 * torch.ones(1, output_dim))

        self.intercept = torch.nn.Parameter(torch.zeros(1, output_dim))
        
        # Lambda -> in the formula of the augmented Lagrangian

        self.Lambda_z1 = Variable(lambda0*torch.ones(1, output_dim, device=device), requires_grad=True)
        
        self.Lambda_z2 = Variable(lambda0*torch.ones(1, output_dim, device=device), requires_grad=True)

        self.Lambda_c = Variable(lambda0*torch.ones(1, output_dim, device=device), requires_grad=True)
        


        self.Lambda_cz1_z1 = Variable(lambda0*torch.ones(self.n_grid_z1, output_dim, device=device), 
                                     requires_grad=True)

        self.Lambda_cz1_c = Variable(lambda0*torch.ones(self.n_grid_c, output_dim, device=device), 
                                     requires_grad=True)

        self.Lambda_cz2_z2 = Variable(lambda0*torch.ones(self.n_grid_z2, output_dim, device=device), 
                                     requires_grad=True)

        self.Lambda_cz2_c = Variable(lambda0*torch.ones(self.n_grid_c, output_dim, device=device), 
                                     requires_grad=True)


        self.Lambda_z1z2_z1 = Variable(lambda0*torch.ones(self.n_grid_z1, output_dim, device=device), 
                                      requires_grad=True)

        self.Lambda_z1z2_z2 = Variable(lambda0*torch.ones(self.n_grid_z2, output_dim, device=device), 
                                      requires_grad=True)


        self.lambda0 = lambda0

        self.device = device

        # RelaxedBernoulli
        self.temperature = 1.0 * torch.ones(1, device=device)

        #LogitRelaxedBernoulli(temperature, probs=None, logits=None, validate_args=None)
        #Creates a LogitRelaxedBernoulli distribution parameterized by probs or logits (but not both),
        #which is the logit of a RelaxedBernoulli distribution.
        #Samples are logits of values in (0, 1)
                #temperature (Tensor) – relaxation temperature
                #probs (Number, Tensor) – the probability of sampling 1
                #logits (Number, Tensor) – the log-odds of sampling 1

        if self.has_feature_level_sparsity:

            # for the prior RelaxedBernoulli(logits)
            # torch.ones: Returns a tensor filled with the scalar value 1,
            # probs_to_logits: #' Converts a tensor of probabilities into logits. For the binary case,
            #' this denotes the probability of occurrence of the event indexed by `1`. 
            # For the multi-dimensional case, the values along the last dimension
            #' denote the probabilities of occurrence of each of the events.
            self.logits_z1 = probs_to_logits(p1 * torch.ones(1, output_dim).to(device), is_binary=True)
            self.logits_z2 = probs_to_logits(p2 * torch.ones(1, output_dim).to(device), is_binary=True)
            self.logits_c = probs_to_logits(p3 * torch.ones(1, output_dim).to(device), is_binary=True)
            self.logits_z1z2 = probs_to_logits(p4 * torch.ones(1, output_dim).to(device), is_binary=True)
            self.logits_cz1 = probs_to_logits(p5 * torch.ones(1, output_dim).to(device), is_binary=True)
            self.logits_cz2 = probs_to_logits(p6 * torch.ones(1, output_dim).to(device), is_binary=True)



###?????????????????????????????? how 3.0 [for z and c] and 2.0 [for cz] are chosen ???
            # KM: set higher proba for the single interaction term
            # but should not matter to much
            # for the approx posterior
            self.qlogits_z1 = torch.nn.Parameter(3.0 * torch.ones(1, output_dim).to(device))
            self.qlogits_z2 = torch.nn.Parameter(3.0 * torch.ones(1, output_dim).to(device))
            self.qlogits_c = torch.nn.Parameter(3.0 * torch.ones(1, output_dim).to(device))
            self.qlogits_cz1 = torch.nn.Parameter(2.0 * torch.ones(1, output_dim).to(device))
            self.qlogits_cz2 = torch.nn.Parameter(2.0 * torch.ones(1, output_dim).to(device))
            self.qlogits_z1z2 = torch.nn.Parameter(2.0 * torch.ones(1, output_dim).to(device))



    def forward_z1(self, z):
        
        print("decoder -forward_z1 - z.shape")
        print(z.shape)
        
        value = self.mapping_z1(z) # mapping = a NN
        
        print("decoder -forward_1 - value.shape")
        print(value.shape)

        
        if self.has_feature_level_sparsity:
            w = rsample_RelaxedBernoulli(self.temperature, self.qlogits_z1)
            return w * value
        else:
            return value
        
    def forward_z2(self, z):
        value = self.mapping_z2(z) # mapping = a NN
        if self.has_feature_level_sparsity:
            w = rsample_RelaxedBernoulli(self.temperature, self.qlogits_z2)
            return w * value
        else:
            return value


    def forward_c(self, c):
        value = self.mapping_c(c)
        if self.has_feature_level_sparsity:
            w = rsample_RelaxedBernoulli(self.temperature, self.qlogits_c)
            return w * value
        else:
            return value

    def forward_cz1(self, z, c):
        return self.forward_cz1_concat(torch.cat([z, c], dim=1))

    def forward_cz1_concat(self, zc_concat):
        value = self.mapping_cz1(zc_concat)
        if self.has_feature_level_sparsity:
            w = rsample_RelaxedBernoulli(self.temperature, self.qlogits_cz1)
            return w * value
        else:
            return value
        
        
    def forward_cz2(self, z, c):
        return self.forward_cz2_concat(torch.cat([z, c], dim=1))

    def forward_cz2_concat(self, zc_concat):
        value = self.mapping_cz2(zc_concat)
        if self.has_feature_level_sparsity:
            w = rsample_RelaxedBernoulli(self.temperature, self.qlogits_cz2)
            return w * value
        else:
            return value
        

        
    def forward_z1z2(self, z1, z2):
        return self.forward_z1z2_concat(torch.cat([z1, z2], dim=1))

    def forward_z1z2_concat(self, zz_concat):
        value = self.mapping_z1z2(zz_concat)
        if self.has_feature_level_sparsity:
            w = rsample_RelaxedBernoulli(self.temperature, self.qlogits_z1z2)
            return w * value
        else:
            return value

    def forward(self, z1, z2, c):
        return self.intercept + self.forward_z1(z1) + self.forward_z2(z2) +\
                    self.forward_z1z2(z1,z2) + self.forward_c(c) +\
                    self.forward_cz1(z1, c) + self.forward_cz2(z2, c)

    def loglik(self, y_pred, y_obs):

        if self.likelihood == "Gaussian":
            sigma = 1e-6 + softplus(self.noise_sd)
            p_data = Normal(loc=y_pred, scale=sigma)
            loglik = p_data.log_prob(y_obs).sum()
        elif self.likelihood == "Bernoulli":
            p_data = Bernoulli(logits=y_pred)
            loglik = p_data.log_prob(y_obs).sum()
        else:
            raise NotImplementedError("Other likelihoods not implemented")

        return loglik

    def set_temperature(self, x):
        self.temperature = x * torch.ones(1, device=self.device)

    # used in this script for calculating the loss
    # do int_cz_dc and int_cz_dz
    def calculate_integrals(self):

        # has shape [1, output_dim]
        int_z1 = self.forward_z1(self.grid_z1).mean(dim=0).reshape(1, self.output_dim)
        int_z2 = self.forward_z2(self.grid_z2).mean(dim=0).reshape(1, self.output_dim)

        # has shape [1, output_dim]
        int_c = self.forward_c(self.grid_c).mean(dim=0).reshape(1, self.output_dim)

        m1 = self.n_grid_z1
        m2 = self.n_grid_c
        out = self.forward_cz1_concat(self.grid_cz1)
        out = out.reshape(m1, m2, self.output_dim)
        int_cz1_dc = out.mean(dim=1)
        int_cz1_dz1 = out.mean(dim=0)

        m1 = self.n_grid_z2
        m2 = self.n_grid_c
        out = self.forward_cz2_concat(self.grid_cz2)
        out = out.reshape(m1, m2, self.output_dim)
        int_cz2_dc = out.mean(dim=1)
        int_cz2_dz2 = out.mean(dim=0)
        
        m1 = self.n_grid_z1
        m2 = self.n_grid_z2
        out = self.forward_z1z2_concat(self.grid_z1z2)
        out = out.reshape(m1, m2, self.output_dim)
        int_z1z2_dz1 = out.mean(dim=1)
        int_z1z2_dz2 = out.mean(dim=0)
        

### ????????????????? how to do for the 3 terms ???!!
        return int_z1, int_z2, int_c, int_cz1_dc, int_cz1_dz1,\
                    int_cz2_dc, int_cz2_dz2, int_z1z2_dz1, int_z1z2_dz2
    

    # used in CVAE.py optimize() for
    # 1) get shapes for integrals
    # 2) logging for integral constraints
    # in _numpy cz_dc and cz_dz are concatenated
    # just needed for the output logging^
    def calculate_integrals_numpy(self):

        with torch.no_grad():

            # has shape [1, output_dim]
            
            int_z1 = self.forward_z1(self.grid_z1).mean(dim=0).reshape(1, self.output_dim).cpu().numpy()
            int_z2 = self.forward_z2(self.grid_z2).mean(dim=0).reshape(1, self.output_dim).cpu().numpy()

            # reshape: convert [1000] -> [1,1000]
            
            # has shape [1, output_dim]
            # KM: was mapping_c => but this was a bug, should be forward_c
            int_c = self.forward_c(self.grid_c).mean(dim=0).reshape(1, self.output_dim).cpu().numpy()

            m1 = self.n_grid_z1
            m2 = self.n_grid_c
            out = self.forward_cz1_concat(self.grid_cz1) # :0 is range of z, :1 is range of c
            print("out.shape")
            print(out.shape)
            print("-ok")
            out = out.reshape(m1, m2, self.output_dim)
            print("out_reshaped.shape")
            print(out.shape)
            print("-ok")
            int_cz1_dc = out.mean(dim=1).cpu().numpy()
            int_cz1_dz1 = out.mean(dim=0).cpu().numpy()
            print("int_cz1_dc")
            print(int_cz1_dc.shape)
            print("-ok")
            print("int_cz1_dz")
            print(int_cz1_dc.shape)
            print("-ok")

            m1 = self.n_grid_z2
            m2 = self.n_grid_c
            out = self.forward_cz2_concat(self.grid_cz2)
            out = out.reshape(m1, m2, self.output_dim)
            int_cz2_dc = out.mean(dim=1).cpu().numpy()
            int_cz2_dz2 = out.mean(dim=0).cpu().numpy()
            
            m1 = self.n_grid_z1
            m2 = self.n_grid_z2
            out = self.forward_z1z2_concat(self.grid_z1z2)
            out = out.reshape(m1, m2, self.output_dim)
            int_z1z2_dz2 = out.mean(dim=1).cpu().numpy()
            int_z1z2_dz1 = out.mean(dim=0).cpu().numpy()

            ### how to stack the 3d ???????
            # np.vstack Stack arrays in sequence vertically (row wise).
            # This is equivalent to concatenation along the first axis after 1-D
            #  arrays of shape (N,) have been reshaped to (1,N).
            int_cz1 = np.vstack((int_cz1_dc, int_cz1_dz1))
            int_cz2 = np.vstack((int_cz2_dc, int_cz2_dz2))
            int_z1z2 = np.vstack((int_z1z2_dz2, int_z1z2_dz1))

            print("int_z1.shape")
            print(int_z1.shape)
            print("int_z2.shape")
            print(int_z2.shape)
            print("int_c.shape")
            print(int_c.shape)
            print("int_cz1.shape")
            print(int_cz1.shape)
            print("int_cz2.shape")
            print(int_cz2.shape)
            print("int_z1z2.shape")
            print(int_z1z2.shape)
            sys.exit(0)
        return int_z1, int_z2, int_c, int_cz1, int_cz2, int_z1z2



# =============================================================================
# print("self.forward_z1(self.grid_z1).shape")
# torch.Size([15, 1000])
# int_z1.shape
# (1, 1000)
# int_z2.shape
# (1, 1000)
# int_c.shape
# (1, 1000)
# int_cz1.shape
# (30, 1000)
# int_cz2.shape
# (30, 1000)
# int_z1z2.shape
# (30, 1000)
# =============================================================================
# out.shape
# torch.Size([225, 1000])
# out_reshaped.shape
# torch.Size([15, 15, 1000])
# =============================================================================
# int_cz1_dc
# (15, 1000)
# int_cz1_dz
# (15, 1000)
# # =============================================================================
# =============================================================================
# =============================================================================

    def calculate_penalty(self):
#                 return int_z1, int_z2, int_c, int_cz1_dc, int_cz1_dz1,
#                     int_cz2_dc, int_cz2_dz2, int_z1z2_dz1, int_z1z2_dz2
        int_z1, int_z2, int_c, int_cz1_dc, int_cz1_dz1,\
                     int_cz2_dc, int_cz2_dz2, int_z1z2_dz1, int_z1z2_dz2,\
                      = self.calculate_integrals()

        # penalty with fixed lambda0
        if self.penalty_type in ["fixed", "MDMM"]:
            print("decoder  calc_pen - int_cz2_dz2.abs().mean()")
            print(int_cz2_dz2.abs().mean())
            print("decoder  calc_pen - lambda0")
            print(self.lambda0)
            #penalty0 = self.lambda0 * (int_cz2_dz2.abs().mean())
            penalty0 = self.lambda0 * (int_z1.abs().mean() + int_z2.abs().mean() + int_c.abs().mean() +\
                                       int_cz1_dc.abs().mean() + int_cz1_dz1.abs().mean()+\
                                       int_cz2_dc.abs().mean() + int_cz2_dz2.abs().mean() +\
                                       int_z1z2_dz1.abs().mean()+ int_z1z2_dz2.abs().mean())
            
        if self.penalty_type in ["BDMM", "MDMM"]:
            print("BENALTY_BDMM")
            penalty_BDMM = (self.Lambda_z1 * int_z1).mean() + (self.Lambda_z2 * int_z2).mean() +\
                            (self.Lambda_c * int_c).mean() +\
                           (self.Lambda_cz1_c * int_cz1_dc).mean() + (self.Lambda_cz1_z1 * int_cz1_dz1).mean()+\
                           (self.Lambda_cz2_c * int_cz2_dc).mean() + (self.Lambda_cz2_z2 * int_cz2_dz2).mean()+\
                           (self.Lambda_z1z2_z2 * int_z1z2_dz2).mean() + (self.Lambda_z1z2_z1 * int_z1z2_dz1).mean()
                           

        if self.penalty_type == "fixed":
            penalty = penalty0
        elif self.penalty_type == "BDMM":
            penalty = penalty_BDMM
        elif self.penalty_type == "MDMM":
            penalty = penalty_BDMM + penalty0
        else:
            raise ValueError("Unknown penalty type")

        return penalty, int_z1, int_z2, int_c, int_cz1_dc, int_cz1_dz1,\
                     int_cz2_dc, int_cz2_dz2, int_z1z2_dz1, int_z1z2_dz2

    def loss(self, y_pred, y_obs):

        penalty, int_z1, int_z2, int_c, int_cz1_dc, int_cz1_dz1,\
                     int_cz2_dc, int_cz2_dz2, int_z1z2_dz1, int_z1z2_dz2,\
                      = self.calculate_penalty()

        total_loss = - self.loglik(y_pred, y_obs) + penalty

        if self.has_feature_level_sparsity:
            KL_z1 = approximate_KLqp(self.logits_z1, self.qlogits_z1)
            KL_z2 = approximate_KLqp(self.logits_z2, self.qlogits_z2)
            KL_c = approximate_KLqp(self.logits_c, self.qlogits_c)
            KL_cz1 = approximate_KLqp(self.logits_cz1, self.qlogits_cz1)
            KL_cz2 = approximate_KLqp(self.logits_cz2, self.qlogits_cz2)
            KL_z1z2 = approximate_KLqp(self.logits_z1z2, self.qlogits_z1z2)
            total_loss += 1.0 * (KL_z1 + KL_z2 + KL_c + KL_cz1 + KL_cz2 + KL_z1z2 )

        return total_loss, penalty, int_z1, int_z2, int_c, int_cz1_dc, int_cz1_dz1,\
                     int_cz2_dc, int_cz2_dz2, int_z1z2_dz1, int_z1z2_dz2

    def fraction_of_variance_explained(self, z1, z2, c, account_for_noise=False, divide_by_total_var=True):

        with torch.no_grad():
            # f_z effect
            f_z1 = self.forward_z1(z1)
            f_z1_var = f_z1.var(dim=0, keepdim=True)
            f_z1 = self.forward_z1(z1)
            f_z1_var = f_z1.var(dim=0, keepdim=True)
            
            f_z2 = self.forward_z2(z2)
            f_z2_var = f_z2.var(dim=0, keepdim=True)
            f_z2 = self.forward_z2(z2)
            f_z2_var = f_z2.var(dim=0, keepdim=True)


            # f_c
            f_c = self.forward_c(c)
            f_c_var = f_c.var(dim=0, keepdim=True)

            # f_int
            f_cz1 = self.forward_cz1(z1, c)
            f_cz1_var = f_cz1.var(dim=0, keepdim=True)
            
            # f_int
            f_cz2 = self.forward_cz2(z2, c)
            f_cz2_var = f_cz2.var(dim=0, keepdim=True)


            # f_int
            f_z1z2 = self.forward_z1z2(z1, z2)
            f_z1z2_var = f_z1z2.var(dim=0, keepdim=True)


            # collect Var([f_z, f_c, f_int]) together
            # and divide by total variance
            f_all_var = torch.cat([f_z1_var, f_z2_var, f_c_var, f_cz1_var,
                                   f_cz2_var, f_z1z2_var], dim=0)

            if divide_by_total_var:

                total_var = f_all_var.sum(dim=0, keepdim=True)

                if account_for_noise:
                    total_var += self.noise_sd.reshape(-1) ** 2

                f_all_var /= total_var

            return f_all_var.t()

    def get_feature_level_sparsity_probs(self):

        with torch.no_grad():
            # f_z effect
            #self.qlogits_z = torch.nn.Parameter(3.0 * torch.ones(1, output_dim).to(device))
            w_z1 = torch.sigmoid(self.qlogits_z1)
            w_z2 = torch.sigmoid(self.qlogits_z2)
            w_z1z2 = torch.sigmoid(self.qlogits_z1z2)
            w_c = torch.sigmoid(self.qlogits_c)
            w_cz1 = torch.sigmoid(self.qlogits_cz1)
            w_cz2 = torch.sigmoid(self.qlogits_cz2)

            return torch.cat([w_z1, w_z2, w_z1z2, w_c, w_cz1, w_cz2], dim=0).t()

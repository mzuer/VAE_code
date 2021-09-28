import torch
import torch.nn as nn
import numpy as np

from math import ceil

#from .helpers import KL_standard_normal
from .helpers import KL_standard_normal_2ld

class CVAE(nn.Module):
    """
    CVAE with Neural Decomposition as part of the decoder
    """

    def __init__(self, encoder, decoder, lr, device="cpu"):
        super().__init__()

        self.encoder = encoder

        self.decoder = decoder

        self.output_dim = self.decoder.output_dim
        
        print("output_dim=")
        print(self.output_dim)

        # optimizer for NN pars and likelihood noise
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.device = device

        self.to(device)


    def forward(self, data_subset, beta=1.0, device="cpu"):
        # we assume data_subset containts two elements
        Y, c = data_subset
        Y, c = Y.to(device), c.to(device)

        # encode
        mu_z, sigma_z = self.encoder(Y, c)

        print("mu_z-forward.shape")
        print(mu_z.shape)

        print(mu_z)
        
        mu_z1 = mu_z[:,0].reshape(-1,1)  # to go from 64,2 to [64,1] (otherwise [64])
        
        print(mu_z1)
        
        sigma_z1 = sigma_z[:,0].reshape(-1,1)
        mu_z2 = mu_z[:,1].reshape(-1,1)
        sigma_z2 = sigma_z[:,1].reshape(-1,1)
        eps_1 = torch.randn_like(mu_z1)
        eps_2 = torch.randn_like(mu_z2)
        z1 = mu_z1 + sigma_z1 * eps_1
        z2 = mu_z2 + sigma_z2 * eps_2

        print("mu_z1-forward.shape")
        print(mu_z1.shape)

        print("mu_z2-forward.shape")
        print(mu_z2.shape)
        # decode
        #   def forward(self, z1, z2, c):
        # !!!!!! modif for 2 LDs !!!
        y_pred = self.decoder.forward(z1, z2, c)
        
#         return total_loss, penalty, int_z1, int_z2, int_c, int_cz1_dc, int_cz1_dz1,\
#                      int_cz2_dc, int_cz2_dz2, int_z1z2_dz1, int_z1z2_dz2,\
        decoder_loss, penalty, int_z1, int_z2, int_c,\
        int_cz1_dc, int_cz1_dz1,\
        int_cz2_dc, int_cz2_dz2, int_z1z2_dz1,\
        int_z1z2_dz2 = self.decoder.loss(y_pred, Y)

        # loss function
        # modif for 2 LDs !!!
        #VAE_KL_loss = KL_standard_normal(mu_z, sigma_z)
        # def KL_standard_normal_2ld(mu_1, mu_2, sigma_1, sigma_2):
        VAE_KL_loss = KL_standard_normal_2ld(mu_z1, mu_z2, sigma_z1, sigma_z2)

        # Note that when this loss (neg ELBO) is calculated on a subset (minibatch),
        # we should scale it by data_size/minibatch_size, but it would apply to all terms
        total_loss = decoder_loss + beta * VAE_KL_loss

        return total_loss, int_z1, int_z2, int_c,\
        int_cz1_dc, int_cz1_dz1,\
        int_cz2_dc, int_cz2_dz2, int_z1z2_dz1,\
        int_z1z2_dz2

    def calculate_test_loglik(self, Y, c):
        """
        maps (Y, x) to z and calculates p(y* | x, z_mu)
        :param Y:
        :param c:
        :return:
        """
        mu_z, sigma_z = self.encoder(Y, c)
        # !!!! CHANGED HERE -> we have 2 
        mu_z1 = mu_z[:,0].reshape(-1,1)
        mu_z2 = mu_z[:,1].reshape(-1,1)
        #    def forward(self, z1, z2, c):
        Y_pred = self.decoder.forward(mu_z1, mu_z2, c)

        return self.decoder.loglik(Y_pred, Y)


    def optimize(self, data_loader, augmented_lagrangian_lr, n_iter=50000, 
                 logging_freq=20, logging_freq_int=100, temperature_start=4.0,
                 temperature_end=0.2, lambda_start=None, lambda_end=None, verbose=True):

        # sample size
        N = len(data_loader.dataset)

        # number of iterations = (numer of epochs) * (number of iters per epoch)
        n_epochs = ceil(n_iter / len(data_loader))
        if verbose:
            print(f"Fitting Neural Decomposition.\n\tData set size {N}. # iterations = {n_iter} (i.e. # epochs <= {n_epochs})\n")

        loss_values = np.zeros(ceil(n_iter // logging_freq))

        if self.decoder.has_feature_level_sparsity:
            temperature_grid = torch.linspace(temperature_start, temperature_end, steps=n_iter // 10, 
                                              device=self.device)

        if lambda_start is None:
            lambda_start = self.decoder.lambda0
            lambda_end = self.decoder.lambda0
        lambda_grid = torch.linspace(lambda_start, lambda_end, steps=n_iter // 10, device=self.device)

        # get shapes for integrals
        _int_z1, _int_z2, _int_c,\
        _int_cz1, _int_cz2, _int_z1z2 = self.decoder.calculate_integrals_numpy()
        # log the integral values
        n_logging_steps = ceil(n_iter // logging_freq_int)
        int_z1_values = np.zeros([n_logging_steps, _int_z1.shape[0], self.output_dim])
        int_z2_values = np.zeros([n_logging_steps, _int_z2.shape[0], self.output_dim])
        int_c_values = np.zeros([n_logging_steps, _int_c.shape[0], self.output_dim])
        int_cz1_values = np.zeros([n_logging_steps, _int_cz1.shape[0], self.output_dim])
        int_cz2_values = np.zeros([n_logging_steps, _int_cz2.shape[0], self.output_dim])
        int_z1z2_values = np.zeros([n_logging_steps, _int_z1z2.shape[0], self.output_dim])


        iteration = 0
        for epoch in range(n_epochs):

            for batch_idx, data_subset in enumerate(data_loader):

                if iteration >= n_iter:
                    break
                import pickle, os
                outfolder="ND_TOY_EXAMPLE"
                outsuffix="_debug_optimize"
                filename = os.path.join(outfolder, 'data_subset'+ outsuffix +'.sav')
                pickle.dump(data_subset, open(filename, 'wb'))
                print("... written: " + filename )
                
                loss, int_z1, int_z2, int_c,\
        int_cz1_dc, int_cz1_dz1,\
        int_cz2_dc, int_cz2_dz2, int_z1z2_dz1,\
        int_z1z2_dz2 = self.forward(data_subset, beta=1.0, device=self.device)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.decoder.has_feature_level_sparsity:
                    self.decoder.set_temperature(temperature_grid[iteration // 10])
                self.decoder.lambda0 = lambda_grid[iteration // 10]

                # update for BDMM
                with torch.no_grad(): # Context-manager that disabled gradient calculation.
                    self.decoder.Lambda_z1 += augmented_lagrangian_lr * int_z1
                    self.decoder.Lambda_z2 += augmented_lagrangian_lr * int_z2
                    self.decoder.Lambda_c += augmented_lagrangian_lr * int_c
                    self.decoder.Lambda_cz1_z1 += augmented_lagrangian_lr * int_cz1_dc
                    self.decoder.Lambda_cz1_c += augmented_lagrangian_lr * int_cz1_dz1
                    self.decoder.Lambda_cz2_z2 += augmented_lagrangian_lr * int_cz2_dc
                    self.decoder.Lambda_cz2_c += augmented_lagrangian_lr * int_cz2_dz2                    
                    self.decoder.Lambda_z1z2_z2 += augmented_lagrangian_lr * int_z1z2_dz1
                    self.decoder.Lambda_z1z2_z1 += augmented_lagrangian_lr * int_z1z2_dz2                    


# init version:
 #Lambda_cz_1 = Variable(lambda0*torch.ones(self.n_grid_z, ... * int_cz_dc  ==> Lambda_cz1_z1 ... n_grid_z1
 #f.Lambda_cz_2 = Variable(lambda0*torch.ones(self.n_grid_c,  * * int_cz_dz


                # logging for the loss function
                if iteration % logging_freq == 0:
                    index = iteration // logging_freq
                    loss_values[index] = loss.item()

                # logging for integral constraints
                if iteration % logging_freq_int == 0:
                    int_z1, int_z2, int_c,\
                    int_cz1, int_cz2, int_z1z2 = self.decoder.calculate_integrals_numpy()

                    index = iteration // logging_freq_int
                    int_z1_values[index, :] = int_z1
                    int_z2_values[index, :] = int_z2
                    int_c_values[index, :] = int_c
                    int_cz1_values[index, :] = int_cz1
                    int_cz2_values[index, :] = int_cz2
                    int_z1z2_values[index, :] = int_z1z2

                if verbose and iteration % 500 == 0:
                    print(f"\tIter {iteration:5}.\tTotal loss {loss.item():.3f}")

                iteration += 1

        # collect all integral values into one array
        integrals = np.hstack([int_z1_values, int_z2_values, int_c_values,\
                               int_cz1_values, int_cz2_values,int_z1z2_values ]).\
                                    reshape(n_iter // logging_freq_int, -1).T

        return loss_values, integrals

class CVAE_with_fixed_z(CVAE):
    """
    Same as the above CVAE class, but assuming a fixed latent variable z, thus effectively only training the decoder.
    We assume z is given by the data_loader, i.e. we assume it returns tuples (Y, c, z)
    """

    def __init__(self, decoder, lr):
        super().__init__(encoder=None, decoder=decoder, lr=lr)

    def forward(self, data_subset, beta=1.0):
        # we assume data_subset containts three elements
        Y, c, z = data_subset

        # decoding step
        y_pred = self.decoder.forward(z, c)
        
        decoder_loss, penalty, int_z1, int_z2, int_c, int_cz1_dc, int_cz1_dz1,\
                     int_cz2_dc, int_cz2_dz2, int_z1z2_dz1, int_z1z2_dz2,\
                      = self.decoder.loss(y_pred, Y)


        # no KL(q(z) | p(z)) term because z fixed
        total_loss = decoder_loss

        return total_loss, int_z1, int_z2, int_c, int_cz1_dc, int_cz1_dz1,\
                     int_cz2_dc, int_cz2_dz2, int_z1z2_dz1, int_z1z2_dz2
import torch
import torch.nn as nn
import torch.nn.Functional as F

import copy

#TODO: docstrings and train loop (can't use from toolbox)

class encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, n_classes, hidden_dim=1, n_layers=0, dropout_p=0.2, input_block=None, hidden_block=None, y_block=None, z_block=None):
        super(encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # create structure
        if (input_block == None):
            self.input_block = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_p)
            )
        else:
            self.input_block = input_block

        blocks = nn.ModuleList()
        if (hidden_block == None):
            hidden_block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_p)
            )        

        for i in range(n_layers):
            blocks.append(copy.deepcopy(hidden_block))
        
        self.blocks = nn.Sequential(*blocks)

        if (y_block == None):
            self.y_block = nn.Sequential(
                nn.Linear(hidden_dim, n_classes),
                nn.Softmax()
            )
        else:
            self.y_block = y_block

        if (z_block == None):
            self.z_block = nn.Sequential(
                nn.Linear(hidden_dim, latent_dim)
            )
        else:
            self.z_block = z_block

    def forward(self, input):
        x = self.input_block(input)
        x = self.blocks(x)

        z = self.z_block(x)
        y_ = self.y_block(x)

        return z, y_


class decoder(nn.Module):
    def __init__(self,
                 latent_dim,
                 output_dim,
                 hidden_dim=1,
                 n_layers=0,
                 dropout_p=0.5,
                 input_block=None,
                 hidden_block=None,
                 output_block=None):
        super(decoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # create structure
        if (input_block == None):
            self.input_block = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.BatchNorm1d(hidden_dim)
            )
        else:
            self.input_block = input_block

        if (hidden_block == None):
            hidden_block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.BatchNorm1d(hidden_dim)
            )

        blocks = nn.ModuleList()
        for i in range(n_layers):
            blocks.append(copy.deepcopy(hidden_block))
        self.blocks = nn.Sequential(*blocks)

        if (output_block == None):
            self.output_block = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.ReLU(),
            )
        else:
            self.output_block = output_block

    def forward(self, input):
        x = self.input_block(input)
        x = self.blocks(x)
        return self.output_block(x)


class discrim(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.2):
        super(discrim, self).__init__()
        self.input_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        blocks = nn.ModuleList()
        for i in range(n_layers):
            blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout),
                nn.ReLU()
            ))
        self.blocks = nn.Sequential(*blocks)
        
        self.out_block = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.input_block(input)
        x = self.blocks(x)
        return self.out_block(x)


class rand_sampler():
    def __init__(self, dim, mu=None, sig=None, mode='cat', prob_tensor=None):
        """
        dim: dimensionality for continuous random variable/num classes for cat mode
        """
        assert mode in ('cat', 'cont')
        self.mode = mode
        self.dim = dim

        if mode is 'cont':
            self.mu = mu
            self.sig = sig
        elif mode is 'cat':
            prob_tensor = torch.ones(dim)/dim if prob_tensor is None else prob_tensor
            self.sampler = torch.distributions.OneHotCategorical(prob_tensor)
    
    def sample(self, n_samples):
        if self.mode is 'cont':
            return self.mu + (torch.randn(n_samples, self.dim) * self.sig)
        elif self.mode is 'cat':
            return self.sampler.sample((n_samples,))


class semi_sup_AAE(nn.Module):
    def __init__(self, encoder, decoder, discrim_z, discrim_y, num_classes, z_rand, y_rand):
        super(semi_sup_AAE, self).__init__()
        self.enc = encoder
        self.dec = decoder
        self.d_z = discrim_z
        self.d_y = discrim_y
        self.num_classes = num_classes
        self.z_rand = z_rand
        self.y_rand = y_rand

    def get_reconst(self, input):
        z, y_ = self.enc(input)
        x_ = self.dec(torch.cat((z, y_), dim=1))
        return x_

    def get_latents(self, input):
        z, y_ = self.enc(input)
        return z, y_
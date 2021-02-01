import torch
import torch.nn as nn
import copy
import yaml
# TODO: docstrings and train loop (can't use from toolbox)


class encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, n_classes, hidden_dim=1, n_layers=0, dropout_p=0.2, input_block=None, blocks=None, y_block=None, y_predictor=None, z_block=None):
        super(encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # create structure
        if (input_block is None):
            self.input_block = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_p)
            )
        else:
            self.input_block = input_block

        if (blocks is None):
            blocks = nn.ModuleList()
            hidden_block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_p)
            )
            for i in range(n_layers):
                blocks.append(copy.deepcopy(hidden_block))
            self.blocks = nn.Sequential(*blocks)
        else:
            self.blocks = blocks

        if (y_block is None):
            self.y_block = nn.Sequential(
                nn.Linear(hidden_dim, latent_dim),
                nn.Softmax()
            )
        else:
            self.y_block = y_block

        if (y_predictor is None):
            self.y_predictor = nn.Sequential(
                nn.Linear(latent_dim, n_classes),
                nn.Softmax()
            )
        else:
            self.y_predictor = y_predictor

        if (z_block is None):
            self.z_block = nn.Sequential(
                nn.Linear(hidden_dim, latent_dim)
            )
        else:
            self.z_block = z_block

    def forward(self, input):
        x = self.input_block(input)
        x = self.blocks(x)

        z = self.z_block(x)
        _y = self.y_block(x)
        y_ = self.y_predictor(_y)

        return z, _y, y_


class decoder(nn.Module):
    def __init__(self,
                 latent_dim,
                 output_dim,
                 hidden_dim=1,
                 n_layers=0,
                 dropout_p=0.5,
                 input_block=None,
                 blocks=None,
                 output_block=None):
        super(decoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # create structure
        if (input_block is None):
            self.input_block = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.BatchNorm1d(hidden_dim)
            )
        else:
            self.input_block = input_block

        if (blocks is None):
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
        else:
            self.blocks = blocks

        if (output_block is None):
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
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 n_layers=1,
                 dropout=0.2,
                 input_block=None,
                 blocks=None,
                 output_block=None):
        super(discrim, self).__init__()

        if input_block is None:
            self.input_block = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Dropout(dropout),
                nn.ReLU()
            )
        else:
            self.input_block = input_block

        if blocks is None:
            blocks = nn.ModuleList()
            for i in range(n_layers):
                blocks.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Dropout(dropout),
                    nn.ReLU()
                ))
            self.blocks = nn.Sequential(*blocks)
        else:
            self.blocks = blocks

        if output_block is None:
            self.output_block = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        else:
            self.output_block = output_block

    def forward(self, input):
        x = self.input_block(input)
        x = self.blocks(x)
        return self.output_block(x)


class rand_sampler():
    def __init__(self, dim, mu=None, sig=None, mode='cat', prob_tensor=None, portion=None):
        """
        dim: dimensionality for continuous random variable/num classes for cat mode
        """
        assert mode in ('cat', 'cont', 'multi')
        self.mode = mode
        self.dim = list(dim)

        if mode == 'cont':
            self.mu = mu
            self.sig = sig

        elif mode == 'cat':
            prob_tensor = torch.ones(dim)/dim if prob_tensor is None else prob_tensor
            self.sampler = torch.distributions.OneHotCategorical(prob_tensor)

        elif mode == 'multi':
            self.portion = portion

    def sample(self, n_samples):
        if self.mode == 'cont':
            return self.mu + (torch.randn(n_samples, *self.dim) * self.sig)
        if self.mode == 'cat':
            return self.sampler.sample((n_samples, ))
        if self.mode == 'multi':
            return 1*(torch.rand(n_samples, *self.dim) < self.portion)


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

    def forward(self, input):
        z, _y, y_ = self.enc(input)
        x_ = self.dec(torch.cat((z, _y), dim=1))
        return x_, z, y_

    def get_reconst(self, input):
        z, _y, y_ = self.enc(input)
        x_ = self.dec(torch.cat((z, _y), dim=1))
        return x_

    def get_latents(self, input):
        z, _y, y_ = self.enc(input)
        return z, _y, y_

    def discriminate(self, input, mode):
        assert mode in 'yz'
        if mode is 'y':
            return self.d_y(input)
        elif mode is 'z':
            return self.d_z(input)

    def get_samples(self, n_samples):
        return self.z_rand.sample(n_samples), self.y_rand.sample(n_samples)


def eval_type(layer_name):
    if layer_name[:3] != 'nn.':
        layer_name = 'nn.' + layer_name
    return eval(layer_name)


def construct_params(param_dict, cfg_dict):
    for blk in cfg_dict.keys():
        if blk == 'params':
            continue

        block = nn.ModuleList()
        for layer in cfg_dict[blk]:
            layer_type = eval_type(layer['type'])
            block.append(layer_type(**layer['params']))

            if layer.get('nlinear') is not None:
                nlinear = layer['nlinear']
                nlinear_type = eval_type(nlinear['type'])
                block.append(nlinear_type(**nlinear['params']))

            if layer.get('dropout') is not None:
                block.append(nn.Dropout(**layer['dropout']))

            if layer.get('norm') is not None:
                norm = layer['norm']
                norm_type = eval_type(norm['type'])
                block.append(norm_type(**norm['params']))
        block = nn.Sequential(*block)
        param_dict[blk] = block
    return param_dict


def construct_model(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.load(f)
        f.close()

    # encoder
    enc_cfg = cfg['enc']
    enc_params = enc_cfg['params']
    enc_params = construct_params(enc_params, enc_cfg)
    enc = encoder(**enc_params)

    # decoder
    dec_cfg = cfg['dec']
    dec_params = dec_cfg['params']
    dec_params = construct_params(dec_params, dec_cfg)
    dec = decoder(**dec_params)

    # discriminator z
    dz_cfg = cfg['disc_z']
    dz_params = dz_cfg['params']
    dz_params = construct_params(dz_params, dz_cfg)
    d_z = discrim(**dz_params)

    # discriminator y
    dy_cfg = cfg['disc_y']
    dy_params = dy_cfg['params']
    dy_params = construct_params(dy_params, dy_cfg)
    d_y = discrim(**dy_params)

    # z sampler
    z_rand = rand_sampler(**cfg['z_rand']['params'])

    # y sampler
    y_rand = rand_sampler(**cfg['y_rand']['params'])

    return semi_sup_AAE(enc, dec, d_z, d_y, enc_params['n_classes'], z_rand,
                        y_rand)

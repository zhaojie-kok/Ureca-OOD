import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parameter as param
import torch.nn.functional as F
import torch.optim as optim

import calc

class quantiser(nn.Module):
    def __init__(self, num_on_states, latent_dim, dist_metric, on_states = None, label_map = None):
        if (label_map != None):
            self.label_map = label_map
        
        # the off state will default to a gaussian with mean 0 and variance 0.1
        self.off_state = torch.stack([torch.zeros(latent_dim), torch.ones(latent_dim)*0.1], dim=-1).view(1, latent_dim, 2)

        if (on_states != None):
            assert isinstance(on_states, param.Parameter) and on_states.requires_grad
            assert on_states.shape == torch.Size([num_on_states, latent_dim, 2])
            self.on_states = on_states
        else:
            # here there is a choice to either randomise learnable distribution parameters
            # or an anchoring approach (yolo) can be adopted
            # we will use the randomised approach since the goal is not to investigate the quantisation
            on_mus = torch.randn(num_on_states, latent_dim)
            on_sigs = torch.rand(num_on_states, latent_dim)
            on_states = torch.stack([on_mus, on_sigs], dim=-1)
            self.on_states = param.Parameter(on_states, requires_grad=True)

        self.dist_metric = dist_metric #TODO: consider changing to maha cosine dist instead but need to find new off state. Also, check if using maha dist will cause hypersphere collapse
        
    def forward(self, input_mu, input_sig, mode):
        # cosine problems: where is off? Since off cannot be at 0 anymore
        # maha dist prob: possibility that the means will collapse to origin
        # also consider using KL divergence and then sampling the quantised distribution instead
        dists = []
        for i in range(self.on_states.shape[0]):
            mus = self.on_states[i, :, 0]
            sigs = self.on_states[i, :, 1]
            dists.append(self.dist_metric(input_mu, input_sig, mus, sigs))
        dists = torch.cat(dists)
        
        assert mode in ('softmin', 'argmin')
        if (mode == 'softmin'):
            ps = F.softmin(dists, dim=0)
            quantised = sum(self.on_states[i, :, 0]*ps[i] for i in range(len(ps)))
        elif (mode == 'argmin'):
            ind = torch.argmin(dists)
            quantised = self.on_states[ind, :, 0]
        
        # should be measuring distance
        # However, mse is just a scaled down euclidean dist anyway
        # need to change to KL div if input_w is a distribution instead
        loss_enc = F.mse_loss(quantised.view(input_mu.shape).detach(), input_mu)
        loss_ref = F.mse_loss(quantised.view(input_mu.shape), input_mu.detach())

        return quantised, loss_enc, loss_ref, dists

class encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout_p=0.5):
        super(encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # create structure
        self.input_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim), # maybe not since a vae shouldnt be trying to achieve regularisation
            nn.Dropout(dropout_p) # maybe not since a vae shouldnt be trying to achieve regularisation
        )

        hidden_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim), # maybe not since a vae shouldnt be trying to achieve regularisation
            nn.Dropout(dropout_p) # maybe not since a vae shouldnt be trying to achieve regularisation
        )
        blocks = nn.ModuleList()
        for i in range(n_layers):
            blocks.append(hidden_block)
        self.blocks = nn.Sequential(*blocks)

        self.mu_block = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim), # maybe not since a vae shouldnt be trying to achieve regularisation
        )

        self.sigma_block = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim), # maybe not since a vae shouldnt be trying to achieve regularisation
        )

    def forward(self, input):
        x = self.input_block(input)
        x = self.blocks(x)
        return self.mu_block(x), self.sigma_block(x)

class encoderZ(encoder):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout_p=0.5, samples=1, mean_fn=None, mean_fn_args=None):
        super(encoderZ, self).__init__(input_dim, hidden_dim, latent_dim, n_layers, dropout_p)
        self.samples = samples

        self.mean_fn = calc.arith_mean if mean_fn is None else mean_fn
        self.mean_fn_args = mean_fn_args


    def forward(self, input):
        x = self.input_block(input)
        x = self.blocks(x)
        mu = self.mu_block(x)
        sigma = self.sigma_block(x)

        samples = []
        for i in range(self.samples):
            sample = torch.randn_like(sigma) * sigma + mu
            samples.append(sample)
        if (self.mean_fn_args != None):
            mean_sample = self.mean_fn(samples, self.mean_fn_args)
        else :
            mean_sample = self.mean_fn(samples)
        return mu, sigma, mean_sample, samples

# To be used only if labels are to be provided during training
class encoderW(encoderZ):
    def __init__(self, input_dim, label_dim, hidden_dim, latent_dim, n_layers, dropout_p=0.5, samples=1, mean_fn=None, mean_fn_args=None):
        super(encoderW, self).__init__(input_dim+label_dim, hidden_dim, latent_dim, n_layers, dropout_p, samples, mean_fn, mean_fn_args)
        self.samples = samples

        def arith_mean(tensors):
            return sum(tensors)/len(tensors)
        self.mean_fn = arith_mean if mean_fn is None else mean_fn
        self.mean_fn_args = mean_fn_args

    
    def forward(self, x, y):
        input = torch.cat([x, y], dim=-1)
        return encoderZ.forward(self, input)


class decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, n_layers, dropout_p = 0.5):
        super(decoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # create structure
        self.input_block = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim), # maybe not since a vae shouldnt be trying to achieve regularisation
            nn.Dropout(dropout_p) # maybe not since a vae shouldnt be trying to achieve regularisation
        )

        hidden_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim), # maybe not since a vae shouldnt be trying to achieve regularisation
            nn.Dropout(dropout_p) # maybe not since a vae shouldnt be trying to achieve regularisation
        )
        blocks = nn.ModuleList()
        for i in range(n_layers):
            blocks.append(hidden_block)
        self.blocks = nn.Sequential(*blocks)

        self.output_block = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim), # maybe not since a vae shouldnt be trying to achieve regularisation
        )
    
    def forward(self, input):
        x = self.input_block(input)
        x = self.blocks(x)
        return self.output_block(x)

# groups format: {group name: [label names]}
class classifier(decoder):
    def __init__(self, latent_dim, hidden_dim, n_layers, groups, dropout_p = 0.5):
        super(classifier, self).__init__(latent_dim, hidden_dim, hidden_dim, n_layers, dropout_p)
        assert all(isinstance(g, list) for g in groups.values())
        self.groups = groups

        group_blocks = nn.ModuleDict()
        for g in groups.items():
            name, classes = g
            group_blocks[name] = nn.Sequential(
                nn.Linear(hidden_dim, len(classes)),
                nn.Softmax()
            )
        self.group_blocks = group_blocks

    
    def forward(self, input):
        x = self.input_block(input)
        x = self.blocks(x)
        x = self.output_block(x)

        outputs = {name:self.group_blocks[name](x) for name, _ in self.groups}
        return outputs


class csvae(nn.Module):
    def __init__(self, encoder_z, decoder, encoder_w_args, classifier_z, classifier_w, quantiser):
        super(csvae, self).__init__()

        self.encoder_z = encoder_z
        self.decoder = decoder
        self.classifier_z = classifier_z
        self.encoders_w = nn.ModuleDict()
        for g in classifier_w.groups:
            self.encoders_w[g] = encoder_z(encoder_w_args) # using encoder_z class since not feeding labels
        self.quantiser = quantiser
    
    def forward(self, x, loss_fn = None):
        z = self.encoder_z(x)

        # leave each w in a group first before concatenating to allow attribute manipulation
        ws = {}
        for group in self.encoders_w.keys():
            ws[group] = self.encoders_w[group](x)
        w = torch.cat([i for i in ws.values()], dim=1)
        x_ = self.decoder(torch.cat([z, w], dim=1))
        y_z = self.classifier_z(z)
        y_w = self.classifier_w(w)

        return x_, y_z, y_w, ws, z
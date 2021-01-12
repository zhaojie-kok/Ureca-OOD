import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parameter as param
import torch.nn.functional as F
import torch.optim as optim

import copy
import calc


class quantiser(nn.Module):
    def __init__(self,
                 num_on_states,
                 latent_dim,
                 dist_metric,
                 on_states=None,
                 label_map=None,
                 n_states=1):
        super(quantiser, self).__init__()
        self.n_states = n_states
        if (label_map != None):
            self.label_map = label_map

        # the off state will default to a gaussian with mean 0 and variance 0.1
        # off_state = torch.stack([torch.zeros(latent_dim), torch.ones(latent_dim)*0.1], dim=-1).view(1, latent_dim, 2)

        if (on_states != None):
            assert isinstance(on_states, param.Parameter)  # and on_states.requires_grad
            assert on_states.shape == torch.Size(
                [num_on_states * n_states, latent_dim, 2])
            self.on_states = on_states
        else:
            # here there is a choice to either randomise learnable distribution parameters
            # or an anchoring approach (yolo) can be adopted
            # we will use the randomised approach since the goal is not to investigate the quantisation
            on_mus = torch.randn(num_on_states * n_states, latent_dim) * 3
            on_sigs = torch.exp(
                torch.randn(num_on_states * n_states, latent_dim) * 2) * 3
            on_states = torch.stack([on_mus, on_sigs], dim=-1)
            self.on_states = param.Parameter(on_states, requires_grad=True)

        self.dist_metric = dist_metric  #TODO: consider changing to maha cosine dist instead but need to find new off state. Also, check if using maha dist will cause hypersphere collapse

    def forward(self, input_mu, input_logsig, mode, y=None):
        # cosine problems: where is off? Since off cannot be at 0 anymore
        # maha dist prob: possibility that the means will collapse to origin
        # also consider using KL divergence and then sampling the quantised distribution instead
        input_sig = torch.exp(input_logsig)
        dists = []
        for i in range(self.on_states.shape[0]):
            mus = self.on_states[i, :, 0]
            sigs = self.on_states[i, :, 1]
            dists.append(self.dist_metric(input_mu, input_sig, mus, sigs))

        dists = torch.stack(dists).view(input_mu.size(0), -1)
        _dists = []

        # states = torch.cat((self.on_states,), dim=0)
        assert mode in ('softmin', 'argmin')
        states = []
        for i in range(0, self.on_states.size(0), self.n_states):
            sub_states = self.on_states[i:i + self.n_states]
            if (mode == 'argmin'):
                mins = torch.argmin(dists[:, i:i + self.n_states], dim=1)
                _states = torch.stack([sub_states[m] for m in mins], dim=0)
                _dists.append(dists[:, i:i + self.n_states].gather(
                    1, mins.unsqueeze(1)).squeeze(-1))
            elif (mode == 'softmin'):
                softs = F.softmin(dists[:, i:i + self.n_states], dim=1)
                _states = torch.matmul(softs, sub_states.permute(1, 0, 2)).permute(1, 0, 2)
                _dists.append(
                    torch.sum(softs * dists[:, i:i + self.n_states], dim=1))
            states.append(_states)

        states = torch.stack(states, dim=1)
        dists = torch.stack(_dists, dim=1)

        if (mode == 'argmin'):
            if (y == None):
                ind = torch.argmin(dists, dim=1)
            else:
                ind = y
            quantised_mu = torch.stack(
                [states[i, ind[i], :, 0] for i in range(ind.size(0))])
            quantised_sig = torch.stack(
                [states[i, ind[i], :, 1] for i in range(ind.size(0))])
            dist = dists.gather(1, ind.unsqueeze(1))
        elif (mode == 'softmin'):
            ps = F.softmin(dists, dim=0)
            quantised_mu = torch.matmul(ps, states[:, :, :, 0])
            quantised_sig = torch.matmul(ps, states[:, :, :, 1])
            dist = torch.sum(dists * ps, dim=1)

        return (quantised_mu, quantised_sig), dists, dist


class encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=1, n_layers=0, dropout_p=0.5, input_block=None, hidden_block=None, mu_block=None, sigma_block=None):
        super(encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # create structure
        if (input_block == None):
            self.input_block = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
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

        if (mu_block == None):
            self.mu_block = nn.Sequential(
                nn.Linear(hidden_dim, latent_dim),
            )
        else:
            self.mu_block = mu_block

        if (sigma_block == None):
            self.sigma_block = nn.Sequential(
                nn.Linear(hidden_dim, latent_dim),
            )
        else:
            self.sigma_block = sigma_block

    def forward(self, input):
        x = self.input_block(input)
        x = self.blocks(x)
        return self.mu_block(x), self.sigma_block(x)

class encoderZ(encoder):
    def __init__(self, input_dim, latent_dim, hidden_dim=1, n_layers=0, dropout_p=0.5, samples=1, input_block=None, hidden_block=None, mu_block=None, sigma_block=None, mean_fn=None, mean_fn_args=None):
        super(encoderZ, self).__init__( input_dim, latent_dim, hidden_dim, n_layers, dropout_p, input_block, hidden_block, mu_block, sigma_block)
        self.samples = samples

        def arith_mean(tensors):
            return sum(tensors)/len(tensors)
        self.mean_fn = arith_mean if mean_fn is None else mean_fn
        self.mean_fn_args = mean_fn_args


    def forward(self, input):
        x = self.input_block(input)
        x = self.blocks(x)
        mu = self.mu_block(x)
        log_sigma = self.sigma_block(x)
        sigma = torch.exp(log_sigma)

        samples = []
        for i in range(self.samples):
            sample = torch.randn_like(sigma) * sigma + mu
            samples.append(sample)
        if (self.mean_fn_args != None):
            mean_sample = self.mean_fn(samples, self.mean_fn_args)
        else :
            mean_sample = self.mean_fn(samples)
        return mu, sigma, mean_sample, samples

class encoderW(encoderZ):
    def __init__(self, input_dim, latent_dim, quantiser, hidden_dim=1, n_layers=0, dropout_p=0.5, samples=1, input_block=None, hidden_block=None, mu_block=None, sigma_block=None, mean_fn=None, mean_fn_args=None):
        super(encoderW, self).__init__(input_dim, latent_dim, hidden_dim, n_layers, dropout_p, samples, input_block, hidden_block, mu_block, sigma_block, mean_fn, mean_fn_args)
        self.quantiser = quantiser

    def forward(self, x, quant_mode='softmin', y=None):
        x = self.input_block(x)
        x = self.blocks(x)
        mu = self.mu_block(x)
        log_sigma = self.sigma_block(x)
        sigma = torch.exp(log_sigma)

        quantised, dists, quant_dist = self.quantiser(mu, sigma, quant_mode, y=y)
        quant_mu = quantised[0]
        quant_sigma = quantised[1]

        samples = []
        for i in range(self.samples):
            sample = torch.randn_like(quant_sigma) * quant_sigma + quant_mu
            samples.append(sample)
        if (self.mean_fn_args != None):
            mean_sample = self.mean_fn(samples, self.mean_fn_args)
        else :
            mean_sample = self.mean_fn(samples)

        # should be measuring distance
        # However, mse is just a scaled down euclidean dist anyway
        # need to change to KL div if input_w is a distribution instead
        loss_enc = KL(quant_mu.detach(), quant_sigma.detach(), mu, sigma)
        loss_ref = KL(quant_mu, quant_sigma, mu.detach(), sigma.detach())

        return mu, sigma, mean_sample, samples, quant_dist, dists, loss_enc, loss_ref


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

# groups format: {group name: [label names]}
class classifier(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, groups, dropout_p = 0.5):
        super(classifier, self).__init__()
        assert all(isinstance(g, (str, int)) for g in groups)

        self.decoder = decoder(latent_dim, hidden_dim, hidden_dim, n_layers, dropout_p)
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
        x = self.decoder(input)

        outputs = {name:self.group_blocks[name](x) for name, _ in self.groups.items()}
        return outputs


class csvae(nn.Module):
    def __init__(self,
                 encoder_z,
                 decoder,
                 encoders_w,
                 classifier_z,
                 classifier_w,
                 quantiser,
                 gamma=0.1):
        super(csvae, self).__init__()

        self.encoder_z = encoder_z
        self.decoder = decoder
        self.classifier_z = classifier_z
        self.encoders_w = nn.ModuleDict(encoders_w)
        self.classifier_w = classifier_w
        assert 0 < gamma < 1
        self.gamma = gamma

    def getMainParams(self):
        ez = list(self.encoder_z.parameters())
        dec = list(self.decoder.parameters())
        enw = list(self.encoders_w.parameters())
        cw = list(self.classifier_w.parameters())

        return ez + dec + enw + cw

    def getClassZParams(self):
        return list(self.classifier_z.parameters())

    def forward(self, x, loss_fn=None, use='mu', quant_mode='softmin', y=None):
        assert use in ['mu', 'sigma', 'mean sample', 'samples']
        mu_z, sig_z, mean_z, samples_z = self.encoder_z(
            x)  # mu, sigma, mean_sample, samples
        z = {
            'mu': mu_z,
            'sigma': sig_z,
            'mean sample': mean_z,
            'samples': samples_z
        }

        # leave each w in a group first before concatenating to allow attribute manipulation
        mu_ws = {}
        sig_ws = {}
        mean_ws = {}
        samples_ws = {}
        quant_dists = {}
        dist = {}
        quant_loss = None

        for group in self.encoders_w.keys():
            quant_mu, quant_sigma, mean_sample, samples, quant_dist, dists, quant_loss_enc, quant_loss_ref = self.encoders_w[
                group](x, quant_mode=quant_mode, y=y)
            mu_ws[group] = quant_mu
            sig_ws[group] = quant_sigma
            mean_ws[group] = mean_sample
            samples_ws[group] = samples
            quant_dists[group] = dists
            dist[group] = quant_dist
            if (quant_loss == None):
                quant_loss = torch.mean(quant_loss_enc + quant_loss_ref)
            else:
                quant_loss = quant_loss + torch.mean(
                    quant_loss_enc + quant_loss_ref
                )  # note that this makes the loss scale according to number of groups. Should consider rescaling based on groups
        ws = {
            'mu': mu_ws,
            'sigma': sig_ws,
            'mean sample': mean_ws,
            'samples': samples_ws
        }

        if (len(ws[use]) > 1):
            w = torch.cat([i for i in ws[use].values()], dim=1)
        else:
            w = list(ws[use].values())[0]
        x_ = self.decoder(
            torch.cat([z[use] * self.gamma, w * (1 - self.gamma)], dim=1))
        y_z = self.classifier_z(z[use].detach())
        y_w = self.classifier_w(w)

        return x_, y_z, y_w, ws, z, quant_loss, quant_dists, dist
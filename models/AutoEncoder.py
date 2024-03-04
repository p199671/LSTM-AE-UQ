import torch
from torch import nn


def make_model(config):
    return AutoEncoder(config)


class AutoEncoder(nn.Module):
    def __init__(self, config):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(config)
        self.decoder_mean = Decoder(config)

        self.is_aleatoric = config['model']['mode'] in ['aleatoric', 'combined']
        if self.is_aleatoric:
            self.decoder_var = Decoder(config)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder_mean(z)
        if self.is_aleatoric:
            x_var = self.decoder_var(z)
            results = {'mean': x_hat, 'aleatoric_var': x_var}
        else:
            results = {'mean': x_hat}

        return results


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        input_dim = config['data']['seq_len']
        embedding_dim = config['model']['embedding_dim']
        hidden_dim = config['model']['hidden_dim']

        self.use_dropout = config['model']['mode'] in ['epistemic', 'combined']
        self.dropout_p = config['model']['dropout']

        layer_dims = [input_dim] + hidden_dim + [embedding_dim]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

            if i < self.num_layers - 1:
                self.layers.append(nn.Sigmoid())
            elif i == self.num_layers - 1:
                self.layers.append(nn.Tanh())

    def forward(self, x):
        x = x.squeeze(-1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if ((type(layer) == nn.modules.activation.Sigmoid or
                 type(layer) == nn.modules.activation.Tanh) and
                    (self.use_dropout and i < self.num_layers * 2 - 1)):
                x = torch.dropout(x, train=True, p=self.dropout_p)

        return x.unsqueeze(-1)


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        input_dim = config['model']['embedding_dim']
        hidden_dims = config['model']['hidden_dim'][::-1]
        output_dim = config['data']['seq_len']

        self.use_dropout = config['model']['mode'] in ['epistemic', 'combined']
        self.dropout_p = config['model']['dropout']

        layer_dims = [input_dim] + hidden_dims + [output_dim]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

            if i < self.num_layers - 1:
                self.layers.append(nn.Sigmoid())

    def forward(self, x):
        x = x.squeeze(-1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if (type(layer) == nn.modules.activation.Sigmoid and
                    (self.use_dropout and i < self.num_layers * 2 - 1)):
                x = torch.dropout(x, train=True, p=self.dropout_p)

        x = x.unsqueeze(-1)
        return x

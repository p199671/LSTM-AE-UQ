import torch
from torch import nn


def make_model(config):
    return LSTMAE(config)


class LSTMAE(nn.Module):
    """Implementation of a LSTM Autoencoder."""
    def __init__(self, config):
        super(LSTMAE, self).__init__()
        self.encoder = Encoder(config)
        self.decoder_mean = Decoder(config)

        self.is_aleatoric = config['model']['mode'] in ['aleatoric', 'combined']
        if self.is_aleatoric:
            self.decoder_var = Decoder(config)

    def forward(self, x):
        z = self.encoder(x)  # latent space
        x_hat = self.decoder_mean(z)  # reconstruction
        if self.is_aleatoric:
            x_var = self.decoder_var(z)  # aleatoric uncertainty
            results = {'mean': x_hat, 'aleatoric_var': x_var}
        else:
            results = {'mean': x_hat}

        return results


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        input_dim = len(config['data']['features'])
        embedding_dim = config['model']['embedding_dim']
        hidden_dim = config['model']['hidden_dim']

        self.use_dropout = config['model']['mode'] in ['epistemic', 'combined']
        self.dropout_p = config['model']['dropout']

        layer_dims = [input_dim] + hidden_dim + [embedding_dim]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = nn.LSTM(
                input_size=layer_dims[i],
                hidden_size=layer_dims[i + 1],
                num_layers=1,
                batch_first=True
            )
            self.layers.append(layer)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x, (h_n, _) = layer(x)
            if self.use_dropout and i < self.num_layers - 1:
                x = torch.dropout(x, train=True, p=self.dropout_p)
        return h_n.squeeze(0)


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        input_dim = config['model']['embedding_dim']
        hidden_dims = config['model']['hidden_dim'][::-1]
        self.seq_len = config['data']['seq_len']
        n_features = len(config['data']['features'])

        self.use_dropout = config['model']['mode'] in ['epistemic', 'combined']
        self.dropout_p = config['model']['dropout']

        layer_dims = [input_dim] + hidden_dims + [hidden_dims[-1]]
        num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.LSTM(
                input_size=layer_dims[i],
                hidden_size=layer_dims[i + 1],
                num_layers=1,
                batch_first=True
            )
            self.layers.append(layer)

        self.dense_matrix = nn.Parameter(torch.randn(hidden_dims[-1], n_features), requires_grad=True)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)

        for i, layer in enumerate(self.layers):
            x, (h_n, _) = layer(x)
            if self.use_dropout:
                x = torch.dropout(x, train=True, p=self.dropout_p)

        x = torch.matmul(x.squeeze(0), self.dense_matrix)  # .unsqueeze(0)

        return x

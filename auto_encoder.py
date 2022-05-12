from torch import nn, optim


class AutoEncoder(nn.Module):
    """ AutoEncoder model for dimensionality reduction.
    Parameters
    ----------
    in_shape [int] : input shape
    enc_shape [int] : desired encoded shape
    """

    def __init__(self, in_shape, enc_shape):
        super(AutoEncoder, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(in_shape, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, enc_shape),
        )

        self.decode = nn.Sequential(
            nn.BatchNorm1d(enc_shape),
            nn.Linear(enc_shape, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, in_shape)
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class ConvolutionalAutoEncoder(nn.Module):
    """ Convolutional AutoEncoder model for dimensionality reduction.
    Parameters
    ----------
    in_shape [int] : input shape
    enc_shape [int] : desired encoded shape
    """

    def __init__(self, in_shape, enc_shape):
        super(ConvolutionalAutoEncoder, self).__init__()

        self.encode = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8*in_shape, in_shape),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_shape, enc_shape)
        )

        self.decode = nn.Sequential(
            nn.BatchNorm1d(enc_shape),
            nn.Linear(enc_shape, in_shape),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_shape, 8*in_shape),
            nn.ReLU(),
            nn.Unflatten(1, (8, in_shape)),
            nn.ConvTranspose1d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(4, 1, kernel_size=3, stride=1, padding=1)
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class LSTMAutoEncoder(nn.Module):
    """ LSTM AutoEncoder model for dimensionality reduction.
    Parameters
    ----------
    in_shape [int] : input shape
    enc_shape [int] : desired encoded shape
    """

    def __init__(self, in_shape, enc_shape):
        super(LSTMAutoEncoder, self).__init__()

        # self.encode = nn.Sequential(
        #     nn.LSTM(in_shape, enc_shape, num_layers=1)
        # )
        #
        # self.decode = nn.Sequential(
        #     nn.LSTM(enc_shape, in_shape, num_layers=1)
        # )

        self.lin_enc1 = nn.Linear(in_shape, 32)
        self.relu_enc1 = nn.ReLU()
        self.dropout_enc1 = nn.Dropout(p=0.1)
        self.lstm_enc = nn.LSTM(32, 8, num_layers=1)
        self.lin_enc2 = nn.Linear(8, enc_shape)

        self.batch_norm = nn.BatchNorm1d(enc_shape)
        self.lin_dec1 = nn.Linear(enc_shape, 8)
        self.relu_dec1 = nn.ReLU()
        self.dropout_dec1 = nn.Dropout(p=0.1)
        self.lstm_dec = nn.LSTM(8, 32, num_layers=1)
        self.lin_dec2 = nn.Linear(32, in_shape)

    def encode(self, x):
        x = self.lin_enc1(x)
        x = self.relu_enc1(x)
        x = self.dropout_enc1(x)
        x = x.unsqueeze(1)
        x, (h_n, c_n) = self.lstm_enc(x)
        x = x.squeeze()
        x = self.lin_enc2(x)
        return x

    def decode(self, x):
        x = self.batch_norm(x)
        x = self.lin_dec1(x)
        x = self.relu_dec1(x)
        x = self.dropout_dec1(x)
        x = x.unsqueeze(1)
        x, (h_n, c_n) = self.lstm_dec(x)
        x = x.squeeze()
        x = self.lin_dec2(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class GRUAutoEncoder(nn.Module):
    """ GRU AutoEncoder model for dimensionality reduction.
    Parameters
    ----------
    in_shape [int] : input shape
    enc_shape [int] : desired encoded shape
    """

    def __init__(self, in_shape, enc_shape):
        super(GRUAutoEncoder, self).__init__()

        self.lin_enc1 = nn.Linear(in_shape, 32)
        self.relu_enc1 = nn.ReLU()
        self.dropout_enc1 = nn.Dropout(p=0.1)
        self.gru_enc = nn.GRU(32, 8, num_layers=1)
        self.lin_enc2 = nn.Linear(8, enc_shape)

        self.batch_norm = nn.BatchNorm1d(enc_shape)
        self.lin_dec1 = nn.Linear(enc_shape, 8)
        self.relu_dec1 = nn.ReLU()
        self.dropout_dec1 = nn.Dropout(p=0.1)
        self.gru_dec = nn.GRU(8, 32, num_layers=1)
        self.lin_dec2 = nn.Linear(32, in_shape)

    def encode(self, x):
        x = self.lin_enc1(x)
        x = self.relu_enc1(x)
        x = self.dropout_enc1(x)
        x = x.unsqueeze(1)
        x, h_n = self.gru_enc(x)
        x = x.squeeze()
        x = self.lin_enc2(x)
        return x

    def decode(self, x):
        x = self.batch_norm(x)
        x = self.lin_dec1(x)
        x = self.relu_dec1(x)
        x = self.dropout_dec1(x)
        x = x.unsqueeze(1)
        x, h_n = self.gru_dec(x)
        x = x.squeeze()
        x = self.lin_dec2(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


def train(model, error, optimizer, n_epochs, x):
    model.train()
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        output = model(x)
        loss = error(output, x)
        loss.backward()
        optimizer.step()

        if epoch % int(0.1 * n_epochs) == 0:
            print(f'epoch {epoch} \t Loss: {loss.item():.4g}')
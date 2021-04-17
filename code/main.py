from utils.dataset import ECG5000
from models.RecurrentAutoencoder import RecurrentAutoencoder
import torch
import torch.nn as nn
import copy
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader



# https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/

if __name__ == '__main__':

    dataset = ECG5000()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len, n_features = 140, 1
    batch_size = 8

    model = RecurrentAutoencoder(seq_len, n_features=n_features, embedding_dim=128, device=device, batch_size=batch_size)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # x = dataset.get_torch_tensor()
    # z = model.encoder(x)  # z.shape = [7]
    # x_prime = model.decoder(z, seq_len=10)  # x_prime.shape = [10, 3]
    #
    # z = model(x)
    #
    # print(x.shape)

    # start training
    n_epochs = 500
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction='sum').to(device) # todo article use L1Loss
    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    for epoch in range(1, n_epochs + 1):
        model = model.train()
        train_losses = []
        for i, seq_true in enumerate(dataloader):
            optimizer.zero_grad()
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        history['train'].append(train_loss)
        print(f'Epoch {epoch}: train loss {train_loss}')
    model.load_state_dict(best_model_wts)
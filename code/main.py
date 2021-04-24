from utils.dataset import ECG5000
from models.RecurrentAutoencoder import RecurrentAutoencoder
import torch
import torch.nn as nn
import copy
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler



# https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/

if __name__ == '__main__':

    dataset_normal = ECG5000(mode='normal')
    dataset_anomaly = ECG5000(mode='anomaly')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len, n_features = 140, 1
    batch_size = 512

    ################################
    validation_split = test_split = 0.15
    random_seed = 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset_normal)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    # suffling
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_indices, val_indices = train_indices[split:], train_indices[:split]

    print('train_indices: ', len(train_indices))
    print('val_indices: ', len(val_indices))
    print('test_indices: ', len(test_indices))

    # check all splits have no intersections
    assert not [value for value in train_indices if value in test_indices]
    assert not [value for value in train_indices if value in val_indices]
    assert not [value for value in val_indices if value in test_indices]
    ##############################

    model = RecurrentAutoencoder(seq_len, n_features=n_features, embedding_dim=128, device=device, batch_size=batch_size)

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset_normal, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset_normal, batch_size=batch_size, sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(dataset_normal, batch_size=batch_size, sampler=test_sampler)
    anomaly_loader = torch.utils.data.DataLoader(dataset_anomaly, batch_size=batch_size)


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
    criterion = nn.MSELoss(reduction='mean').to(device) # todo article use L1Loss
    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    for epoch in tqdm(range(1, n_epochs + 1)):
        model = model.train()

        train_losses = []
        val_losses = []
        test_losses = []
        anomaly_losses = []

        for i, seq_true in enumerate(train_loader):
            optimizer.zero_grad()
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model = model.eval()
        with torch.no_grad():

            # validation steps
            for i, seq_true in enumerate(validation_loader):
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                val_losses.append(loss.item())

            # normal_test steps
            for i, seq_true in enumerate(test_loader):
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                test_losses.append(loss.item())

            # anomaly_test steps
            for i, seq_true in enumerate(anomaly_loader):
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                anomaly_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        test_loss = np.mean(test_losses)
        anomaly_loss = np.mean(anomaly_losses)
        history['train'].append(train_loss)
        print(f'Epoch {epoch}: train loss {train_loss} {" "*6} val loss {val_loss} {" "*6} test loss {test_loss} {" "*6} anomaly loss {anomaly_loss}')

    model.load_state_dict(best_model_wts)
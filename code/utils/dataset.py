import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm
import pandas as pd
from arff2pandas import a2p
from sklearn.model_selection import train_test_split
import torch

# http://timeseriesclassification.com/description.php?Dataset=ECG5000
class ECG5000(Dataset):

    def __init__(self):
        trainset_file = '/opt/data_and_extra/ECG5000/ECG5000_TRAIN.arff'
        testset_file = '/opt/data_and_extra/ECG5000/ECG5000_TEST.arff'

        with open(trainset_file) as f:
            train = a2p.load(f)
        with open(testset_file) as f:
            test = a2p.load(f)

        df = train.append(test)

        # drop label
        df = df.iloc[:, :-1]

        train_df, val_df = train_test_split(
            df,
            test_size=0.15,
            random_state=2
        )

        self.X = df.astype(np.float32).to_numpy()

    def get_torch_tensor(self):
        return torch.from_numpy(self.X)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]).reshape(-1, 1)

    # return len of dataset
    def __len__(self):
        return self.X.shape[0]


if __name__ == '__main__':

    dataset = ECG5000()

    for x in dataset:
        print(x.shape)
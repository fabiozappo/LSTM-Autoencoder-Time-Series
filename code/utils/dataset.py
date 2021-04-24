import numpy as np
from torch.utils.data import Dataset, DataLoader
from arff2pandas import a2p
from sklearn.model_selection import train_test_split
import torch

# http://timeseriesclassification.com/description.php?Dataset=ECG5000
class ECG5000(Dataset):

    def __init__(self, mode):

        assert mode in ['normal', 'anomaly']

        trainset_file = '/opt/data_and_extra/ECG5000/ECG5000_TRAIN.arff'
        testset_file = '/opt/data_and_extra/ECG5000/ECG5000_TEST.arff'

        with open(trainset_file) as f:
            train = a2p.load(f)
        with open(testset_file) as f:
            test = a2p.load(f)

        df = train.append(test)

        # split in normal and anomaly data, then drop label
        CLASS_NORMAL = 1
        new_columns = list(df.columns)
        new_columns[-1] = 'target'
        df.columns = new_columns

        if mode == 'normal':
            df = df[df.target == str(CLASS_NORMAL)].drop(labels='target', axis=1)
        else:
            df = df[df.target != str(CLASS_NORMAL)].drop(labels='target', axis=1)

        print(df.shape)
        # train_df, val_df = train_test_split(
        #     normal_df,
        #     test_size=0.15,
        #     random_state=random_seed
        # )
        #
        # val_df, test_df = train_test_split(
        #     val_df,
        #     test_size=0.33,
        #     random_state=random_seed
        # )

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
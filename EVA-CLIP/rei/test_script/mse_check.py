import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
import numpy as np
import lmdb
from tqdm import tqdm
import msgpack
import msgpack_numpy
msgpack_numpy.patch()


# LMDBDataset
class LMDBDataset(Dataset):
    def __init__(self, lmdb_path):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.txn = self.env.begin()
        self.keys = [key.decode() for key, _ in tqdm(self.txn.cursor(), total=self.txn.stat()['entries'])]

    def __getitem__(self, index):
        key = self.keys[index]
        value = self.txn.get(key.encode())
        feats = np.array(msgpack.unpackb(value))
        feats = torch.from_numpy(feats)

        return feats

    def __len__(self):
        return len(self.keys)


# Define MSE loss for codebook supervision
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, output, target):
        loss_mse = self.loss_fn(output, target)
        return loss_mse



# Set hyperparameters

batch_size = 4096
num_epochs = 100
num_workers=16
dataset_dir = "/workspace/code/clip-tmp/cc-train/clip_cls_emb_train"



dataset = LMDBDataset(dataset_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

mse_loss = MSELoss()

# Training
for epoch in range(num_epochs):
    pbar = tqdm(dataloader, total=len(dataloader))
    sum_loss_mse = 0.0
    for i, batch in enumerate(pbar):
        # img + text
        # batch = batch.reshape(-1,1024)
        # idx = torch.randperm(batch_size*2)

        # img
        # batch = batch[:,0,:].reshape(-1,1024)
        # idx = torch.randperm(batch_size)

        # text
        # batch = batch[:,1,:].reshape(-1,1024)
        # idx = torch.randperm(batch_size)
        # batch2 = batch[idx]

        # pair
        batch1 = batch[:,0,:].reshape(-1,1024)
        batch2 = batch[:,1,:].reshape(-1,1024)


        # Loss
        loss_mse = mse_loss(batch1,batch2)
        sum_loss_mse+=loss_mse


        if (i + 1) % 10 == 0:
            sum_loss_mse /= 10.0
            pbar.set_description(f"[{epoch}/{num_epochs}] Loss: mse_{sum_loss_mse:.4f}")
            sum_loss_mse = 0.0

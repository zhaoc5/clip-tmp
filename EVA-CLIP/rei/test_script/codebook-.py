import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed
import torch.multiprocessing as mp
import time
import numpy as np
import lmdb
from tqdm import tqdm
import msgpack
import msgpack_numpy
msgpack_numpy.patch()


def h_asign(d):
    h_idx = linear_sum_assignment(d.squeeze().numpy())[1]
    return h_idx

# Define the VQ-VAE model
class VQVAE(nn.Module):
    def __init__(self, codebook_size, hidden_dim, obj_num):
        super(VQVAE, self).__init__()
        self.codebook_size = codebook_size
        self.hidden_dim = hidden_dim
        self.obj_num = obj_num

        # Codebook
        self.codebook = nn.Embedding(codebook_size, hidden_dim)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, self.obj_num*hidden_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.obj_num*hidden_dim, hidden_dim),
        )

    def forward(self, x):
        e = self.encoder(x).reshape(-1,self.obj_num,self.hidden_dim) # [batch_size,self.obj_num,1024]
        q = self.quantize(e) # [batch_size,self.obj_num,1024]
        q_in = e + (q-e).detach()
        output = self.decoder(q_in.reshape(-1,self.obj_num*self.hidden_dim)) # [batch_size,1024]

        return output, q, e

    def quantize(self, encoding):
        encoding = encoding.reshape(-1,self.hidden_dim)
        # Compute distances between encoding and codebook entries
        #distances = torch.cdist(encoding, self.codebook.unsqueeze(0))
        distances = (torch.sum(encoding**2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * torch.matmul(encoding, self.codebook.weight.t())).sqrt()
        distances = distances.reshape(-1, self.obj_num, self.codebook_size).detach().cpu() # [batch_size, self.obj_num, 8196]
        distances = distances.split(1) # batch_size * [self.obj_num, 8196]
        #indices = torch.argmin(distances, dim=2)  # Find closest codebook indices
        start = time.time()
        # indices = [
        #     linear_sum_assignment(d.squeeze().numpy())[1]
        #     for d in distances
        # ]

        pool = mp.Pool(16)
        indices = []
        for d in distances:
            p1 = pool.apply_async(h_asign,args=d)
            indices.append(p1)
        indices = [i.get() for i in indices]
        pool.close()
        pool.join()
        # task = [delayed(h_asign)(d) for d in distances]
        # indices = Parallel(n_jobs=2)(task)
        end = time.time()
        print('====>{:.4f} s'.format(end-start))

        indices = torch.from_numpy(np.concatenate(indices))
        indices = indices.to(encoding.device)
        quantized = self.codebook(indices)
        quantized = quantized.reshape(-1,self.obj_num,self.hidden_dim)
        
        return quantized

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
hidden_dim = 1024
codebook_size = 8192
obj_num = 32
learning_rate = 0.001
batch_size = 2048
num_epochs = 100
num_workers = 8
dataset_dir = "/workspace/code/clip-tmp/cc-train/clip_cls_emb_train"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQVAE(codebook_size, hidden_dim, obj_num)
model = model.to(device)

dataset = LMDBDataset(dataset_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, drop_last=True)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
mse_loss = MSELoss()

# Training
for epoch in range(num_epochs):
    pbar = tqdm(dataloader, total=len(dataloader))
    sum_loss_reconstruction = 0.0
    sum_loss_codebook = 0.0
    sum_loss_commitmentn = 0.0
    for i, batch in enumerate(pbar):
        optimizer.zero_grad()
        batch = batch.reshape(-1,1024).to(device)

        # Forward 
        output, q, e = model(batch)

        # Loss
        loss_reconstruction = mse_loss(batch, output)
        loss_codebook = mse_loss(q.detach(), e)
        loss_commitment = mse_loss(q, e.detach())
        total_loss = loss_reconstruction + loss_codebook + 0.25*loss_commitment

        sum_loss_reconstruction+=loss_reconstruction
        sum_loss_codebook+=loss_codebook
        sum_loss_commitmentn+=loss_commitment

        # Backward and optimization
        total_loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            sum_loss_reconstruction /= 10.0
            sum_loss_codebook /= 10.0
            sum_loss_commitmentn /= 10.0
            pbar.set_description(f"[{epoch}/{num_epochs}] Loss: recon_{sum_loss_reconstruction:.4f} codebook_{sum_loss_codebook:.4f} commit_{sum_loss_commitmentn:.4f} ")
            sum_loss_reconstruction = 0.0
            sum_loss_codebook = 0.0
            sum_loss_commitmentn = 0.0
    torch.save(model.state_dict(), f"/workspace/code/clip-tmp/EVA-CLIP/rei/logs_codebook/{epoch}.pth")
    print(f"Model weights saved at epoch {epoch}.")

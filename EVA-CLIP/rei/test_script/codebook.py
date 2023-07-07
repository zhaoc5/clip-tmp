import argparse

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

msgpack_numpy.patch()


class HungarianVQVAE(nn.Module):
    def __init__(self, codebook_size, hidden_dim, code_dim, num_codes):
        super(HungarianVQVAE, self).__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.num_codes = num_codes

        self.codebook = nn.Embedding(codebook_size, code_dim)
        self.encoder = nn.Linear(hidden_dim, num_codes * code_dim)
        self.decoder = nn.Linear(num_codes * code_dim, hidden_dim)

    def forward(self, x):
        bs = x.shape[0]
        e = self.encoder(x)
        e = e.reshape(bs, self.num_codes, self.code_dim)
        q, indices = self.quantize(e)
        q_ = e + (q - e).detach()
        q_ = q_.reshape(bs, self.num_codes * self.code_dim)
        r = self.decoder(q_)
        return r, q, e, indices

    def quantize(self, e):
        e = e.reshape(-1, self.code_dim)

        distances = (torch.sum(e ** 2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight ** 2, dim=1)
            - 2 * torch.matmul(e, self.codebook.weight.t())).sqrt()
        distances = distances.reshape(-1, self.num_codes, self.codebook_size)
        distances = distances.split(1)

        indices = [
            linear_sum_assignment(d[0].detach().cpu().numpy())[1]
            for d in distances
        ]
        indices = torch.from_numpy(np.concatenate(indices))
        indices = indices.to(e.device)

        q = self.codebook(indices)
        q = q.reshape(-1, self.num_codes, self.code_dim)
        return q, indices


class LMDBDataset(Dataset):
    def __init__(self, lmdb_path):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.txn = self.env.begin()
        self.keys = [key.decode() for key, _ in tqdm(self.txn.cursor())]

    def __getitem__(self, index):
        key = self.keys[index]
        value = self.txn.get(key.encode())
        e = np.array(msgpack.unpackb(value))
        return torch.from_numpy(e)

    def __len__(self):
        return len(self.keys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--code-dim", type=int, default=1024)
    parser.add_argument("--codebook-size", type=int, default=8192)
    parser.add_argument("--num-codes", type=int, default=32)
    parser.add_argument("--commitment", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--log-frequency", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    model = HungarianVQVAE(codebook_size=args.codebook_size,
                           hidden_dim=args.hidden_dim,
                           code_dim=args.code_dim,
                           num_codes=args.num_codes)
    model = model.to("cuda")

    dataset = LMDBDataset(args.data_dir)

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  betas=(0.9, 0.999),
                                  weight_decay=1e-2,
                                  eps=1e-8)

    max_step = len(dataset) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=max_step,
                                                           eta_min=args.min_lr)

    log_dir = "{}/imgmlp_bs{}_cs{}_cd{}_nc{}".format(args.log_dir,
                                                  args.batch_size,
                                                  args.codebook_size,
                                                  args.code_dim,
                                                  args.num_codes)
    writer = SummaryWriter(log_dir=log_dir)

    current_epoch = -1
    if args.resume:
        checkpoint = torch.load(f"{log_dir}/latest.pth",
                                map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint.pop("model"))
        optimizer.load_state_dict(checkpoint.pop("optimizer"))
        scheduler.load_state_dict(checkpoint.pop("scheduler"))
        current_epoch = checkpoint.pop("current_epoch")
        del checkpoint

    for epoch in range(current_epoch + 1, args.epochs):
        for i, x in tqdm(enumerate(dataloader), total=len(dataloader)):
            x = x.reshape(-1, args.hidden_dim).to("cuda")

            r, q, e, indices = model(x)
            loss = F.mse_loss(x, r)
            loss_code = F.mse_loss(q.detach(), e)
            loss_commitment = F.mse_loss(q, e.detach())
            total_loss = loss + loss_code + loss_commitment * args.commitment

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            if i % args.log_frequency == 0:
                j = epoch * len(dataloader) + i
                writer.add_scalar("loss_r", loss.detach().cpu().numpy(), j)
                writer.add_scalar("loss_c", loss_code.detach().cpu().numpy(), j)
                writer.add_scalar("indices", len(indices.unique()), j)

        checkpoint = {"model": model.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(),
                      "current_epoch": epoch}
        torch.save(checkpoint, f"{log_dir}/latest.pth")
        torch.save(model.state_dict(), f"{log_dir}/epoch_{epoch}")
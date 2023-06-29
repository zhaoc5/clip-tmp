import torch
import torch.nn.functional as F
import numpy as np
import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
from torch.utils.tensorboard import SummaryWriter

# tensorboard --logdir=/workspace/code/clip-tmp/_debug --port=6006 --host=129.127.104.58

# img_cls_embeddings = np.load("/data/cc-train/clip_emb/img/rank0_img_emb_00000001.npy")[:500][:,0,:]
# img_cls_embeddings = torch.from_numpy(img_cls_embeddings)
# print("Image Load Sucess")
# text_cls_embeddings = np.load("/data/cc-train/clip_emb/text/rank0_text_emb_00000001.npy")[:500]
# text_cls_embeddings = torch.from_numpy(text_cls_embeddings)
# print("Text Load Sucess")

img_cls_embeddings = []
text_cls_embeddings = []
flag = 0
lmdb_path = '/workspace/code/clip-tmp/_debug/clip_cls_emb_val'
env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
txn = env.begin()
cursor = txn.cursor()
for key, value in cursor:
    data = msgpack.unpackb(value)
    img_cls_embeddings.append(np.reshape(data[0],(1,-1)))
    text_cls_embeddings.append(np.reshape(data[1],(1,-1)))
    flag +=1
    # if flag >=5000:
    #     break
cursor.close()
txn.abort() 
env.close()
print(f"Data Load Sucess ({flag})")

img_cls_embeddings = np.concatenate(img_cls_embeddings)
img_cls_embeddings = torch.from_numpy(img_cls_embeddings)
img_cls_embeddings = F.normalize(img_cls_embeddings, dim=-1)

text_cls_embeddings = np.concatenate(text_cls_embeddings)
text_cls_embeddings = torch.from_numpy(text_cls_embeddings)
text_cls_embeddings = F.normalize(text_cls_embeddings, dim=-1)

# Combine the image and text embeddings into a single embedding tensor
combined_embeddings = torch.cat([img_cls_embeddings, text_cls_embeddings], dim=0)
# Create labels for the embeddings indicating their type (image or text)
labels = ['Image'] * img_cls_embeddings.shape[0] + ['Text'] * text_cls_embeddings.shape[0]

# Initialize a TensorBoard writer
log_dir = f"/workspace/code/clip-tmp/_debug/tb_debug/{flag}_l2"
writer = SummaryWriter(log_dir=log_dir)

writer.add_embedding(combined_embeddings, metadata=labels, tag="Image and Text CLS Token Embeddings")
print("Visualization Ready")

# Save the writer's content and close it
writer.close()



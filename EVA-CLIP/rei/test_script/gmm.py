import torch
import torch.nn.functional as F
import numpy as np
import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


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
    # if flag >=50:
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
print(f"Data L2-norm Sucess ({flag})")


image_gmm = GaussianMixture(n_components=2).fit(img_cls_embeddings)
# text_gmm = GaussianMixture(n_components=1).fit(text_cls_embeddings)
# labels = img_gmm.predict(img_cls_embeddings)

m = image_gmm.means_
# np.save('/workspace/code/clip-tmp/_debug/_means', m)
# print(f"means_ Save Sucess, shape:{str(m.shape)} ({flag})")

c = image_gmm.covariances_
# np.save('/workspace/code/clip-tmp/_debug/_covariances', c)
# print(f"covariances_ Save Sucess, shape:{str(c.shape)} ({flag})")


# plt.scatter(X[:, 0], X[:, 1], c=labels, s=20, cmap='viridis', alpha=0.1)
# plt.show()
# plt.savefig(f'/workspace/code/clip-tmp/_debug/gmm.jpg')
# print(f"Image Vis Save Sucess ({flag})")






# n_samples = 1000  # Number of synthetic samples to generate
# image_samples = image_gmm.sample(n_samples)[0]
# text_samples = text_gmm.sample(n_samples)[0]


# plt.figure(figsize=(10, 5))

# # Plot image embeddings
# plt.subplot(1, 2, 1)
# plt.scatter(img_cls_embeddings[:, 0], img_cls_embeddings[:, 1], label='Original')
# plt.scatter(image_samples[:, 0], image_samples[:, 1], label='Synthetic')
# plt.title('Image Embeddings')
# plt.legend()

# # Plot text embeddings
# plt.subplot(1, 2, 2)
# plt.scatter(text_cls_embeddings[:, 0], text_cls_embeddings[:, 1], label='Original')
# plt.scatter(text_samples[:, 0], text_samples[:, 1], label='Synthetic')
# plt.title('Text Embeddings')
# plt.legend()

# plt.tight_layout()
# plt.show()


# means=[]
# covariances=[]
# for i in range(m.shape[0]):
#     means.append(m[i])
#     covariances.append(c[i])

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Example mean and covariance values for Group 1
mean1 = m[0] # Mean vector for Group 1
covariance1 = c[0]  # Covariance matrix for Group 1

# Example mean and covariance values for Group 2
mean2 = m[1]  # Mean vector for Group 2
covariance2 = c[1]  # Covariance matrix for Group 2

# Generate points from a multivariate Gaussian distribution for Group 1
x = np.linspace(-1, 1, 1000)
pdf1 = multivariate_normal.pdf(x, mean=mean1[0], cov=covariance1[0, 0])

# Generate points from a multivariate Gaussian distribution for Group 2
pdf2 = multivariate_normal.pdf(x, mean=mean2[0], cov=covariance2[0, 0])

# Plot the Gaussian distribution probability density curves
plt.plot(x, pdf1, label='Group 1')
plt.plot(x, pdf2, label='Group 2')

# Add legend, labels, and title
plt.legend()
plt.xlabel('X')
plt.ylabel('Probability Density')
plt.title('Marginal Distributions')

# Display the plot
plt.show()




plt.savefig(f'/workspace/code/clip-tmp/_debug/gmm_new.jpg')
print(f"Vis Save Sucess")
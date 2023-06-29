import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Example mean and covariance values for Group 1
mean1 = np.array([1, 2])  # Mean vector for Group 1
covariance1 = np.array([[1, 0.5], [0.5, 2]])  # Covariance matrix for Group 1

# Example mean and covariance values for Group 2
mean2 = np.array([-2, 3])  # Mean vector for Group 2
covariance2 = np.array([[2, 0.3], [0.3, 1]])  # Covariance matrix for Group 2

# Generate points from a multivariate Gaussian distribution for Group 1
x1, y1 = np.mgrid[-10:10:.1, -10:10:.1]
pos1 = np.dstack((x1, y1))
rv1 = multivariate_normal(mean1, covariance1)
z1 = rv1.pdf(pos1)

# Generate points from a multivariate Gaussian distribution for Group 2
x2, y2 = np.mgrid[-10:10:.1, -10:10:.1]
pos2 = np.dstack((x2, y2))
rv2 = multivariate_normal(mean2, covariance2)
z2 = rv2.pdf(pos2)

# Plot the Gaussian distribution probability density curves
plt.plot(x1[:, 0], z1[:, 0], label='Group 1')
plt.plot(x2[:, 0], z2[:, 0], label='Group 2')

# Add legend, labels, and title
plt.legend()
plt.xlabel('X')
plt.ylabel('Probability Density')
plt.title('Gaussian Distribution')

# Display the plot
plt.show()




plt.savefig(f'/workspace/code/clip-tmp/_debug/gmm_new.jpg')
print(f"Vis Save Sucess")

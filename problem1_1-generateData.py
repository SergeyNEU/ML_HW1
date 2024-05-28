# By Sergey Petrushkevich
# Problem 1.1 Data Generation

import numpy as np
from scipy.stats import multivariate_normal

# I used docs from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html

# -------- GIVEN PARAMETERS --------

# Given parameters for the multivariate Gaussian distributions
mean_vector_class0 = np.array([-1, 1, -1, 1])
covariance_matrix_class0 = np.array(
    [[2, -0.5, 0.3, 0], [-0.5, 1, -0.5, 0], [0.3, -0.5, 1, 0], [0, 0, 0, 2]]
)

mean_vector_class1 = np.array([1, 1, 1, 1])
covariance_matrix_class1 = np.array(
    [[1, 0.3, -0.2, 0], [0.3, 2, 0.3, 0], [-0.2, 0.3, 1, 0], [0, 0, 0, 3]]
)

# Given class priors
prior_class0 = 0.7  # The probability of class 0
prior_class1 = 0.3  # The probability of class 1

# Number of samples to generate
number_of_samples = 10000

# -------- DATA GENERATION --------

# Generate samples for each class based on prior probabilities
number_of_samples_class0 = int(prior_class0 * number_of_samples)
number_of_samples_class1 = number_of_samples - number_of_samples_class0

# Generating samples for class 0 and 1 using their mean vector and covariance matrices
samples_class0 = multivariate_normal.rvs(
    mean=mean_vector_class0, cov=covariance_matrix_class0, size=number_of_samples_class0
)
samples_class1 = multivariate_normal.rvs(
    mean=mean_vector_class1, cov=covariance_matrix_class1, size=number_of_samples_class1
)

# Combine samples from both classes into a single array
samples = np.vstack((samples_class0, samples_class1))

# Create labels for the samples: 0 for class 0 and 1 for class 1
labels = np.hstack(
    (np.zeros(number_of_samples_class0), np.ones(number_of_samples_class1))
)

# Shuffle the samples and labels together
shuffled_indices = np.random.permutation(number_of_samples)
samples = samples[shuffled_indices]
labels = labels[shuffled_indices]

# -------- SAVE THE DATA --------

# Save the data and labels
np.savez("data_samples.npz", samples=samples, labels=labels)

# By Sergey Petrushkevich
# Problem 1.2 Part A and Part B

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------- PART A --------

mean_vector_class1 = np.array([1, 0, 0])
covariance_matrix_class1 = np.array([[1, 0.5, 0], [0.5, 1, 0.5], [0, 0.5, 1]])

mean_vector_class2 = np.array([-1, 0, 0])
covariance_matrix_class2 = np.array([[1, -0.5, 0], [-0.5, 1, -0.5], [0, -0.5, 1]])

mean_vector_class3a = np.array([0, 1, 1])
covariance_matrix_class3a = np.array([[1, 0.25, 0.1], [0.25, 1, 0.25], [0.1, 0.25, 1]])

mean_vector_class3b = np.array([0, -1, -1])
covariance_matrix_class3b = np.array(
    [[1, -0.25, -0.1], [-0.25, 1, -0.25], [-0.1, -0.25, 1]]
)

# Class priors
prior_class1 = 0.3
prior_class2 = 0.3
prior_class3 = 0.4

# Generating samples
num_samples = 10000
samples_class1 = np.random.multivariate_normal(
    mean_vector_class1, covariance_matrix_class1, int(num_samples * prior_class1)
)
samples_class2 = np.random.multivariate_normal(
    mean_vector_class2, covariance_matrix_class2, int(num_samples * prior_class2)
)
samples_class3a = np.random.multivariate_normal(
    mean_vector_class3a, covariance_matrix_class3a, int(num_samples * prior_class3 / 2)
)
samples_class3b = np.random.multivariate_normal(
    mean_vector_class3b, covariance_matrix_class3b, int(num_samples * prior_class3 / 2)
)
samples_class3 = np.vstack((samples_class3a, samples_class3b))

# Concatenating all samples
samples = np.vstack((samples_class1, samples_class2, samples_class3))
labels = np.hstack(
    (
        np.ones(samples_class1.shape[0]),
        2 * np.ones(samples_class2.shape[0]),
        3 * np.ones(samples_class3.shape[0]),
    )
)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    samples_class1[:, 0],
    samples_class1[:, 1],
    samples_class1[:, 2],
    marker="o",
    color="red",
    label="Class 1",
)
ax.scatter(
    samples_class2[:, 0],
    samples_class2[:, 1],
    samples_class2[:, 2],
    marker="^",
    color="green",
    label="Class 2",
)
ax.scatter(
    samples_class3[:, 0],
    samples_class3[:, 1],
    samples_class3[:, 2],
    marker="s",
    color="blue",
    label="Class 3",
)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3")
ax.legend()
plt.title("3D Scatter Plot of Generated Data")
plt.savefig("3d_scatter_plot.png", facecolor="white", edgecolor="white")
plt.show()


# Minimum Probability of Error (MAP) Classifier
def classify_sample(x):
    likelihood_class1 = multivariate_normal.pdf(
        x, mean=mean_vector_class1, cov=covariance_matrix_class1
    )
    likelihood_class2 = multivariate_normal.pdf(
        x, mean=mean_vector_class2, cov=covariance_matrix_class2
    )
    likelihood_class3a = multivariate_normal.pdf(
        x, mean=mean_vector_class3a, cov=covariance_matrix_class3a
    )
    likelihood_class3b = multivariate_normal.pdf(
        x, mean=mean_vector_class3b, cov=covariance_matrix_class3b
    )
    likelihood_class3 = 0.5 * likelihood_class3a + 0.5 * likelihood_class3b

    posterior_class1 = prior_class1 * likelihood_class1
    posterior_class2 = prior_class2 * likelihood_class2
    posterior_class3 = prior_class3 * likelihood_class3

    return np.argmax([posterior_class1, posterior_class2, posterior_class3]) + 1


predicted_labels = np.array([classify_sample(x) for x in samples])

confusion_matrix = np.zeros((3, 3))
for i in range(len(labels)):
    true_label = int(labels[i])
    predicted_label = int(predicted_labels[i])
    confusion_matrix[predicted_label - 1, true_label - 1] += 1

confusion_matrix /= num_samples

print("--------------------Part A--------------------")
print("Confusion Matrix:")
print(confusion_matrix)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
correctly_classified = labels == predicted_labels
incorrectly_classified = ~correctly_classified
ax.scatter(
    samples[correctly_classified, 0],
    samples[correctly_classified, 1],
    samples[correctly_classified, 2],
    marker="o",
    color="green",
    label="Correctly Classified",
)
ax.scatter(
    samples[incorrectly_classified, 0],
    samples[incorrectly_classified, 1],
    samples[incorrectly_classified, 2],
    marker="x",
    color="red",
    label="Incorrectly Classified",
)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3")
ax.legend()
plt.title("3D Scatter Plot of Classification Results (Part A)")
plt.savefig(
    "3d_classification_results_part_a.png", facecolor="white", edgecolor="white"
)
plt.show()

# -------- PART B --------


# ERM classification with loss matrices
def classify_sample_erm(x, loss_matrix):
    likelihood_class1 = multivariate_normal.pdf(
        x, mean=mean_vector_class1, cov=covariance_matrix_class1
    )
    likelihood_class2 = multivariate_normal.pdf(
        x, mean=mean_vector_class2, cov=covariance_matrix_class2
    )
    likelihood_class3a = multivariate_normal.pdf(
        x, mean=mean_vector_class3a, cov=covariance_matrix_class3a
    )
    likelihood_class3b = multivariate_normal.pdf(
        x, mean=mean_vector_class3b, cov=covariance_matrix_class3b
    )
    likelihood_class3 = 0.5 * likelihood_class3a + 0.5 * likelihood_class3b

    posterior_class1 = prior_class1 * likelihood_class1
    posterior_class2 = prior_class2 * likelihood_class2
    posterior_class3 = prior_class3 * likelihood_class3

    risks = np.array(
        [
            loss_matrix[0, 0] * posterior_class1
            + loss_matrix[0, 1] * posterior_class2
            + loss_matrix[0, 2] * posterior_class3,
            loss_matrix[1, 0] * posterior_class1
            + loss_matrix[1, 1] * posterior_class2
            + loss_matrix[1, 2] * posterior_class3,
            loss_matrix[2, 0] * posterior_class1
            + loss_matrix[2, 1] * posterior_class2
            + loss_matrix[2, 2] * posterior_class3,
        ]
    )

    return np.argmin(risks) + 1


# Define the loss matrices
loss_matrix_10 = np.array([[0, 1, 10], [1, 0, 10], [1, 1, 0]])
loss_matrix_100 = np.array([[0, 1, 100], [1, 0, 100], [1, 1, 0]])

# Classify using the loss matrix 10
predicted_labels_10 = np.array(
    [classify_sample_erm(x, loss_matrix_10) for x in samples]
)

# Confusion matrix calculation for loss matrix 10
confusion_matrix_10 = np.zeros((3, 3))
for true_label, predicted_label in zip(labels, predicted_labels_10):
    confusion_matrix_10[int(predicted_label) - 1, int(true_label) - 1] += 1

confusion_matrix_10 /= num_samples

# Display the confusion matrix for loss matrix 10
print("Confusion Matrix (Loss Matrix 10):")
print(confusion_matrix_10)

# Classify using the loss matrix 100
predicted_labels_100 = np.array(
    [classify_sample_erm(x, loss_matrix_100) for x in samples]
)

# Confusion matrix calculation for loss matrix 100
confusion_matrix_100 = np.zeros((3, 3))
for true_label, predicted_label in zip(labels, predicted_labels_100):
    confusion_matrix_100[int(predicted_label) - 1, int(true_label) - 1] += 1

confusion_matrix_100 /= num_samples

# Display the confusion matrix for loss matrix 100
print("Confusion Matrix (Loss Matrix 100):")
print(confusion_matrix_100)

# Calculate minimum expected risk for both loss matrices
min_expected_risk_10 = np.sum(confusion_matrix_10 * loss_matrix_10)
min_expected_risk_100 = np.sum(confusion_matrix_100 * loss_matrix_100)

print(f"Minimum Expected Risk (Loss Matrix 10): {min_expected_risk_10}")
print(f"Minimum Expected Risk (Loss Matrix 100): {min_expected_risk_100}")

# Plot the classification results for Loss Matrix 10
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
correctly_classified_10 = labels == predicted_labels_10
incorrectly_classified_10 = ~correctly_classified_10
ax.scatter(
    samples[correctly_classified_10, 0],
    samples[correctly_classified_10, 1],
    samples[correctly_classified_10, 2],
    marker="o",
    color="green",
    label="Correctly Classified",
)
ax.scatter(
    samples[incorrectly_classified_10, 0],
    samples[incorrectly_classified_10, 1],
    samples[incorrectly_classified_10, 2],
    marker="x",
    color="red",
    label="Incorrectly Classified",
)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3")
ax.legend()
plt.title("3D Scatter Plot of Classification Results (Loss Matrix 10)")
plt.savefig(
    "3d_classification_results_loss_matrix_10.png", facecolor="white", edgecolor="white"
)
plt.show()

# Plot the classification results for Loss Matrix 100
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
correctly_classified_100 = labels == predicted_labels_100
incorrectly_classified_100 = ~correctly_classified_100
ax.scatter(
    samples[correctly_classified_100, 0],
    samples[correctly_classified_100, 1],
    samples[correctly_classified_100, 2],
    marker="o",
    color="green",
    label="Correctly Classified",
)
ax.scatter(
    samples[incorrectly_classified_100, 0],
    samples[incorrectly_classified_100, 1],
    samples[incorrectly_classified_100, 2],
    marker="x",
    color="red",
    label="Incorrectly Classified",
)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3")
ax.legend()
plt.title("3D Scatter Plot of Classification Results (Loss Matrix 100)")
plt.savefig(
    "3d_classification_results_loss_matrix_100.png",
    facecolor="white",
    edgecolor="white",
)
plt.show()

# By Sergey Petrushkevich
# Problem 1.1 Part A and B

import numpy as np
from scipy.stats import multivariate_normal

# Using directions from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html
import matplotlib.pyplot as plt

# Part 1: Specify the minimum expected risk classification rule in the form of a likelihood-ratio test

# -------- GIVEN PARAMETERS --------

mean_vector_class0 = np.array([-1, 1, -1, 1])
covariance_matrix_class0 = np.array(
    [[2, -0.5, 0.3, 0], [-0.5, 1, -0.5, 0], [0.3, -0.5, 1, 0], [0, 0, 0, 2]]
)

mean_vector_class1 = np.array([1, 1, 1, 1])
covariance_matrix_class1 = np.array(
    [[1, 0.3, -0.2, 0], [0.3, 2, 0.3, 0], [-0.2, 0.3, 1, 0], [0, 0, 0, 3]]
)

prior_class0 = 0.7
prior_class1 = 0.3

# -------- MINIMUM EXPECTED RISK CLASSIFICATION RULE --------

loss_true_negative = 0  # Loss when the decision is 0 and the true class is 0
loss_false_negative = 1  # Loss when the decision is 0 and the true class is 1
loss_false_positive = 1  # Loss when the decision is 1 and the true class is 0
loss_true_positive = 0  # Loss when the decision is 1 and the true class is 1

# Calculate the threshold gamma, I use this later
theoretical_threshold_gamma = (
    prior_class0 * (loss_false_positive - loss_true_negative)
) / (prior_class1 * (loss_false_negative - loss_true_positive))

# Part 2: Implement the classifier and apply it on the ten thousand samples, vary threshold gamma and plot ROC curve

# -------- LOAD THE DATA --------
data = np.load("data_samples.npz")
samples = data["samples"]
true_class_labels = data["labels"]

# -------- CLASSIFICATION USING MINIMUM EXPECTED RISK CLASSIFIER --------

likelihood_class0 = multivariate_normal.pdf(
    samples, mean=mean_vector_class0, cov=covariance_matrix_class0
)
likelihood_class1 = multivariate_normal.pdf(
    samples, mean=mean_vector_class1, cov=covariance_matrix_class1
)

# I use this likelihood ratio to make decisions
likelihood_ratio = likelihood_class1 / likelihood_class0

# -------- VARY THE THRESHOLD GAMMA --------
threshold_values = np.linspace(0, 100, 1000)  # Threshold gamma from 0 to 100
true_positive_rate = []  # True Positive Rate
false_positive_rate = []  # False Positive Rate
probability_of_error = []  # Probability of Error

for gamma in threshold_values:
    # Making decisions based on the likelihood ratio test
    decisions = (likelihood_ratio > gamma).astype(int)

    # Calculate True Positives, False Positives, True Negatives, and False Negatives
    true_positives = np.sum((decisions == 1) & (true_class_labels == 1))
    false_positives = np.sum((decisions == 1) & (true_class_labels == 0))
    true_negatives = np.sum((decisions == 0) & (true_class_labels == 0))
    false_negatives = np.sum((decisions == 0) & (true_class_labels == 1))

    # Calculate True Positive Rate and False Positive Rate
    true_positive_rate_value = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    false_positive_rate_value = (
        false_positives / (false_positives + true_negatives)
        if (false_positives + true_negatives) > 0
        else 0
    )

    true_positive_rate.append(true_positive_rate_value)
    false_positive_rate.append(false_positive_rate_value)

    # Calculate Probability of Error
    probability_error_value = (
        false_positive_rate_value * prior_class0
        + (1 - true_positive_rate_value) * prior_class1
    )
    probability_of_error.append(probability_error_value)

# Part 3: Determine the threshold value that achieves minimum probability of error and plot ROC curve

# -------- DETERMINE THE OPTIMAL THRESHOLD AND PLOT THE ROC CURVE --------
minimum_probability_of_error = min(probability_of_error)
minimum_probability_of_error_index = probability_of_error.index(
    minimum_probability_of_error
)
optimal_threshold_gamma = threshold_values[minimum_probability_of_error_index]
optimal_true_positive_rate = true_positive_rate[minimum_probability_of_error_index]
optimal_false_positive_rate = false_positive_rate[minimum_probability_of_error_index]

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(
    false_positive_rate,
    true_positive_rate,
    label="Receiver Operating Characteristic (ROC) Curve",
)
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.scatter(
    [optimal_false_positive_rate],
    [optimal_true_positive_rate],
    color="red",
    zorder=99,
    label=f"Minimum Probability of Error at Gamma={optimal_threshold_gamma:.2f}",
)
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("PART A - ROC Curve of Minimum Expected Risk Classifier")
plt.legend(loc="best")
plt.grid(True)
plt.savefig("roc_curve_a.png", facecolor="white", edgecolor="white")
plt.show()

# Output the results
print(f"Theoretical Optimal Gamma: {theoretical_threshold_gamma:.2f}")
print("--------------------Part A--------------------")
print(f"Optimal Threshold Gamma: {optimal_threshold_gamma}")
print(f"Minimum Probability of Error: {minimum_probability_of_error}")
print(f"Optimal True Positive Rate (TPR): {optimal_true_positive_rate}")
print(f"Optimal False Positive Rate (FPR): {optimal_false_positive_rate}")

# -------- Part B: ERM Classification using Incorrect Knowledge of Data Distribution --------

# Redefine the covariance matrices as diagonal
covariance_matrix_class0_naive = np.diag(np.diag(covariance_matrix_class0))
covariance_matrix_class1_naive = np.diag(np.diag(covariance_matrix_class1))

# Calculate the likelihoods for each class with naive assumption
likelihood_class0_naive = multivariate_normal.pdf(
    samples, mean=mean_vector_class0, cov=covariance_matrix_class0_naive
)
likelihood_class1_naive = multivariate_normal.pdf(
    samples, mean=mean_vector_class1, cov=covariance_matrix_class1_naive
)

# Calculate the likelihood ratio
likelihood_ratio_naive = likelihood_class1_naive / likelihood_class0_naive

# -------- VARY THE THRESHOLD GAMMA FOR NAIVE BAYESIAN CLASSIFIER --------
true_positive_rate_naive = []  # True Positive Rate
false_positive_rate_naive = []  # False Positive Rate
probability_of_error_naive = []  # Probability of Error

for gamma in threshold_values:
    # Make decisions based on the likelihood ratio test
    decisions_naive = (likelihood_ratio_naive > gamma).astype(int)

    # Calculate True Positives, False Positives, True Negatives, and False Negatives
    true_positives_naive = np.sum((decisions_naive == 1) & (true_class_labels == 1))
    false_positives_naive = np.sum((decisions_naive == 1) & (true_class_labels == 0))
    true_negatives_naive = np.sum((decisions_naive == 0) & (true_class_labels == 0))
    false_negatives_naive = np.sum((decisions_naive == 0) & (true_class_labels == 1))

    # Calculate True Positive Rate and False Positive Rate
    true_positive_rate_value_naive = (
        true_positives_naive / (true_positives_naive + false_negatives_naive)
        if (true_positives_naive + false_negatives_naive) > 0
        else 0
    )
    false_positive_rate_value_naive = (
        false_positives_naive / (false_positives_naive + true_negatives_naive)
        if (false_positives_naive + true_negatives_naive) > 0
        else 0
    )

    true_positive_rate_naive.append(true_positive_rate_value_naive)
    false_positive_rate_naive.append(false_positive_rate_value_naive)

    # Calculate Probability of Error
    probability_error_value_naive = (
        false_positive_rate_value_naive * prior_class0
        + (1 - true_positive_rate_value_naive) * prior_class1
    )
    probability_of_error_naive.append(probability_error_value_naive)

# -------- DETERMINE THE OPTIMAL THRESHOLD AND PLOT THE ROC CURVE FOR NAIVE BAYESIAN CLASSIFIER --------
minimum_probability_of_error_naive = min(probability_of_error_naive)
minimum_probability_of_error_index_naive = probability_of_error_naive.index(
    minimum_probability_of_error_naive
)
optimal_threshold_gamma_naive = threshold_values[
    minimum_probability_of_error_index_naive
]
optimal_true_positive_rate_naive = true_positive_rate_naive[
    minimum_probability_of_error_index_naive
]
optimal_false_positive_rate_naive = false_positive_rate_naive[
    minimum_probability_of_error_index_naive
]

plt.figure(figsize=(8, 6))
plt.plot(
    false_positive_rate_naive,
    true_positive_rate_naive,
    label="Naive Bayesian ROC Curve",
)
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.scatter(
    [optimal_false_positive_rate_naive],
    [optimal_true_positive_rate_naive],
    color="red",
    zorder=99,
    label=f"Minimum Probability of Error at Gamma={optimal_threshold_gamma_naive:.2f}",
)
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("PART B - ROC Curve of Naive Bayesian Classifier")
plt.legend(loc="best")
plt.grid(True)
plt.savefig("roc_curve_b.png", facecolor="white", edgecolor="white")
plt.show()

print("--------------------Part B--------------------")
print(f"Optimal Threshold Gamma (Naive): {optimal_threshold_gamma_naive}")
print(f"Minimum Probability of Error (Naive): {minimum_probability_of_error_naive}")
print(f"Optimal True Positive Rate (TPR) (Naive): {optimal_true_positive_rate_naive}")
print(f"Optimal False Positive Rate (FPR) (Naive): {optimal_false_positive_rate_naive}")

# By Sergey Petrushkevich
# Problem 1.3

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

wine_data_red = pd.read_csv("wine+quality/winequality-red.csv", sep=";")
wine_data_white = pd.read_csv("wine+quality/winequality-white.csv", sep=";")
wine_data = pd.concat([wine_data_red, wine_data_white], ignore_index=True)

# Separating features and labels
wine_features = wine_data.iloc[:, :-1].values
wine_labels = wine_data.iloc[:, -1].values

# Split the data into training and test sets
wine_features_train, wine_features_test, wine_labels_train, wine_labels_test = (
    train_test_split(wine_features, wine_labels, test_size=0.3, random_state=42)
)

wine_classes = np.unique(wine_labels_train)
class_priors_wine = {}
class_means_wine = {}
class_covariances_wine = {}
regularization_lambda = 0.01

for class_label in wine_classes:
    class_data = wine_features_train[wine_labels_train == class_label]
    class_priors_wine[class_label] = class_data.shape[0] / wine_features_train.shape[0]
    class_means_wine[class_label] = np.mean(class_data, axis=0)
    class_covariances_wine[class_label] = np.cov(
        class_data, rowvar=False
    ) + regularization_lambda * np.eye(class_data.shape[1])


# Classifier function for wine data
def classify_wine(samples):
    posteriors = []
    for class_label in wine_classes:
        likelihood = multivariate_normal.pdf(
            samples,
            mean=class_means_wine[class_label],
            cov=class_covariances_wine[class_label],
        )
        posterior = likelihood * class_priors_wine[class_label]
        posteriors.append(posterior)
    return wine_classes[np.argmax(posteriors, axis=0)]


# Classify the test samples and calculate the error rate for wine data
wine_predictions = classify_wine(wine_features_test)
wine_error_rate = np.mean(wine_predictions != wine_labels_test)

wine_confusion_matrix = pd.crosstab(
    wine_labels_test,
    wine_predictions,
    rownames=["True"],
    colnames=["Predicted"],
    margins=True,
)

print("Error rate (Wine Quality):", wine_error_rate)
print("Confusion Matrix (Wine Quality):")
print(wine_confusion_matrix)

wine_df = pd.DataFrame(wine_features_train, columns=wine_data.columns[:-1])
wine_df["Quality"] = wine_labels_train
sns.pairplot(
    wine_df, hue="Quality", vars=wine_data.columns[:5], plot_kws={"alpha": 0.5}
)
plt.suptitle("Wine Quality Dataset - Pair Plot of First Five Features", y=1.02)
plt.tight_layout()
plt.savefig("wine_quality_pairplot.png")
plt.show()

# Load the Human Activity Recognition dataset
har_train_features = np.loadtxt(
    "human+activity+recognition+using+smartphones/UCI HAR Dataset/train/X_train.txt"
)
har_train_labels = np.loadtxt(
    "human+activity+recognition+using+smartphones/UCI HAR Dataset/train/y_train.txt"
).astype(int)
har_test_features = np.loadtxt(
    "human+activity+recognition+using+smartphones/UCI HAR Dataset/test/X_test.txt"
)
har_test_labels = np.loadtxt(
    "human+activity+recognition+using+smartphones/UCI HAR Dataset/test/y_test.txt"
).astype(int)

# Calculate class priors, means, and covariance matrices for HAR data
har_classes = np.unique(har_train_labels)
class_priors_har = {}
class_means_har = {}
class_covariances_har = {}

for class_label in har_classes:
    class_data = har_train_features[har_train_labels == class_label]
    class_priors_har[class_label] = class_data.shape[0] / har_train_features.shape[0]
    class_means_har[class_label] = np.mean(class_data, axis=0)
    class_covariances_har[class_label] = np.cov(
        class_data, rowvar=False
    ) + regularization_lambda * np.eye(class_data.shape[1])


# Classifier function for HAR data
def classify_har(samples):
    posteriors = []
    for class_label in har_classes:
        likelihood = multivariate_normal.pdf(
            samples,
            mean=class_means_har[class_label],
            cov=class_covariances_har[class_label],
        )
        posterior = likelihood * class_priors_har[class_label]
        posteriors.append(posterior)
    return har_classes[np.argmax(posteriors, axis=0)]


har_predictions = classify_har(har_test_features)
har_error_rate = np.mean(har_predictions != har_test_labels)

har_confusion_matrix = pd.crosstab(
    har_test_labels,
    har_predictions,
    rownames=["True"],
    colnames=["Predicted"],
    margins=True,
)

print("Error rate (Human Activity Recognition):", har_error_rate)
print("Confusion Matrix (Human Activity Recognition):")
print(har_confusion_matrix)

har_df = pd.DataFrame(
    har_train_features[:, :5], columns=[f"Feature {i+1}" for i in range(5)]
)
har_df["Activity"] = har_train_labels
sns.pairplot(har_df, hue="Activity", plot_kws={"alpha": 0.5})
plt.suptitle(
    "Human Activity Recognition Dataset - Pair Plot of First Five Features", y=1.02
)
plt.tight_layout()
plt.savefig("human_activity_recognition_pairplot.png")
plt.show()

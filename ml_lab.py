# =====================================================================
# HCIA-AI V3.5 - Machine Learning Lab Guide: complete code
# Works as one script, or paste each "# %%" block into its own
# Jupyter / VS Code cell.
#
# Needs:  pip install numpy matplotlib scikit-learn pandas
# Data:   download and unzip ML.zip so you have ./ML/02/lr2_data.txt
#         and ./ML/tennis.txt
#         https://certification-data.obs.cn-north-4.myhuaweicloud.com/ENG/HCIA-AI/V3.5/chapter2/ML.zip
# =====================================================================


# =====================================================================
# EXPERIMENT 1 - Linear regression (scikit-learn)
# =====================================================================
# %%
from sklearn.linear_model import LinearRegression  # Linear regression model
import matplotlib.pyplot as plt                    # Plotting library
import numpy as np

# %%
# Build and visualize a house price dataset.
x = np.array([121, 125, 131, 141, 152, 161]).reshape(-1, 1)  # House area (feature)
y = np.array([300, 350, 425, 405, 496, 517])                 # House price
plt.scatter(x, y)
plt.xlabel("area")
plt.ylabel("price")
plt.show()

# %%
# Train the model.
lr = LinearRegression()
lr.fit(x, y)

# %%
# Slope and intercept of the model.
w = lr.coef_
b = lr.intercept_
print('Slope: ', w)
print('Intercept: ', b)

# %%
# Visualize the fitted line.
plt.scatter(x, y)
plt.xlabel("area")
plt.ylabel("price")
plt.plot([x[0], x[-1]], [x[0] * w + b, x[-1] * w + b])
plt.show()

# %%
# Predict the price of a house with an area of 130.
testX = np.array([[130]])
print(lr.predict(testX))   # expected: [373.13029447]


# =====================================================================
# EXPERIMENT 2 - Linear regression from scratch (numpy, gradient descent)
# =====================================================================
# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# Gradient: 1/m * sum((h(x^i) - y^i) * x_j^i), computed with matrices.
def generate_gradient(X, theta, y):
    sample_count = X.shape[0]
    return (1. / sample_count) * X.T.dot(X.dot(theta) - y)

# %%
# Read the dataset.
def get_training_data(file_path):
    orig_data = np.loadtxt(file_path, skiprows=1)  # Skip the title row
    cols = orig_data.shape[1]
    return (orig_data, orig_data[:, :cols - 1], orig_data[:, cols - 1:])

# %%
# Initialize the theta array.
def init_theta(feature_count):
    return np.ones(feature_count).reshape(feature_count, 1)

# %%
# Gradient descent.
def gradient_descending(X, y, theta, alpha):
    Jthetas = []  # Records the trend of the cost function J(theta)
    # Loss: square of the difference between actual and predicted values
    Jtheta = (X.dot(theta) - y).T.dot(X.dot(theta) - y)
    index = 0
    gradient = generate_gradient(X, theta, y)  # Calculate the gradient
    while not np.all(np.absolute(gradient) <= 1e-5):  # Stop when gradient < 0.00001
        theta = theta - alpha * gradient
        gradient = generate_gradient(X, theta, y)  # New gradient
        Jtheta = (X.dot(theta) - y).T.dot(X.dot(theta) - y)
        if (index + 1) % 10 == 0:
            Jthetas.append((index, Jtheta[0]))  # Record every 10 iterations
        index += 1
    return theta, Jthetas

# %%
# Plot the loss function curve.
def showJTheta(diff_value):
    p_x = []
    p_y = []
    for (index, sum) in diff_value:
        p_x.append(index)
        p_y.append(sum)
    plt.plot(p_x, p_y, color='b')
    plt.xlabel('steps')
    plt.ylabel('loss funtion')
    plt.title('step - loss function curve')
    plt.show()

# %%
# Plot the data points and the fitted line.
def showlinercurve(theta, sample_training_set):
    x, y = sample_training_set[:, 1], sample_training_set[:, 2]
    z = theta[0] + theta[1] * x
    plt.scatter(x, y, color='b', marker='x', label="sample data")
    plt.plot(x, z, 'r', color="r", label="regression curve")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('liner regression curve')
    plt.legend()
    plt.show()

# %%
# Run everything and plot the final results.
training_data_include_y, training_x, y = get_training_data("./ML/02/lr2_data.txt")
sample_count, feature_count = training_x.shape   # Number of samples and features
alpha = 0.01                                     # Learning step
theta = init_theta(feature_count)                # Initialize theta
result_theta, Jthetas = gradient_descending(training_x, y, theta, alpha)
print("w:{}".format(result_theta[0][0]), "b:{}".format(result_theta[1][0]))
showJTheta(Jthetas)
showlinercurve(result_theta, training_data_include_y)
# expected: w:3.0076279423997594 b:1.668677412281192


# =====================================================================
# EXPERIMENT 3 - Logistic regression
# =====================================================================
# %%
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# %%
# Each item in X is [rent, area]; y says whether to rent the room (0: no, 1: yes).
X = [[2200, 15], [2750, 20], [5000, 40], [4000, 20], [3300, 20], [2000, 10], [2500, 12], [12000, 80],
     [2880, 10], [2300, 15], [1500, 10], [3000, 8], [2000, 14], [2000, 10], [2150, 8], [3400, 20],
     [5000, 20], [4000, 10], [3300, 15], [2000, 12], [2500, 14], [10000, 100], [3150, 10],
     [2950, 15], [1500, 5], [3000, 18], [8000, 12], [2220, 14], [6000, 100], [3050, 10]]
y = [1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0]

# %%
# Standardize the data (mean 0, variance 1 per feature).
ss = StandardScaler()
X_train = ss.fit_transform(X)
print(X_train)

# %%
# Train.
lr = LogisticRegression()
lr.fit(X_train, y)

# %%
# Predict.
testX = [[2000, 8]]
X_test = ss.transform(testX)
print("Value to be predicted: ", X_test)
label = lr.predict(X_test)
print("predicted label = ", label)
prob = lr.predict_proba(X_test)   # Predicted probability
print("probability = ", prob)
# expected: label [1], probability [[0.41886952 0.58113048]]


# =====================================================================
# EXPERIMENT 4 - Decision tree
# =====================================================================
# %%
import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

# %%
# Generate a decision tree.
def createTree(trainingData):
    data = trainingData.iloc[:, :-1]    # Feature matrix
    labels = trainingData.iloc[:, -1]   # Labels
    trainedTree = tree.DecisionTreeClassifier(criterion="entropy")
    trainedTree.fit(data, labels)
    return trainedTree

# %%
# Draw the tree with matplotlib (no Graphviz needed) and save it as a PDF.
# (The guide uses Graphviz + pydotplus here; this is a replacement.)
def showtree2pdf(trainedTree, finename):
    plt.figure(figsize=(12, 8))
    tree.plot_tree(trainedTree, filled=True)
    plt.savefig(finename)
    plt.show()

# %%
# Convert categorical columns to numeric codes.
def data2vectoc(data):
    names = data.columns[:-1]
    for i in names:
        col = pd.Categorical(data[i])
        data[i] = col.codes
    return data

# %%
data = pd.read_table("./ML/tennis.txt", header=None, sep='\t')  # Read training data
trainingvec = data2vectoc(data)                                 # Vectorize
decisionTree = createTree(trainingvec)                          # Build the tree
showtree2pdf(decisionTree, "tennis.pdf")                        # Saves tennis.pdf

# %%
# Predict a new sample: sunny, cold, high humidity, strong wind.
# Codes are alphabetical: outlook cloudy=0 rain=1 sunny=2 | temp cold=0 hot=1 moderate=2
#                         humidity high=0 normal=1        | wind strong=0 weak=1
# (The guide's original [0, 0, 1, 1] actually means cloudy, cold, normal, weak.)
testVec = [2, 0, 0, 0]
print(decisionTree.predict(np.array(testVec).reshape(1, -1)))   # expected: ['N']


# =====================================================================
# EXPERIMENT 5 - K-means clustering
# =====================================================================
# %%
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# %%
# Generate the dataset (500 points, 2 features, 4 centers).
X, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)

# %%
print("Dimension of X is {}".format(X.shape))   # (500, 2)
print("Dimension of y is {}".format(y.shape))   # (500,)

# %%
# Scatter plot without labels.
fig, ax1 = plt.subplots(1)
ax1.scatter(X[:, 0], X[:, 1],
            marker='o',  # Point shape
            s=8)         # Point size
plt.show()

# %%
# Scatter plot colored by the generated labels.
color = ["red", "pink", "orange", "green"]
fig, ax1 = plt.subplots(1)
for i in range(4):
    ax1.scatter(X[y == i, 0], X[y == i, 1],
                marker='o',
                s=8,
                c=color[i])
plt.show()

# %%
# K-means with 3 clusters.
n_clusters = 3
cluster1 = KMeans(n_clusters=n_clusters, random_state=3).fit(X)

# %%
y_pred1 = cluster1.labels_
print(y_pred1)

# %%
centroid1 = cluster1.cluster_centers_
print(centroid1)

# %%
# Visualize the 3-cluster result.
color = ["red", "pink", "orange", "gray"]
fig, ax1 = plt.subplots(1)
for i in range(n_clusters):
    ax1.scatter(X[y_pred1 == i, 0], X[y_pred1 == i, 1],
                marker='o',
                s=8,
                c=color[i])
ax1.scatter(centroid1[:, 0], centroid1[:, 1],
            marker="x",
            s=15,
            c="black")
plt.show()

# %%
# K-means with 4 clusters.
n_clusters = 4
cluster2 = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
y_pred2 = cluster2.labels_
centroid2 = cluster2.cluster_centers_
print("Centroid: {}".format(centroid2))

# %%
# Visualize the 4-cluster result.
color = ["red", "pink", "orange", "green"]
fig, ax1 = plt.subplots(1)
for i in range(n_clusters):
    ax1.scatter(X[y_pred2 == i, 0], X[y_pred2 == i, 1],
                marker='o',
                s=8,
                c=color[i])
ax1.scatter(centroid2[:, 0], centroid2[:, 1],
            marker="x",
            s=15,
            c="black")
plt.show()

"""
This file contains the code for various AI and ML algorithms.
"""

# A* Algorithm
a_star_algorithm = """
import heapq

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic (estimated cost from current to goal)
        self.f = 0  # Total cost: g + h

    def __eq__(self, other):
        return self.position == other.position
    
    def __lt__(self, other):
        return self.f < other.f

def heuristic(node, goal):
    # Manhattan distance
    return abs(node.position[0] - goal.position[0]) + abs(node.position[1] - goal.position[1])

def get_neighbors(node, grid):
    neighbors = []
    # Define possible moves (up, right, down, left)
    moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    
    for move in moves:
        new_position = (node.position[0] + move[0], node.position[1] + move[1])
        
        # Check if the new position is valid
        if (0 <= new_position[0] < len(grid) and 
            0 <= new_position[1] < len(grid[0]) and 
            grid[new_position[1]][new_position[0]] != 1):  # 1 represents obstacles
            
            new_node = Node(new_position, node)
            neighbors.append(new_node)
    
    return neighbors

def a_star(grid, start, goal):
    # Create start and goal nodes
    start_node = Node(start)
    goal_node = Node(goal)
    
    # Initialize open and closed lists
    open_list = []
    closed_list = []
    
    # Add the start node to the open list
    heapq.heappush(open_list, start_node)
    
    # Loop until the open list is empty
    while open_list:
        # Get the node with the lowest f value
        current_node = heapq.heappop(open_list)
        closed_list.append(current_node)
        
        # Check if we've reached the goal
        if current_node == goal_node:
            path = []
            current = current_node
            while current:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # Return reversed path
        
        # Generate neighbors
        neighbors = get_neighbors(current_node, grid)
        
        for neighbor in neighbors:
            # Skip if neighbor is in the closed list
            if neighbor in closed_list:
                continue
            
            # Calculate g, h, and f values
            neighbor.g = current_node.g + 1
            neighbor.h = heuristic(neighbor, goal_node)
            neighbor.f = neighbor.g + neighbor.h
            
            # Skip if neighbor is already in open list with a lower f value
            if any(open_node == neighbor and open_node.f < neighbor.f for open_node in open_list):
                continue
            
            # Add neighbor to open list
            heapq.heappush(open_list, neighbor)
    
    # No path found
    return None

# Example usage
if __name__ == "__main__":
    # 0 represents free path, 1 represents obstacles
    grid = [
        [0, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0]
    ]
    
    start = (0, 0)
    goal = (5, 4)
    
    path = a_star(grid, start, goal)
    
    if path:
        print("Path found:", path)
    else:
        print("No path found")
"""

# K-Means Clustering
k_means_algorithm = """
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin

def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    
    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)
        
        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        
        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
    
    return centers, labels

def plot_kmeans(X, n_clusters=4):
    # Generate sample data
    centers, labels = find_clusters(X, n_clusters)
    
    # Plot the data and cluster centers
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate sample data
    X, y = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=0.60)
    
    # Apply K-means clustering
    plot_kmeans(X, n_clusters=4)
"""

# KNN (K-Nearest Neighbors)
knn_algorithm = """
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        # Compute distances
        distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train]
        
        # Get indices of k nearest samples
        k_indices = np.argsort(distances)[:self.k]
        
        # Get the labels of the k nearest samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Example usage
if __name__ == "__main__":
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the KNN classifier
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    
    # Make predictions
    predictions = knn.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Visualize the results (for 2D data)
    if X.shape[1] == 2:
        plt.figure(figsize=(10, 6))
        
        # Plot training points
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', marker='o', s=50, alpha=0.8, label='Training data')
        
        # Plot testing points
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='s', s=50, edgecolors='k', alpha=0.8, label='Testing data')
        
        plt.title('KNN Classification')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()
"""

# Logistic Regression
logistic_regression_algorithm = """
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.num_iterations):
            # Linear model
            linear_model = np.dot(X, self.weights) + self.bias
            # Apply sigmoid function
            y_predicted = self.sigmoid(linear_model)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

# Example usage
if __name__ == "__main__":
    # Generate a random binary classification problem
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the logistic regression model
    model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 6))
    
    # Plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o', s=50, alpha=0.8)
    
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Make predictions on the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    
    plt.title('Logistic Regression Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
"""

# Naive Bayes
naive_bayes_algorithm = """
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Initialize mean, variance, and prior probability for each class
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        
        # Calculate mean, variance, and prior for each class
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / n_samples
    
    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def _predict_single(self, x):
        posteriors = []
        
        # Calculate posterior probability for each class
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)
            
        # Return class with highest posterior probability
        return self.classes[np.argmax(posteriors)]
    
    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

# Example usage
if __name__ == "__main__":
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Use only two features for visualization
    X = X[:, :2]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the Naive Bayes classifier
    nb = GaussianNaiveBayes()
    nb.fit(X_train, y_train)
    
    # Make predictions
    predictions = nb.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 6))
    
    # Plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o', s=50, alpha=0.8)
    
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Make predictions on the mesh grid
    Z = nb.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    
    plt.title('Naive Bayes Decision Boundary')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.show()
"""

# Decision Tree Classifier
decision_tree_algorithm = """
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Example usage
if __name__ == "__main__":
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create and train the Decision Tree classifier
    # We'll use a simple configuration for clarity
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions
    predictions = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")

    # Visualize the decision tree
    plt.figure(figsize=(12, 8))
    plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=list(iris.target_names), rounded=True)
    plt.title("Decision Tree Classifier for Iris Dataset")
    plt.show()

    # Example of predicting a new sample
    # Sample format: [sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)]
    new_sample = np.array([[5.1, 3.5, 1.4, 0.2]]) # Example: Iris Setosa
    prediction_new = clf.predict(new_sample)
    predicted_class_name = iris.target_names[prediction_new[0]]
    print(f"Prediction for new sample {new_sample}: {predicted_class_name}")

    new_sample_versicolor = np.array([[6.0, 2.7, 4.0, 1.2]]) # Example data that might be Versicolor
    prediction_versicolor = clf.predict(new_sample_versicolor)
    predicted_class_name_versicolor = iris.target_names[prediction_versicolor[0]]
    print(f"Prediction for new sample {new_sample_versicolor}: {predicted_class_name_versicolor}")

    new_sample_virginica = np.array([[7.0, 3.2, 6.0, 2.0]]) # Example data that might be Virginica
    prediction_virginica = clf.predict(new_sample_virginica)
    predicted_class_name_virginica = iris.target_names[prediction_virginica[0]]
    print(f"Prediction for new sample {new_sample_virginica}: {predicted_class_name_virginica}")
"""

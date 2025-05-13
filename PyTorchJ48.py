import torch
import pandas as pd
import numpy as np
import time
import sys
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

# Set recursion limit to avoid RecursionError
sys.setrecursionlimit(5000)

class J48DecisionTree:
    def __init__(self, confidence=0.25, min_samples=2, max_depth=50):
        self.tree = None
        self.confidence = confidence
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.most_common_class = None

    def entropy(self, y):
        class_counts = Counter(y)
        total = len(y)
        return -sum((count / total) * np.log2(count / total) for count in class_counts.values() if count > 0)

    def information_gain(self, X, y, feature_index):
        total_entropy = self.entropy(y)
        values, counts = np.unique(X[:, feature_index], return_counts=True)
        weighted_entropy = sum((counts[i] / len(y)) * self.entropy(y[X[:, feature_index] == v]) for i, v in enumerate(values))
        return total_entropy - weighted_entropy

    def best_split(self, X, y):
        best_feature, best_gain = None, -1
        for feature_index in range(X.shape[1]):
            gain = self.information_gain(X, y, feature_index)
            if gain > best_gain:
                best_feature, best_gain = feature_index, gain
        return best_feature

    def prune(self, tree, samples, labels):
        if not isinstance(tree, dict):
            return tree

        feature = next(iter(tree))
        for value in list(tree[feature].keys()):
            subtree = tree[feature][value]
            if isinstance(subtree, dict):
                pruned_subtree = self.prune(subtree, samples[samples[:, feature] == value], labels[samples[:, feature] == value])
                tree[feature][value] = pruned_subtree

        leaf_values = [v for k, v in tree[feature].items() if not isinstance(v, dict)]
        if len(set(leaf_values)) == 1:
            return leaf_values[0]

        return tree

    def fit(self, X, y, depth=0):
        self.most_common_class = Counter(y).most_common(1)[0][0]
        
        if len(set(y)) == 1 or len(y) < self.min_samples or depth >= self.max_depth:
            return self.most_common_class
        if X.shape[1] == 0:
            return self.most_common_class

        best_feature = self.best_split(X, y)
        if best_feature is None:
            return self.most_common_class

        tree = {best_feature: {}}
        for value in np.unique(X[:, best_feature]):
            subset_X = X[X[:, best_feature] == value]
            subset_y = y[X[:, best_feature] == value]
            subtree = self.fit(subset_X, subset_y, depth + 1)
            tree[best_feature][value] = subtree

        self.tree = self.prune(tree, X, y)
        return self.tree

    def predict_instance(self, instance, tree):
        if not isinstance(tree, dict):
            return tree
        feature = next(iter(tree))
        feature_value = instance[feature]
        if feature_value in tree[feature]:
            return self.predict_instance(instance, tree[feature][feature_value])
        else:
            return self.most_common_class

    def predict(self, X):
        return [self.predict_instance(x, self.tree) for x in X]

def perform_cross_validation(filepath, n_splits=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and normalize data
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    le = LabelEncoder()
    y = le.fit_transform(y)

    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.int64).to(device)

    kf = KFold(n_splits=n_splits, shuffle=True)
    accuracies = []
    all_true, all_pred = [], []

    total_start_time = time.time()
    
    for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
        print(f"Training Fold {fold}")
        
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = J48DecisionTree(confidence=0.25, min_samples=2, max_depth=50)
        start_time = time.time()
        model.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
        end_time = time.time()

        y_pred = model.predict(X_val.cpu().numpy())
        accuracy = accuracy_score(y_val.cpu().numpy(), y_pred)
        accuracies.append(accuracy)

        all_true.extend(y_val.cpu().numpy())
        all_pred.extend(y_pred)

        print(f"Fold {fold} Accuracy: {accuracy:.4f}")
        print(f"Training Time: {end_time - start_time:.4f} seconds")
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    print("Average Accuracy: {:.4f}".format(np.mean(accuracies)))
    print("Standard Deviation: {:.4f}".format(np.std(accuracies)))
    print(f"Total Training Time: {total_duration:.4f} seconds")
    print("Classification Report:")
    print(classification_report(all_true, all_pred, target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(all_true, all_pred))

perform_cross_validation('hdcData.csv', n_splits=10)

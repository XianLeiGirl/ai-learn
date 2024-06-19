import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 加载示例数据集（以鸢尾花数据集为例）
data = load_iris()
X = data.data
y = data.target

# One-hot 编码目标变量
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义决策树节点类
class DecisionTreeNode:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

# 计算基尼指数
def gini(y):
    m = y.shape[0]
    return 1.0 - np.sum((np.sum(y, axis=0) / m) ** 2)

def gini_split(X_column, threshold, y):
    left_mask = X_column <= threshold
    right_mask = X_column > threshold
    left_gini = gini(y[left_mask])
    right_gini = gini(y[right_mask])
    n = len(y)
    n_left = np.sum(left_mask)
    n_right = np.sum(right_mask)
    return (n_left / n) * left_gini + (n_right / n) * right_gini

# 构建决策树
class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y[:, i]) for i in range(y.shape[1])]
        predicted_class = np.argmax(num_samples_per_class)
        node = DecisionTreeNode(
            gini=gini(y),
            num_samples=y.shape[0],
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] <= thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        num_parent = [np.sum(y[:, c]) for c in range(y.shape[1])]
        best_gini = 1.0 - sum((num / m) ** 2 for num in num_parent)
        best_idx, best_thr = None, None

        for idx in range(n):
            sorted_indices = np.argsort(X[:, idx])
            thresholds = X[sorted_indices, idx]
            classes = y[sorted_indices]
            num_left = [0] * y.shape[1]
            num_right = num_parent.copy()
            for i in range(1, m):
                c = np.argmax(classes[i - 1])
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(y.shape[1]))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(y.shape[1]))
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def _predict(self, inputs):
        node = self.tree
        while node.left is not None:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

# 创建决策树分类器
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = np.sum(y_pred == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("Accuracy:", accuracy)


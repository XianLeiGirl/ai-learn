import numpy as np
import tensorflow as tf
from sklearn.datasets import make_blobs

# 生成样本数据
n_samples = 1500
n_features = 2
n_clusters = 3

data, labels = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=42)

# 将数据转换为TensorFlow张量
data = tf.convert_to_tensor(data, dtype=tf.float32)

# K均值聚类参数
k = 3
max_iters = 100

# 随机选择初始聚类中心
def initialize_centroids(data, k):
    indices = tf.random.shuffle(tf.range(tf.shape(data)[0]))[:k]
    centroids = tf.gather(data, indices)
    return centroids

# 计算点到簇中心的距离
def compute_distances(data, centroids):
    expanded_data = tf.expand_dims(data, axis=1)
    expanded_centroids = tf.expand_dims(centroids, axis=0)
    distances = tf.reduce_sum(tf.square(expanded_data - expanded_centroids), axis=2)
    return distances

# 重新分配每个点到最近的簇
def assign_clusters(data, centroids):
    distances = compute_distances(data, centroids)
    cluster_assignments = tf.argmin(distances, axis=1)
    return cluster_assignments

# 计算新的聚类中心
def update_centroids(data, cluster_assignments, k):
    new_centroids = []
    for i in range(k):
        points_in_cluster = tf.boolean_mask(data, cluster_assignments == i)
        new_centroid = tf.reduce_mean(points_in_cluster, axis=0)
        new_centroids.append(new_centroid)
    new_centroids = tf.stack(new_centroids)
    return new_centroids

# K均值聚类算法
def kmeans(data, k, max_iters):
    centroids = initialize_centroids(data, k)
    for i in range(max_iters):
        cluster_assignments = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, cluster_assignments, k)
        if tf.reduce_all(tf.equal(new_centroids, centroids)):
            break
        centroids = new_centroids
    return centroids, cluster_assignments

# 运行K均值聚类
centroids, cluster_assignments = kmeans(data, k, max_iters)

# 打印聚类中心
print("聚类中心:")
print(centroids.numpy())

# 可视化聚类结果
import matplotlib.pyplot as plt

plt.scatter(data[:, 0], data[:, 1], c=cluster_assignments.numpy(), cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
plt.title('K-means Clustering with TensorFlow 2.x')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


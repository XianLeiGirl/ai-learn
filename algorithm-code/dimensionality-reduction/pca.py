import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载 MNIST 数据集
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_data = np.concatenate([x_train, x_test], axis=0)
x_data = x_data / 255.0  # 归一化

# 2. 数据预处理：展平图像数据
n_samples, h, w = x_data.shape
x_flattened = x_data.reshape(n_samples, h * w).astype(np.float32)

# 3. 计算协方差矩阵
x_mean = tf.reduce_mean(x_flattened, axis=0)
x_centered = x_flattened - x_mean
cov_matrix = tf.matmul(x_centered, x_centered, transpose_a=True) / tf.cast(n_samples, tf.float32)

# 4. 特征值和特征向量
eigenvalues, eigenvectors = tf.linalg.eigh(cov_matrix)

# 5. 选择前 k 个主成分
k = 50
eigenvectors = tf.reverse(eigenvectors, axis=[1])
selected_eigenvectors = eigenvectors[:, :k]

# 6. 降维和重构
x_pca = tf.matmul(x_centered, selected_eigenvectors)
x_reconstructed = tf.matmul(x_pca, selected_eigenvectors, transpose_b=True) + x_mean
x_reconstructed = tf.reshape(x_reconstructed, (n_samples, h, w))

# 7. 展示原始和重构图像
def plot_images(images, title, n_row=2, n_col=5):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.suptitle(title, size=16)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.xticks(())
        plt.yticks(())
    plt.show()

plot_images(x_data, "Original Images")
plot_images(x_reconstructed.numpy(), f"Reconstructed Images with {k} Components")


import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 只选择两类用于二分类
X = X[y != 2]
y = y[y != 2]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

# 定义SVM模型
class SVM(tf.Module):
    def __init__(self):
        self.w = tf.Variable(tf.random.normal(shape=[X_train.shape[1], 1], dtype=tf.float32))
        self.b = tf.Variable(tf.random.normal(shape=[1], dtype=tf.float32))

    def __call__(self, X):
        return tf.matmul(X, self.w) + self.b

    def loss(self, X, y):
        regularization_loss = 0.5 * tf.reduce_sum(self.w ** 2)
        hinge_loss = tf.reduce_mean(tf.maximum(0., 1. - y * self.__call__(X)))
        return regularization_loss + hinge_loss

# 标签转换为-1和1
y_train = y_train * 2 - 1
y_test = y_test * 2 - 1
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# 创建SVM模型
model = SVM()

# 训练模型
optimizer = tf.optimizers.Adam(learning_rate=0.01)
num_epochs = 100

for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        loss = model.loss(X_train, y_train[:, np.newaxis])
    gradients = tape.gradient(loss, [model.w, model.b])
    optimizer.apply_gradients(zip(gradients, [model.w, model.b]))
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')

# 预测函数
def predict(X):
    return tf.sign(model(X))

# 评估模型
train_accuracy = tf.reduce_mean(tf.cast(tf.equal(predict(X_train), y_train[:, np.newaxis]), tf.float32))
test_accuracy = tf.reduce_mean(tf.cast(tf.equal(predict(X_test), y_test[:, np.newaxis]), tf.float32))

print(f'Training accuracy: {train_accuracy.numpy() * 100:.2f}%')
print(f'Test accuracy: {test_accuracy.numpy() * 100:.2f}%')


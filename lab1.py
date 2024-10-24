import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. 加载数据集并进行预处理
data = load_breast_cancer()
X = data.data  # 输入特征
y = data.target  # 目标标签（0或1）

# 进行标准化处理，确保特征均值为0，方差为1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 2. 定义网络结构
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=X_train.shape[1]))  # 第一层，10个神经元
model.add(Dense(8, activation='relu'))  # 第二层，8个神经元
model.add(Dense(8, activation='relu'))  # 第三层，8个神经元
model.add(Dense(4, activation='relu'))  # 第四层，4个神经元
model.add(Dense(1, activation='sigmoid'))  # 输出层，1个神经元（Sigmoid）

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. 训练模型
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# 4. 模型评估
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")

# 5. 可视化训练过程中的损失和准确率
# 绘制训练过程中的准确率变化
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 绘制训练过程中的损失变化
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

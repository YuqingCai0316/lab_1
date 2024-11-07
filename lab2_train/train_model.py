import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model_structure import create_model, visualize_model

# 1. Load and preprocess the dataset
data = load_breast_cancer()
X = data.data  # Input features
y = data.target  # Target labels (0 or 1)

# Standardize features to have mean=0 and variance=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 2. Load model structure and compile the model
model = create_model(X_train.shape[1])

# Visualize and save the model structure
visualize_model(model)

# 3. Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# 4. Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")

# Optional: Plot the training accuracy and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

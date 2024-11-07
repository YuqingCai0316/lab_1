import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

def create_model(input_shape):
    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim=input_shape))  # First layer with 10 neurons
    model.add(Dense(8, activation='relu'))  # Second layer with 8 neurons
    model.add(Dense(8, activation='relu'))  # Third layer with 8 neurons
    model.add(Dense(4, activation='relu'))  # Fourth layer with 4 neurons
    model.add(Dense(1, activation='sigmoid'))  # Output layer with 1 neuron (Sigmoid)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def visualize_model(model, file_path='model_structure.png'):
    plot_model(model, to_file=file_path, show_shapes=True, show_layer_names=True)
    print(f"Model structure saved as {file_path}")

import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

MODEL_PATH = 'mnist_digit_classifier_tf.keras'

def train_model():
    # Load and prep the MNIST image dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    # Build the model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), # Learning 32 3x3 filters, with an ReLU avctivation and an input shape of a 28x28 pixel grid.
        layers.MaxPooling2D((2, 2)), # shrink input to 2x2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(), # flattening our unit to shape it to the NN
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5), # helps to prevent overfitting
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

    # Evaluate the model for accuracy
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc}")

    # make the final prediction
    predictions = model.predict(test_images)

    # Store the model for later use
    model.save(MODEL_PATH)

    return predictions, test_images, test_labels # for later use if needed

def init():
    if not os.path.exists(MODEL_PATH):
        print("Training the model")
        train_model()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

if __name__ == '__main__':
    init()

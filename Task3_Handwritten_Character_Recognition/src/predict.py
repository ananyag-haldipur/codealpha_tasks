import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


# Load the trained model
MODEL_PATH = "models/handwritten_digit_cnn.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully! ✅")


# Load MNIST test data
(_, _), (X_test, y_test) = mnist.load_data()

# Normalize the images
X_test = X_test.astype("float32") / 255.0

# Add channel dimension
X_test = np.expand_dims(X_test, axis=-1)


# Select one test image
index = 0

image = X_test[index]
actual_digit = y_test[index]


# Make prediction
prediction = model.predict(
    np.expand_dims(image, axis=0),
    verbose=0
)

predicted_digit = np.argmax(prediction)
confidence = np.max(prediction) * 100


# Display result
print(f"Actual Digit: {actual_digit}")
print(f"Predicted Digit: {predicted_digit}")
print(f"Confidence: {confidence:.2f}%")


# Display image
plt.imshow(image.squeeze(), cmap="gray")
plt.title(
    f"Actual: {actual_digit} | "
    f"Predicted: {predicted_digit} "
    f"({confidence:.2f}%)"
)
plt.axis("off")
plt.show()
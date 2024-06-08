import os
from utils import load_data
from model import create_model
import tensorflow as tf

# Define data directory
data_dir = "dataset"

# Load data
X_train, X_test, y_train, y_test = load_data(data_dir)

# Create model
model = create_model()

# Train model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, batch_size=32)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Save the model
model.save("model.h5")

# Predict on a new image
def predict_image(model, img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0) / 255.0
    prediction = model.predict(img)
    return "Wanted" if prediction > 0.5 else "Not Wanted"

print(predict_image(model, "path_to_new_image.jpg"))

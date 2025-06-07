import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils import plot_grad_cam  # Import Grad-CAM function

def predict_image(image_path, model_path, label_map, visualize_grad_cam=False):
    """Loads model, processes image, and predicts disease class."""
    model = load_model(model_path)
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img, axis=0) / 255.0
    
    # Make prediction
    pred = model.predict(img_array)[0]
    predicted_class = np.argmax(pred)
    confidence = pred[predicted_class]
    class_name = list(label_map.keys())[list(label_map.values()).index(predicted_class)]
    
    # Optional: Visualize Grad-CAM heatmap
    if visualize_grad_cam:
        plot_grad_cam(model, img_array)

    return class_name, confidence

# Example Usage:
# label_map = {"Healthy": 0, "Disease1": 1, "Disease2": 2}
# class_name, confidence = predict_image("test_leaf.jpg", "model/leaf_model.h5", label_map, visualize_grad_cam=True)
# print(f"Predicted: {class_name}, Confidence: {confidence:.2f}")

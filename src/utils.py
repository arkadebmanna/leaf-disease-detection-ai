import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import confusion_matrix

def plot_metrics(history):
    """Plot training accuracy and loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.show()

def show_confusion_matrix(y_true, y_pred, class_names):
    """Generate and display the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def grad_cam(model, img_array, layer_name="Conv_1"):
    """
    Generate Grad-CAM heatmap for a given image.

    Args:
    - model: Trained Keras model
    - img_array: Preprocessed image as numpy array
    - layer_name: Target layer for Grad-CAM (default: last convolutional layer)

    Returns:
    - Heatmap overlayed on the original image
    """
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlayed_img = cv2.addWeighted(img_array[0] * 255, 0.6, heatmap, 0.4, 0)

    return overlayed_img

def plot_grad_cam(model, img_array):
    """Display Grad-CAM heatmap."""
    gradcam_img = grad_cam(model, img_array)
    plt.imshow(gradcam_img.astype("uint8"))
    plt.axis("off")
    plt.title("Grad-CAM Heatmap")
    plt.show()

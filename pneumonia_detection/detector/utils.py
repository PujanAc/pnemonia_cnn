"""
Utility functions for ML predictions and Grad-CAM visualization.
"""

import os
import numpy as np
from PIL import Image
import io
from django.core.files.base import ContentFile

# TensorFlow imports
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications.resnet50 import preprocess_input
    TF_AVAILABLE = True
except ImportError:
    print("WARNING: TensorFlow not available. Using mock predictions.")


class ModelPredictor:
    """
    Handles model loading, prediction, and Grad-CAM preparation.
    """

    def __init__(self, model_type='custom_cnn'):
        self.model_type = model_type
        self.model = None
        self.model_loaded = False

        if TF_AVAILABLE:
            self._load_model()

    def _load_model(self):
        """Load the model from disk."""
        from django.conf import settings
        model_path = settings.MODEL_PATHS.get(self.model_type)
        if model_path and os.path.exists(model_path):
            try:
                self.model = keras.models.load_model(model_path, compile=False, safe_mode=False)
                self.model_loaded = True
                print(f"✓ Model '{self.model_type}' loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load {self.model_type}: {e}")
                self.model_loaded = False
        else:
            print(f"✗ Model path not found: {model_path}")
            self.model_loaded = False

    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Preprocess image for model input."""
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size, Image.LANCZOS)
        if self.model_type == 'custom_cnn':
            img_array = np.array(img) / 255.0 
        else:     # Normalize to [0

           img_array = preprocess_input(np.array(img))  # ImageNet preprocessing
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image_path):
        """Predict class and confidence for an image."""
        if self.model_loaded and self.model:
            try:
                img_array = self.preprocess_image(image_path)
                prediction = self.model.predict(img_array, verbose=0)[0]

                # Handle binary sigmoid vs multiclass softmax
                if prediction.ndim == 0 or prediction.shape[-1] == 1:
                    pneumonia_prob = float(prediction)
                else:
                    pneumonia_prob = float(prediction[1])

                normal_prob = 1 - pneumonia_prob
                predicted_class = 'pneumonia' if pneumonia_prob > 0.5 else 'normal'
                confidence = max(pneumonia_prob, normal_prob) * 100

                return {
                    'class': predicted_class,
                    'confidence': float(confidence),
                    'probabilities': {
                        'normal': float(normal_prob * 100),
                        'pneumonia': float(pneumonia_prob * 100)
                    }
                }

            except Exception as e:
                print(f"Prediction error: {e}")
                return self._mock_prediction()
        else:
            return self._mock_prediction()

    def _mock_prediction(self):
        """Return random prediction when model is unavailable."""
        import random
        predicted_class = 'pneumonia' if random.random() > 0.4 else 'normal'
        if self.model_type == 'resnet50':
            confidence = random.uniform(85, 98)
        else:
            confidence = random.uniform(78, 95)
        if predicted_class == 'pneumonia':
            pneumonia_prob = confidence
            normal_prob = 100 - confidence
        else:
            normal_prob = confidence
            pneumonia_prob = 100 - confidence

        return {
            'class': predicted_class,
            'confidence': confidence,
            'probabilities': {
                'normal': normal_prob,
                'pneumonia': pneumonia_prob
            },
            'mock': True
        }

    def get_last_conv_layer_name(self):
        """Return the name of the last conv layer for Grad-CAM."""
        if not self.model_loaded or not self.model:
            return None
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower():
                return layer.name
        return None


def generate_gradcam(model_predictor, image_path, target_size=(224, 224)):
    """Generate Grad-CAM heatmap and overlay on image."""
    if not model_predictor.model_loaded or not model_predictor.model:
        return generate_mock_gradcam(image_path, target_size)

    try:
        img_array = model_predictor.preprocess_image(image_path, target_size)
        last_conv = model_predictor.get_last_conv_layer_name()
        if not last_conv:
            return generate_mock_gradcam(image_path, target_size)

        grad_model = keras.models.Model(
            inputs=model_predictor.model.input,
            outputs=[
                model_predictor.model.get_layer(last_conv).output,
                model_predictor.model.output
            ]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            tape.watch(conv_outputs)
            if predictions.shape[-1] == 1:
                class_channel = predictions[:, 0]
            else:
                class_channel = predictions[:, tf.argmax(predictions[0])]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap /= tf.reduce_max(heatmap) + 1e-10
        heatmap = heatmap.numpy()

        # Resize & overlay
        heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(target_size, Image.LANCZOS)
        original_img = Image.open(image_path).convert("RGB").resize(target_size, Image.LANCZOS)
        heatmap_colored = apply_colormap(np.array(heatmap_img))
        superimposed = Image.blend(original_img, Image.fromarray(heatmap_colored), alpha=0.4)
        return superimposed

    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return generate_mock_gradcam(image_path, target_size)


def generate_mock_gradcam(image_path, target_size=(224, 224)):
    """Generate a demo Grad-CAM heatmap for mock predictions."""
    try:
        original_img = Image.open(image_path).convert("RGB").resize(target_size, Image.LANCZOS)
        height, width = target_size
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        heatmap = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(height, width) / 3)**2))
        noise = np.random.rand(height, width) * 0.3
        heatmap = (heatmap * (1 - noise) - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
        heatmap_colored = apply_colormap(np.uint8(255 * heatmap))
        return Image.blend(original_img, Image.fromarray(heatmap_colored), alpha=0.4)
    except Exception as e:
        print(f"Mock Grad-CAM error: {e}")
        return None


def apply_colormap(heatmap_array):
    """Apply jet-like colormap to a heatmap array."""
    normalized = heatmap_array / 255.0
    r = np.clip(1.5 - 4 * np.abs(normalized - 0.75), 0, 1)
    g = np.clip(1.5 - 4 * np.abs(normalized - 0.5), 0, 1)
    b = np.clip(1.5 - 4 * np.abs(normalized - 0.25), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return np.uint8(rgb * 255)


def save_gradcam_to_file(gradcam_image):
    """Save PIL Grad-CAM image to Django ContentFile."""
    if gradcam_image is None:
        return None
    try:
        buffer = io.BytesIO()
        gradcam_image.save(buffer, format="PNG")
        buffer.seek(0)
        return ContentFile(buffer.read(), name="gradcam.png")
    except Exception as e:
        print(f"Error saving Grad-CAM: {e}")
        return None

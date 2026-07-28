import os
import numpy as np
import tensorflow as tf
from src.ml.dataset_prep import CATEGORIES
from src.ml.train_classifier import build_and_train_classifier
from config.settings import settings

class DocumentClassifier:
    def __init__(self):
        self.model_path = str(settings.MODEL_PATH)
        self.model = None
        self._load_or_train_model()

    def _load_or_train_model(self):
        """
        Loads the classifier model from disk, or trains a new one if it's missing.
        """
        if not os.path.exists(self.model_path):
            print(f"Classifier model not found at {self.model_path}. Training a new model...")
            try:
                self.model, _ = build_and_train_classifier()
            except Exception as e:
                print(f"Error training model on startup: {e}")
                # Fallback to empty model container if training fails (or we will handle it gracefully)
                self.model = None
        else:
            try:
                print(f"Loading TensorFlow classifier from {self.model_path}...")
                self.model = tf.keras.models.load_model(self.model_path)
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Failed to load model from disk: {e}. Retraining...")
                try:
                    self.model, _ = build_and_train_classifier()
                except Exception as ex:
                    print(f"Error training model: {ex}")
                    self.model = None

    def predict_category(self, text: str) -> str:
        """
        Predicts the category of a given text block (e.g. abstract of a document).
        """
        if not self.model:
            print("Classifier model not initialized. Returning default 'Unknown'.")
            return "Unknown"

        if not text or not text.strip():
            return "Unknown"

        try:
            # The model has TextVectorization layer built-in, so we can pass raw string array
            predictions = self.model.predict(tf.convert_to_tensor([text], dtype=tf.string), verbose=0)
            class_idx = int(np.argmax(predictions[0]))
            
            # Confidence check
            confidence = float(predictions[0][class_idx])
            print(f"Predicted category index: {class_idx} ({CATEGORIES[class_idx]}) with confidence {confidence:.4f}")
            
            return CATEGORIES[class_idx]
        except Exception as e:
            print(f"Inference failed: {e}")
            return "Unknown"

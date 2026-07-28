import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from src.ml.dataset_prep import generate_synthetic_dataset, CATEGORIES
from config.settings import settings

def build_and_train_classifier():
    """
    Builds, trains, evaluates and saves a TensorFlow model for document classification.
    """
    print("Preparing synthetic dataset for training...")
    train_texts, train_labels, val_texts, val_labels = generate_synthetic_dataset(samples_per_category=200)

    print(f"Dataset generated. Train size: {len(train_texts)}, Val size: {len(val_texts)}")

    # Parameters
    vocab_size = 5000
    max_len = 150
    num_classes = len(CATEGORIES)

    print("Building Text Vectorization layer...")
    # Initialize Vectorization Layer
    vectorize_layer = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=max_len
    )
    # Adapt vectorization to training texts
    vectorize_layer.adapt(tf.convert_to_tensor(train_texts, dtype=tf.string))

    print("Building Sequential Keras model...")
    # Build Neural Network
    model = models.Sequential([
        vectorize_layer,
        layers.Embedding(vocab_size, 64, mask_zero=True),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    print("Compiling model...")
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Training TensorFlow model...")
    # Prepare datasets as tensorflow tensors / numpy arrays
    x_train = tf.convert_to_tensor(train_texts, dtype=tf.string)
    y_train = np.array(train_labels)
    x_val = tf.convert_to_tensor(val_texts, dtype=tf.string)
    y_val = np.array(val_labels)

    # Train model
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=12,
        batch_size=32,
        verbose=1
    )

    # Evaluate model
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    print(f"Validation complete. Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    # Ensure models directory exists
    settings.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving TensorFlow model to {settings.MODEL_PATH}...")
    # In newer Keras/TF, saving as .h5 or natively works.
    model.save(str(settings.MODEL_PATH))

    # We also save the vocabulary just in case we need it independently,
    # though it is packaged in the saved model file.
    vocab = vectorize_layer.get_vocabulary()
    with open(settings.TOKENIZER_PATH, 'wb') as f:
        pickle.dump(vocab, f)

    print("Model and vocabulary saved successfully.")
    return model, history

if __name__ == "__main__":
    build_and_train_classifier()

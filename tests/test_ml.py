import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from src.ml.dataset_prep import generate_synthetic_dataset, CATEGORIES

def test_dataset_generation():
    train_texts, train_labels, val_texts, val_labels = generate_synthetic_dataset(samples_per_category=5)
    assert len(train_texts) == 7 * 4  # 80% of 5 is 4
    assert len(val_texts) == 7 * 1    # 20% of 5 is 1
    assert len(train_labels) == len(train_texts)
    assert len(val_labels) == len(val_texts)

def test_tiny_model_training():
    # Arrange simple inputs
    train_texts = [
        "cognitive agent system heuristic search",
        "deep neural net gradient descent loss optimization",
        "image bounding boxes YOLO object detection feature",
        "attention model translation token sequence",
        "robotic arm kinematics trajectory SLAM navigation",
        "sql injection script payload malware firewall",
        "cloud container scaling vm kubernetes cloud storage"
    ] * 2
    train_labels = [0, 1, 2, 3, 4, 5, 6] * 2

    # Vectorizer
    vectorize_layer = layers.TextVectorization(
        max_tokens=100,
        output_mode='int',
        output_sequence_length=10
    )
    vectorize_layer.adapt(tf.convert_to_tensor(train_texts, dtype=tf.string))

    # Neural network
    model = models.Sequential([
        vectorize_layer,
        layers.Embedding(100, 16, mask_zero=True),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation='relu'),
        layers.Dense(7, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Act: train 1 epoch
    history = model.fit(
        tf.convert_to_tensor(train_texts, dtype=tf.string),
        np.array(train_labels),
        epochs=1,
        verbose=0
    )

    # Assert
    assert len(history.history['loss']) == 1
    
    # Run prediction
    pred = model.predict(tf.convert_to_tensor(["test cloud kubernetes container"], dtype=tf.string), verbose=0)
    assert pred.shape == (1, 7)
    assert np.isclose(np.sum(pred[0]), 1.0, atol=1e-5)

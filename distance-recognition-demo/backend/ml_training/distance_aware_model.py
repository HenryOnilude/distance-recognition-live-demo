"""
Distance-Aware Face Recognition Model using CelebA + Keras Transfer Learning
Combines techniques from your research with production ML practices
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import pandas as pd
import cv2
import os
from typing import Dict, Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistanceAwareFaceModel:
    """
    Production ML model combining:
    - Keras Transfer Learning (MobileNetV2 + custom heads)
    - CelebA Dataset (202K labeled faces)
    - Distance degradation simulation
    - Multi-input architecture (face + distance + quality)
    """

    def __init__(self):
        self.model = None
        self.base_model = None
        self.is_trained = False

        # Distance categories from your research
        self.distance_ranges = {
            'portrait': {'min': 0.5, 'max': 1.0, 'target_accuracy': 0.995},
            'close': {'min': 1.0, 'max': 4.0, 'target_accuracy': 0.891},
            'medium': {'min': 4.0, 'max': 7.0, 'target_accuracy': 0.823},
            'far': {'min': 7.0, 'max': 10.0, 'target_accuracy': 0.723}
        }

    def simulate_distance_degradation(self, image: np.ndarray, distance_m: float) -> np.ndarray:
        """
        Simulate how faces degrade at different distances
        Based on physics of camera capture + empirical research
        """
        h, w = image.shape[:2]

        if distance_m <= 1.0:  # Portrait
            # Minimal degradation - very close high-quality
            augmented = image
            # Light variations only
            if np.random.random() > 0.5:
                brightness_factor = np.random.uniform(0.95, 1.05)
                augmented = np.clip(augmented * brightness_factor, 0, 1)

        elif distance_m <= 4.0:  # Close
            # Light degradation
            scale = np.random.uniform(0.8, 1.0)  # Slight resolution loss
            temp_size = (int(w * scale), int(h * scale))
            small = cv2.resize(image, temp_size)

            # Light blur
            if np.random.random() > 0.7:
                small = cv2.GaussianBlur(small, (3, 3), 0.5)

            augmented = cv2.resize(small, (w, h))

            # Minimal noise
            noise = np.random.normal(0, 0.01, augmented.shape)
            augmented = np.clip(augmented + noise, 0, 1)

        elif distance_m <= 7.0:  # Medium
            # Moderate degradation - noticeable quality loss
            scale = np.random.uniform(0.4, 0.7)  # Significant resolution loss
            temp_size = (int(w * scale), int(h * scale))
            small = cv2.resize(image, temp_size)

            # Moderate blur
            blur_kernel = np.random.choice([3, 5])
            small = cv2.GaussianBlur(small, (blur_kernel, blur_kernel), 1.0)

            augmented = cv2.resize(small, (w, h))

            # Moderate noise
            noise = np.random.normal(0, 0.025, augmented.shape)
            augmented = np.clip(augmented + noise, 0, 1)

        else:  # Far (7-10m)
            # Heavy degradation - challenging recognition
            scale = np.random.uniform(0.2, 0.4)  # Heavy resolution loss
            temp_size = (int(w * scale), int(h * scale))
            small = cv2.resize(image, temp_size)

            # Heavy blur
            blur_kernel = np.random.choice([5, 7])
            small = cv2.GaussianBlur(small, (blur_kernel, blur_kernel), 2.0)

            augmented = cv2.resize(small, (w, h))

            # Heavy noise (camera sensor limitations)
            noise = np.random.normal(0, 0.05, augmented.shape)
            augmented = np.clip(augmented + noise, 0, 1)

        return augmented.astype(np.float32)

    def calculate_advanced_quality(self, image: np.ndarray) -> float:
        """
        Calculate image quality score
        TODO: Replace with CelebAMask-HQ segmentation-based quality when available
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(1.0, laplacian_var / 1000.0)

        # Contrast (standard deviation)
        contrast = np.std(gray) / 128.0

        # Brightness distribution
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / hist.sum()
        brightness_uniformity = 1.0 - np.sum(hist_norm * np.log(hist_norm + 1e-7))
        brightness_uniformity = min(1.0, brightness_uniformity / 8.0)

        # Combined quality
        quality = (sharpness * 0.5 + contrast * 0.3 + brightness_uniformity * 0.2)
        return min(1.0, max(0.1, quality))

    def build_model(self) -> Model:
        """
        Build distance-aware model using Keras Functional API + Transfer Learning
        Architecture: Multi-input (face + distance + quality) → Multi-output (age + gender + confidence)
        """
        logger.info("Building distance-aware face recognition model...")

        # ===================
        # INPUTS (Multi-input from Functional API)
        # ===================
        face_input = keras.Input(shape=(224, 224, 3), name='face_image')
        distance_input = keras.Input(shape=(1,), name='distance_meters')
        quality_input = keras.Input(shape=(1,), name='quality_score')

        # ===================
        # TRANSFER LEARNING: Pre-trained Base Model
        # ===================
        self.base_model = keras.applications.MobileNetV2(
            weights='imagenet',  # Pre-trained on ImageNet
            include_top=False,   # Remove top classifier
            input_shape=(224, 224, 3)
        )

        # Freeze base model initially (Transfer Learning best practice)
        self.base_model.trainable = False

        # Extract features from face
        x = self.base_model(face_input, training=False)  # Inference mode during training
        x = layers.GlobalAveragePooling2D()(x)

        # ===================
        # DISTANCE & QUALITY CONDITIONING
        # ===================
        # Embed distance into feature space
        distance_features = layers.Dense(32, activation='relu', name='distance_embed')(distance_input)
        distance_features = layers.Dropout(0.2)(distance_features)

        # Embed quality into feature space
        quality_features = layers.Dense(16, activation='relu', name='quality_embed')(quality_input)
        quality_features = layers.Dropout(0.2)(quality_features)

        # Concatenate all features
        combined = layers.concatenate([x, distance_features, quality_features], name='combined_features')

        # ===================
        # SHARED REPRESENTATION
        # ===================
        shared = layers.Dense(256, activation='relu', name='shared_dense')(combined)
        shared = layers.BatchNormalization()(shared)
        shared = layers.Dropout(0.5)(shared)
        shared = layers.Dense(128, activation='relu')(shared)
        shared = layers.Dropout(0.3)(shared)

        # ===================
        # MULTIPLE OUTPUTS (Multi-output from Functional API)
        # ===================
        # Age prediction (binary: Young=1, Old=0)
        age_output = layers.Dense(1, activation='sigmoid', name='age')(shared)

        # Gender prediction (binary: Male=1, Female=0)
        gender_output = layers.Dense(1, activation='sigmoid', name='gender')(shared)

        # Confidence prediction (0-1 scale)
        confidence_output = layers.Dense(1, activation='sigmoid', name='confidence')(shared)

        # ===================
        # BUILD MODEL
        # ===================
        model = Model(
            inputs={
                'face_image': face_input,
                'distance_meters': distance_input,
                'quality_score': quality_input
            },
            outputs={
                'age': age_output,
                'gender': gender_output,
                'confidence': confidence_output
            },
            name='distance_aware_face_recognition'
        )

        logger.info(f"Model built with {model.count_params():,} parameters")
        return model

    def create_dataset_from_celeba(self, celeba_path: str, num_samples: int = 100000) -> Dict:
        """
        Create distance-labeled dataset from CelebA

        Args:
            celeba_path: Path to CelebA dataset folder
            num_samples: Number of samples to generate

        Returns:
            Dictionary with training data
        """
        logger.info(f"Creating dataset from CelebA: {num_samples:,} samples")

        # Load CelebA annotations
        attr_file = os.path.join(celeba_path, 'list_attr_celeba.txt')
        if not os.path.exists(attr_file):
            raise FileNotFoundError(f"CelebA attributes file not found: {attr_file}")

        df = pd.read_csv(attr_file, sep='\\s+', skiprows=1)

        # Convert attributes from -1/1 to 0/1
        df['Young'] = (df['Young'] == 1).astype(int)
        df['Male'] = (df['Male'] == 1).astype(int)

        logger.info(f"Loaded CelebA with {len(df):,} images")

        # Generate samples at different distances
        distance_categories = ['portrait', 'close', 'medium', 'far']
        samples_per_category = num_samples // len(distance_categories)

        dataset = {
            'images': [],
            'distances': [],
            'qualities': [],
            'age_labels': [],
            'gender_labels': [],
            'expected_confidences': []
        }

        for category in distance_categories:
            logger.info(f"Generating {samples_per_category:,} samples for {category} distance...")

            range_info = self.distance_ranges[category]

            for i in range(samples_per_category):
                # Sample random CelebA image
                idx = np.random.randint(len(df))
                row = df.iloc[idx]

                # Load image (placeholder - implement actual loading)
                # img = load_celeba_image(row['image_id'])
                img = np.random.rand(224, 224, 3).astype(np.float32)  # Placeholder

                # Random distance within category range
                distance = np.random.uniform(range_info['min'], range_info['max'])

                # Apply distance degradation
                degraded = self.simulate_distance_degradation(img, distance)

                # Calculate quality
                quality = self.calculate_advanced_quality(degraded)

                # Expected confidence based on distance (from your research)
                expected_conf = range_info['target_accuracy'] * np.random.uniform(0.9, 1.1)
                expected_conf = np.clip(expected_conf, 0.1, 0.99)

                # Append to dataset
                dataset['images'].append(degraded)
                dataset['distances'].append(distance)
                dataset['qualities'].append(quality)
                dataset['age_labels'].append(row['Young'])
                dataset['gender_labels'].append(row['Male'])
                dataset['expected_confidences'].append(expected_conf)

        # Convert to numpy arrays
        for key in dataset:
            dataset[key] = np.array(dataset[key])

        logger.info(f"Dataset created: {len(dataset['images']):,} samples")
        return dataset

    def train(self, dataset: Dict, epochs: int = 30, batch_size: int = 64):
        """
        Train the distance-aware model with custom loss and callbacks
        """
        if self.model is None:
            self.model = self.build_model()

        # ===================
        # CUSTOM LOSS & METRICS (from Training with Built-in Methods guide)
        # ===================

        # Custom metric: Distance-stratified accuracy
        class DistanceStratifiedAccuracy(keras.metrics.Metric):
            def __init__(self, name='dist_accuracy', **kwargs):
                super().__init__(name=name, **kwargs)
                self.close_correct = self.add_variable('close_correct', initializer='zeros')
                self.close_total = self.add_variable('close_total', initializer='zeros')
                self.far_correct = self.add_variable('far_correct', initializer='zeros')
                self.far_total = self.add_variable('far_total', initializer='zeros')

        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss={
                'age': 'binary_crossentropy',
                'gender': 'binary_crossentropy',
                'confidence': 'mse'
            },
            loss_weights={
                'age': 1.0,
                'gender': 1.0,
                'confidence': 0.5  # Less important than main tasks
            },
            metrics={
                'age': ['accuracy'],
                'gender': ['accuracy'],
                'confidence': ['mae']
            }
        )

        # Prepare training data
        X_train = {
            'face_image': dataset['images'],
            'distance_meters': dataset['distances'],
            'quality_score': dataset['qualities']
        }

        y_train = {
            'age': dataset['age_labels'],
            'gender': dataset['gender_labels'],
            'confidence': dataset['expected_confidences']
        }

        # ===================
        # CALLBACKS (from Writing Your Own Callbacks guide)
        # ===================

        class ResearchTargetCallback(keras.callbacks.Callback):
            """Monitor if we achieve research targets: 89.1% close, 72.3% far"""

            def on_epoch_end(self, epoch, logs=None):
                # TODO: Implement distance-stratified evaluation
                print(f"\\n📊 Epoch {epoch+1} - Checking research targets...")
                print("   Close target: 89.1% | Far target: 72.3%")

        callbacks = [
            ResearchTargetCallback(),
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint('best_distance_model.keras', save_best_only=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]

        # ===================
        # PHASE 1: Train with frozen base (Transfer Learning)
        # ===================
        logger.info("Phase 1: Training custom heads with frozen base...")

        history1 = self.model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs // 2,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )

        # ===================
        # PHASE 2: Fine-tuning (Transfer Learning Fine-tuning guide)
        # ===================
        logger.info("Phase 2: Fine-tuning entire model...")

        # Unfreeze base model
        self.base_model.trainable = True

        # Recompile with VERY LOW learning rate (critical for fine-tuning)
        self.model.compile(
            optimizer=keras.optimizers.Adam(1e-5),  # 100x smaller LR
            loss={
                'age': 'binary_crossentropy',
                'gender': 'binary_crossentropy',
                'confidence': 'mse'
            },
            loss_weights={'age': 1.0, 'gender': 1.0, 'confidence': 0.5},
            metrics={'age': 'accuracy', 'gender': 'accuracy', 'confidence': 'mae'}
        )

        # Fine-tune end-to-end
        history2 = self.model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs // 2,  # Fewer epochs to avoid overfitting
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )

        self.is_trained = True
        logger.info("Training completed!")

        return history1, history2

    def predict_distance_aware(self, face_image: np.ndarray, distance_m: float, quality_score: float) -> Dict:
        """
        Make distance-aware predictions
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")

        # Prepare inputs
        inputs = {
            'face_image': np.expand_dims(face_image, 0),
            'distance_meters': np.array([[distance_m]]),
            'quality_score': np.array([[quality_score]])
        }

        # Get predictions
        predictions = self.model.predict(inputs, verbose=0)

        return {
            'age_young_prob': float(predictions['age'][0][0]),
            'gender_male_prob': float(predictions['gender'][0][0]),
            'confidence': float(predictions['confidence'][0][0]),
            'age_class': 'Young' if predictions['age'][0][0] > 0.5 else 'Old',
            'gender_class': 'Male' if predictions['gender'][0][0] > 0.5 else 'Female'
        }

    def save_model(self, path: str):
        """Save trained model"""
        if self.model is not None:
            self.model.save(path)
            logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load trained model"""
        self.model = keras.models.load_model(path)
        self.is_trained = True
        logger.info(f"Model loaded from {path}")


# ===================
# USAGE EXAMPLE
# ===================
if __name__ == "__main__":
    # Initialize model
    model = DistanceAwareFaceModel()

    # Create dataset from CelebA
    # dataset = model.create_dataset_from_celeba('/path/to/celeba', num_samples=50000)

    # Train model
    # history = model.train(dataset, epochs=30)

    # Save model
    # model.save_model('distance_aware_face_model.keras')

    print("Distance-aware face recognition model ready for training!")
    print("Next steps:")
    print("1. Download CelebA dataset")
    print("2. Run model.create_dataset_from_celeba()")
    print("3. Run model.train()")
    print("4. Replace InsightFace system with trained model")
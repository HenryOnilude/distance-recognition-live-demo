"""
Training Script for Advanced Gender Ensemble
Includes distance-aware data augmentation and transfer learning
"""

import os
import logging
import numpy as np
import cv2
import tensorflow as tf

# Force CPU training to avoid M1 Mac Metal/GPU bug
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

from tensorflow import keras
from typing import Dict, List, Tuple
from advanced_gender_model import advanced_gender_model
from ml_training.celeba_data_loader import CelebALoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistanceAwareDataGenerator(keras.utils.Sequence):
    """
    Data generator with distance degradation simulation
    Applies augmentations to simulate different distance ranges
    """
    
    def __init__(self, images, labels, batch_size=32, shuffle=True, augment=True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.images))
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))
    
    def __getitem__(self, index):
        # Get batch indexes
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Generate batch
        X, y = self.__data_generation(indexes)
        
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, indexes):
        # Initialize batch
        X = np.empty((len(indexes), 224, 224, 3), dtype=np.float32)
        y = np.empty((len(indexes), 1), dtype=np.float32)
        
        for i, idx in enumerate(indexes):
            # Load image
            image = self.images[idx]
            
            # Apply distance-aware augmentation
            if self.augment:
                image = self.apply_distance_augmentation(image)
            
            # Normalize
            if image.max() > 1.0:
                image = image.astype(np.float32) / 255.0
            
            X[i] = image
            y[i] = self.labels[idx]
        
        return X, y
    
    def apply_distance_augmentation(self, image):
        """Apply random distance degradation"""
        # Random distance category
        distance_category = np.random.choice(['portrait', 'close', 'medium', 'far'])
        
        if distance_category == 'portrait':
            # Minimal degradation
            if np.random.random() > 0.5:
                brightness = np.random.uniform(0.9, 1.1)
                image = np.clip(image * brightness, 0, 255)
                
        elif distance_category == 'close':
            # Light degradation
            scale = np.random.uniform(0.85, 1.0)
            h, w = image.shape[:2]
            new_size = (int(w * scale), int(h * scale))
            image = cv2.resize(image, new_size)
            image = cv2.resize(image, (w, h))
            
            # Light blur
            if np.random.random() > 0.7:
                image = cv2.GaussianBlur(image, (3, 3), 0.5)
                
        elif distance_category == 'medium':
            # Moderate degradation
            scale = np.random.uniform(0.5, 0.8)
            h, w = image.shape[:2]
            new_size = (int(w * scale), int(h * scale))
            image = cv2.resize(image, new_size)
            image = cv2.GaussianBlur(image, (5, 5), 1.0)
            image = cv2.resize(image, (w, h))
            
        else:  # far
            # Heavy degradation
            scale = np.random.uniform(0.3, 0.5)
            h, w = image.shape[:2]
            new_size = (int(w * scale), int(h * scale))
            image = cv2.resize(image, new_size)
            image = cv2.GaussianBlur(image, (7, 7), 2.0)
            image = cv2.resize(image, (w, h))
            
            # Add noise
            noise = np.random.normal(0, 15, image.shape)
            image = np.clip(image + noise, 0, 255)
        
        # Additional augmentations
        if np.random.random() > 0.5:
            # Horizontal flip
            image = cv2.flip(image, 1)
        
        if np.random.random() > 0.7:
            # Slight rotation
            angle = np.random.uniform(-10, 10)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
        
        return image


def train_ensemble(use_real_data: bool = False, epochs: int = 30, batch_size: int = 32):
    """
    Train the advanced gender ensemble
    
    Args:
        use_real_data: Use real CelebA dataset (requires download)
        epochs: Number of training epochs per model
        batch_size: Batch size for training
    """
    logger.info("=" * 60)
    logger.info("🚀 TRAINING ADVANCED GENDER ENSEMBLE")
    logger.info("=" * 60)
    
    # Step 1: Load data
    if use_real_data:
        logger.info("Loading CelebA dataset...")
        loader = CelebALoader("./celeba_data")
        
        if not loader.verify_dataset():
            logger.error("❌ CelebA dataset not found. Please download first.")
            return
        
        # Load dataset (reduced from 50k to 10k to avoid memory issues)
        data = loader.create_sample_dataset(num_samples=10000)
    else:
        logger.info("Creating mock dataset for demonstration...")
        data = create_mock_celeba_data(num_samples=10000)
    
    # Extract images and labels
    images = data['images']
    genders = data['genders']  # 0=Female, 1=Male
    
    logger.info(f"Dataset: {len(images):,} images")
    logger.info(f"Gender distribution: Female={np.sum(genders==0)}, Male={np.sum(genders==1)}")
    
    # Step 2: Split data
    from sklearn.model_selection import train_test_split
    
    X_train, X_val, y_train, y_val = train_test_split(
        images, genders, test_size=0.2, random_state=42, stratify=genders
    )
    
    logger.info(f"Train: {len(X_train):,} | Validation: {len(X_val):,}")
    
    # Step 3: Create data generators
    train_gen = DistanceAwareDataGenerator(
        X_train, y_train, batch_size=batch_size, shuffle=True, augment=True
    )
    
    val_gen = DistanceAwareDataGenerator(
        X_val, y_val, batch_size=batch_size, shuffle=False, augment=False
    )
    
    # Step 4: Initialize ensemble
    logger.info("\nBuilding ensemble...")
    advanced_gender_model.create_ensemble()
    advanced_gender_model.compile_models()
    
    # Step 5: Train each model
    save_dir = "./gender_models"
    os.makedirs(save_dir, exist_ok=True)
    
    for model_name, model in advanced_gender_model.models.items():
        logger.info("\n" + "=" * 60)
        logger.info(f"Training {model_name.upper()} ({model.count_params():,} params)")
        logger.info("=" * 60)
        
        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                os.path.join(save_dir, f"{model_name}_gender_weights.h5"),
                save_best_only=True,
                save_weights_only=True,
                monitor='val_accuracy',
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        val_loss, val_acc, val_auc, val_prec, val_rec = model.evaluate(val_gen, verbose=0)
        
        logger.info(f"\n✅ {model_name} training complete:")
        logger.info(f"   Accuracy: {val_acc:.4f}")
        logger.info(f"   AUC: {val_auc:.4f}")
        logger.info(f"   Precision: {val_prec:.4f}")
        logger.info(f"   Recall: {val_rec:.4f}")
    
    # Step 6: Mark as trained
    advanced_gender_model.is_trained = True
    
    # Step 7: Test ensemble
    logger.info("\n" + "=" * 60)
    logger.info("🧪 TESTING ENSEMBLE")
    logger.info("=" * 60)
    
    # Test on validation samples
    test_samples = X_val[:10]
    test_labels = y_val[:10]
    
    correct = 0
    for i, (img, label) in enumerate(zip(test_samples, test_labels)):
        result = advanced_gender_model.predict_ensemble(img, distance_m=3.0, quality_score=0.8)
        predicted = 1 if result['gender'] == 'Male' else 0
        actual = int(label)
        
        is_correct = predicted == actual
        correct += is_correct
        
        logger.info(f"Sample {i+1}: Pred={result['gender']}, Actual={'Male' if actual==1 else 'Female'}, "
                   f"Conf={result['confidence']:.3f} {'✅' if is_correct else '❌'}")
    
    accuracy = correct / len(test_samples)
    logger.info(f"\nEnsemble Test Accuracy: {accuracy:.1%}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Models saved to: {save_dir}/")
    logger.info(f"Expected improvement: 75% → 90%+ gender accuracy")
    logger.info("\nTo enable in production:")
    logger.info("  export USE_ADVANCED_GENDER=true")
    logger.info("  python main.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Set random seeds
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Train ensemble
    train_ensemble(
        use_real_data=True,  # Set to True when CelebA is available
        epochs=20,  # Increase for better performance
        batch_size=32
    )
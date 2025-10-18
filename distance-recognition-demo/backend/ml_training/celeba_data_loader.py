"""
CelebA Dataset Loader for Distance-Aware Face Recognition
Downloads and prepares CelebA dataset for training
"""

import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import logging
from typing import Tuple, Dict
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CelebALoader:
    """
    Handles downloading and loading CelebA dataset
    """

    def __init__(self, data_dir: str = "celeba_data"):
        self.data_dir = data_dir
        
        # Handle both standard and nested (Kaggle) directory structures
        nested_dir = os.path.join(data_dir, "img_align_celeba", "img_align_celeba")
        standard_dir = os.path.join(data_dir, "img_align_celeba")
        
        if os.path.exists(nested_dir) and os.path.isdir(nested_dir):
            self.images_dir = nested_dir  # Kaggle format
        else:
            self.images_dir = standard_dir  # Standard format
        
        # Handle both .txt and .csv for attributes file
        # Prefer CSV format over TXT (more reliable parsing)
        csv_file = os.path.join(data_dir, "list_attr_celeba.csv")
        txt_file = os.path.join(data_dir, "list_attr_celeba.txt")
        
        if os.path.exists(csv_file):
            self.attr_file = csv_file
        elif os.path.exists(txt_file):
            self.attr_file = txt_file
        else:
            self.attr_file = csv_file  # Default to CSV

        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

    def download_dataset(self):
        """
        Download CelebA dataset
        Note: This is a large dataset (1.4GB images + annotations)
        """
        logger.info("Downloading CelebA dataset...")

        # CelebA download URLs (from official sources)
        urls = {
            'images': 'https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM',  # img_align_celeba.zip
            'attributes': 'https://drive.google.com/uc?id=0B7EVK8r0v71pblRyaVFSWGxPY0U'  # list_attr_celeba.txt
        }

        # Note: Google Drive direct download requires special handling
        # For now, provide instructions for manual download
        print("\n" + "="*60)
        print("MANUAL DOWNLOAD REQUIRED")
        print("="*60)
        print("Please download CelebA dataset manually:")
        print("1. Go to: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
        print("2. Download 'Align&Cropped Images' (img_align_celeba.zip)")
        print("3. Download 'Attribute Annotations' (list_attr_celeba.txt)")
        print(f"4. Extract images to: {self.images_dir}")
        print(f"5. Place attributes file at: {self.attr_file}")
        print("="*60)

        return False  # Manual download required

    def verify_dataset(self) -> bool:
        """
        Verify that CelebA dataset is properly downloaded and extracted
        """
        # Check if attributes file exists
        if not os.path.exists(self.attr_file):
            logger.error(f"Attributes file not found: {self.attr_file}")
            return False

        # Check if images directory exists and has files
        if not os.path.exists(self.images_dir):
            logger.error(f"Images directory not found: {self.images_dir}")
            return False

        # Count images
        image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg')]
        logger.info(f"Found {len(image_files):,} images in {self.images_dir}")

        if len(image_files) < 1000:  # Expect ~202K images
            logger.warning(f"Expected ~202,000 images, found only {len(image_files):,}")
            return False

        # Load and check attributes
        try:
            # Detect file format and load accordingly
            if self.attr_file.endswith('.csv'):
                # Kaggle CSV format (force comma delimiter)
                df = pd.read_csv(self.attr_file, sep=',', header=0, encoding='utf-8')
            else:
                # Original TXT format (whitespace separated)
                df = pd.read_csv(self.attr_file, sep='\s+', skiprows=1)
            
            logger.info(f"Loaded attributes for {len(df):,} images")
            logger.info(f"Available attributes: {list(df.columns)[:10]}...")  # Show first 10

            # Check required attributes (case-insensitive)
            required_attrs = ['Young', 'Male']
            for attr in required_attrs:
                if attr not in df.columns:
                    logger.error(f"Required attribute '{attr}' not found in dataset")
                    logger.error(f"Available columns: {list(df.columns)}")
                    return False

        except Exception as e:
            logger.error(f"Error loading attributes file: {e}")
            return False

        logger.info("✅ CelebA dataset verification passed!")
        return True

    def load_image(self, image_id: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Load and preprocess a single image from CelebA

        Args:
            image_id: Image filename (e.g., "000001.jpg")
            target_size: Target size for resizing

        Returns:
            Preprocessed image as numpy array (0-1 normalized)
        """
        image_path = os.path.join(self.images_dir, image_id)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Resize
        image = image.resize(target_size, Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize
        image_array = np.array(image).astype(np.float32) / 255.0

        return image_array

    def load_attributes(self) -> pd.DataFrame:
        """
        Load CelebA attributes as pandas DataFrame

        Returns:
            DataFrame with binary attributes (converted from -1/1 to 0/1)
        """
        # Detect file format and load accordingly
        if self.attr_file.endswith('.csv'):
            # Kaggle CSV format (force comma delimiter)
            df = pd.read_csv(self.attr_file, sep=',', header=0, encoding='utf-8')
            # First column is image filename, set it as index
            if 'image_id' in df.columns:
                df.set_index('image_id', inplace=True)
            else:
                # If first column isn't named 'image_id', it's still the image filename
                df.set_index(df.columns[0], inplace=True)
        else:
            # Original TXT format (whitespace separated)
            df = pd.read_csv(self.attr_file, sep='\s+', skiprows=1)

        # Convert attributes from -1/1 to 0/1 (skip image_id column)
        for col in df.columns:
            if col not in ['image_id', df.columns[0]]:  # Skip identifier columns
                if df[col].dtype in ['int64', 'int32']:
                    df[col] = (df[col] == 1).astype(int)

        logger.info(f"Loaded attributes for {len(df):,} images with {len(df.columns)} attributes")
        return df

    def create_sample_dataset(self, num_samples: int = 1000) -> Dict:
        """
        Create a small sample dataset for testing/development

        Args:
            num_samples: Number of samples to create

        Returns:
            Dictionary with sample data
        """
        if not self.verify_dataset():
            logger.error("Dataset verification failed. Cannot create sample.")
            return None

        logger.info(f"Creating sample dataset with {num_samples:,} samples...")

        # Load attributes
        df = self.load_attributes()

        # Sample random images
        sample_indices = np.random.choice(len(df), size=min(num_samples, len(df)), replace=False)
        sample_df = df.iloc[sample_indices].copy()

        images = []
        ages = []
        genders = []

        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Loading images"):
            try:
                # Load image
                image = self.load_image(row.name)  # row.name contains the image filename

                images.append(image)
                ages.append(row['Young'])
                genders.append(row['Male'])

            except Exception as e:
                logger.warning(f"Failed to load image {row.name}: {e}")
                continue

        dataset = {
            'images': np.array(images),
            'ages': np.array(ages),
            'genders': np.array(genders)
        }

        logger.info(f"Sample dataset created: {len(images):,} images")
        logger.info(f"Age distribution - Young: {np.sum(ages):,}, Old: {len(ages) - np.sum(ages):,}")
        logger.info(f"Gender distribution - Male: {np.sum(genders):,}, Female: {len(genders) - np.sum(genders):,}")

        return dataset


def create_mock_celeba_data(num_samples: int = 1000) -> Dict:
    """
    Create mock CelebA-like data for development/testing when real dataset not available

    Args:
        num_samples: Number of mock samples to generate

    Returns:
        Dictionary with mock data matching CelebA format
    """
    logger.info(f"Creating mock CelebA data: {num_samples:,} samples")

    # Generate mock face images (random noise with face-like structure)
    images = []
    for i in range(num_samples):
        # Create face-like pattern
        img = np.random.rand(224, 224, 3).astype(np.float32)

        # Add face-like oval shape (brighter center)
        y, x = np.ogrid[:224, :224]
        center_y, center_x = 112, 112
        mask = ((x - center_x) ** 2 / 80**2) + ((y - center_y) ** 2 / 100**2) <= 1
        img[mask] = img[mask] * 0.7 + 0.3  # Brighten face area

        images.append(img)

    # Generate realistic age/gender distribution
    # Based on CelebA statistics: ~75% young, ~60% female
    ages = np.random.choice([0, 1], size=num_samples, p=[0.25, 0.75])  # 75% young
    genders = np.random.choice([0, 1], size=num_samples, p=[0.6, 0.4])  # 60% female

    dataset = {
        'images': np.array(images),
        'ages': ages,
        'genders': genders
    }

    logger.info(f"Mock dataset created: {len(images):,} images")
    logger.info(f"Age distribution - Young: {np.sum(ages):,}, Old: {len(ages) - np.sum(ages):,}")
    logger.info(f"Gender distribution - Male: {np.sum(genders):,}, Female: {len(genders) - np.sum(genders):,}")

    return dataset


# ===================
# USAGE EXAMPLE
# ===================
if __name__ == "__main__":
    # Initialize loader
    loader = CelebALoader("./celeba_data")

    # Try to verify dataset (will show download instructions if needed)
    if loader.verify_dataset():
        print("✅ CelebA dataset found and verified!")

        # Create sample dataset
        sample_data = loader.create_sample_dataset(1000)

    else:
        print("❌ CelebA dataset not found.")
        print("Creating mock data for development...")

        # Create mock data for development
        mock_data = create_mock_celeba_data(1000)
        print("✅ Mock dataset created for development!")
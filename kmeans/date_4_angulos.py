import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops

def calculate_glcm_features(image_path):
    image = imread(image_path)

    # Si la imagen tiene 4 canales (RGBA), usa solo los 3 primeros (RGB)
    if image.shape[-1] == 4:
        image = image[..., :3]

    gray_image = rgb2gray(image)
    gray_image = (gray_image * 255).astype(np.uint8)
    distances = [1] # Operador D
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # Operador Q
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    features = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in properties:
        feat = graycoprops(glcm, prop).flatten()
        features.extend(feat)

    return features

def process_images(image_folder):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))]
    all_features = []
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        features = calculate_glcm_features(image_path)
        all_features.append(features)

    columns = [f'{prop}_{angle}' for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'] for angle in range(1)]
    df = pd.DataFrame(all_features, columns=columns)
    return df

image_folder = 'all'
df = process_images(image_folder)
df.to_csv('date_all.csv', index=False)
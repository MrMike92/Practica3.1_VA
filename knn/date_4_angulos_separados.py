import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops

def calculate_glcm_features(image_path, angle):
    image = imread(image_path)

    # Si la imagen tiene 4 canales (RGBA), usa solo los 3 primeros (RGB)
    if image.shape[-1] == 4:
        image = image[..., :3]

    gray_image = rgb2gray(image)
    gray_image = (gray_image * 255).astype(np.uint8)
    distances = [1]  # Define distances
    angles = [angle]  # Define angles

    glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

    features = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in properties:
        feat = graycoprops(glcm, prop).flatten()
        features.extend(feat)

    return features

def process_images(image_folder, angle):
    all_features = []
    for subdir, dirs, files in os.walk(image_folder):
        for folder in dirs:
            label = int(folder.replace('tema', ''))
            folder_path = os.path.join(subdir, folder)
            image_files = [f for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg'))]
            for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)
                features = calculate_glcm_features(image_path, angle)
                all_features.append([label] + features)

    columns = ['label'] + [f'{prop}' for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']]
    df = pd.DataFrame(all_features, columns=columns)
    return df

image_folder = 'imgs'  # Ruta de la carpeta principal que contiene las 40 carpetas de imágenes
angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

for angle in angles:
    df = process_images(image_folder, angle)
    angle_degrees = int(np.degrees(angle))
    df.to_csv(f'date_angle_{angle_degrees}.csv', index=False)

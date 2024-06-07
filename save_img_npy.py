import numpy as np
import matplotlib.pyplot as plt
import os

faces = np.load('olivetti_faces.npy')
targets = np.load('olivetti_faces_target.npy')
assert faces.shape == (400, 64, 64), "El archivo olivetti_faces.npy no tiene la forma esperada (400, 64, 64)"
assert targets.shape == (400,), "El archivo olivetti_faces_target.npy no tiene la forma esperada (400,)"

# Crear una carpeta principal para guardar las imágenes
output_dir = 'imgs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Crear una carpeta para todas las imágenes juntas
all_images_dir = os.path.join(output_dir, 'all')
if not os.path.exists(all_images_dir):
    os.makedirs(all_images_dir)

num_temas = len(np.unique(targets)) # Obtener el número de temas

# Guardar las imágenes individualmente en carpetas por tema y en la carpeta común
for tema in range(num_temas):
    # Crear una carpeta para el tema
    tema_dir = os.path.join(output_dir, f'tema{tema}')
    if not os.path.exists(tema_dir):
        os.makedirs(tema_dir)
    
    tema_indices = np.where(targets == tema)[0]
    for i, idx in enumerate(tema_indices):
        fig, ax = plt.subplots(figsize=(4, 4))  # Asegurarse de que la figura sea cuadrada
        ax.imshow(faces[idx], cmap='gray')  # Usar cmap='gray' para imágenes en escala de grises
        ax.axis('off')
        
        # Guardar cada imagen en la carpeta del tema
        tema_image_path = os.path.join(tema_dir, f'img{i + 1}.png')
        plt.savefig(tema_image_path, bbox_inches='tight', pad_inches=0)
        
        # Guardar cada imagen en la carpeta común
        common_image_path = os.path.join(all_images_dir, f'img{idx + 1}.png')
        plt.savefig(common_image_path, bbox_inches='tight', pad_inches=0)
        
        plt.close(fig)  # Cerrar la figura para liberar memoria
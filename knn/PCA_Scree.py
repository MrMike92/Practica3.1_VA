import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('date_angle_0.csv')

# Separar las etiquetas de las características
labels = df['label']
features = df.drop(columns=['label'])

# Estandarizar los datos
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# Paso 1: Calcular la media de cada característica y centrar los datos
mean_vector = np.mean(scaled_data, axis=0)
centered_data = scaled_data - mean_vector

# Paso 2: Calcular la matriz de covarianza
cov_matrix = np.cov(centered_data, rowvar=False)

# Paso 3: Calcular los valores y vectores propios de la matriz de covarianza
eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)

# Paso 4: Ordenar los vectores propios por la magnitud de los valores propios (en orden descendente)
sorted_index = np.argsort(eigen_values)[::-1]
sorted_eigenvalue = eigen_values[sorted_index]
sorted_eigenvectors = eigen_vectors[:, sorted_index]
explained_variance = sorted_eigenvalue / np.sum(sorted_eigenvalue) # Varianza explicada por cada componente

# Gráfica de Scree
plt.figure(figsize=(10, 7))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Gráfico de Sedimentación/Scree')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza Explicada')
plt.grid(True)
plt.show()

# Elegir el número óptimo de componentes basados en la gráfica de Scree
cumulative_variance = np.cumsum(explained_variance)
plt.figure(figsize=(10, 7))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.title('Varianza Acumulativa Explicada')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza Acumulativa Explicada')
plt.axhline(y=0.90, color='r', linestyle='-')  # Línea para el 90% de la varianza explicada
plt.grid(True)
plt.show()

# Paso 5: Proyectar los datos originales en los componentes principales seleccionados
n_components = np.argmax(cumulative_variance >= 0.90) + 1  # Número de componentes para explicar al menos el 90% de la varianza
principal_components = np.dot(centered_data, sorted_eigenvectors[:, :n_components])

# Crear un DataFrame con los componentes principales y las etiquetas
principal_df = pd.DataFrame(data=principal_components, columns=[f'Componente Principal {i+1}' for i in range(n_components)])
principal_df['label'] = labels

# Guardar el DataFrame con los componentes principales y las etiquetas en un archivo CSV
principal_df.to_csv('PrincipalComponents.csv', index=False)

# Visualizar los primeros dos componentes principales (si n_components >= 2)
if n_components >= 2:
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(principal_df.iloc[:, 0], principal_df.iloc[:, 1], c=principal_df['label'], cmap='viridis', edgecolor='k', s=40)
    plt.colorbar(scatter, label='Etiqueta')
    plt.title('Características de PCA de GLCM')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.grid()
    plt.show()
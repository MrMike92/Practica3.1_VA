import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Leer el archivo CSV (por ejemplo, para el ángulo 0)
df = pd.read_csv('date_angle_0.csv')

# Aplicar PCA
pca = PCA(n_components=2)  # Reducir a 2 componentes principales para visualización
principal_components = pca.fit_transform(df)

# Crear un DataFrame con los componentes principales
principal_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])

# Visualizar los componentes principales
plt.figure(figsize=(8, 6))
plt.scatter(principal_df['Principal Component 1'], principal_df['Principal Component 2'], edgecolor='k', s=40)
plt.title('PCA of GLCM Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.show()
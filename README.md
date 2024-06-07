# Clasificación de rostros con textura y PCA
Un clasificador KNN y K-Means para una base de datos de imágenes diferentes rostros.

## Instrucciones de uso.

- Clona este repositorio en tu máquina local.
- Asegure que el respositorio se haya descargado correctamente.
- Abre el programa que deseas ejecutar en tu entorno de desarrollo que soporte Python 3.11.2 64-bit.

## Diccionario
- ***Varianza Explicada***: Medida de cuánto de la varianza total en la variable dependiente se explica por las variables independientes.
</br> ![Varianza](https://github.com/MrMike92/Practica3.1_VA/assets/93272523/bd55f486-d0b5-486a-a3bb-11bdadd6ba4b)
- ***GLCM***: Método estadístico para analizar la textura que considera la relación espacial de los píxeles es la matriz de coocurrencia de nivel de gris, también denominada la matriz de dependencia espacial de escala de gris. Ejemplo:
</br> ![GLCM](https://github.com/MrMike92/Practica3.1_VA/assets/93272523/2a8122ed-1086-486c-9190-4726bfbbad2f)
- ***Contraste***: Mide las variaciones locales de la matriz de coocurrencia de nivel gris.
</br> ![Contraste](https://github.com/MrMike92/Practica3.1_VA/assets/93272523/0a0dbd6a-497e-448b-b6b1-92ededef6e84)
- ***Correlación***: Mide la probabilidad de aparición conjunta de los pares de píxeles especificados.
</br> ![Correlacion](https://github.com/MrMike92/Practica3.1_VA/assets/93272523/572d321d-0865-41b4-8d0c-66a9c20e4c5d)
- ***Energía***: Proporciona la suma de elementos al cuadrado de la GLCM. También conocido como uniformidad o el segundo momento angular.
</br> ![energia](https://github.com/MrMike92/Practica3.1_VA/assets/93272523/69ee52a5-1ec1-4190-abc8-8eebd77eca0e)
- ***Homogeneidad***: Mide la cercanía de la distribución de elementos de la GLCM a la diagonal de la GLCM.
</br> ![homogenidad](https://github.com/MrMike92/Practica3.1_VA/assets/93272523/d2370b05-e453-46e8-b968-4824b0f18b9a)
- ***Disimilitud***: describe cuán diferentes son los pares de elementos en términos de intensidad de grises dentro de una imagen.
</br> ![disimilitud](https://github.com/MrMike92/Practica3.1_VA/assets/93272523/3aac5848-343e-41fa-8e7c-5519604b7c35)
- ***ASM o Angular Second Moment (Segundo Momento Angular)***: Mide la homogeneidad de una imagen, donde los valores altos indican una textura homogénea, con poca variación en los valores de los píxeles.
  * Alto ASM: Indica una textura con alta uniformidad. Esto ocurre cuando hay poca variación en la intensidad de los píxeles dentro de una región.
  * Bajo ASM: Indica una textura con baja uniformidad. Esto ocurre cuando hay más variación en la intensidad de los píxeles.
</br> ![ASM](https://github.com/MrMike92/Practica3.1_VA/assets/93272523/fdb422c7-bf70-426d-8d1c-77c7641a2a91)


## Funcionamiento
Para este caso primero se tiene que pasar las imagenes a un archivo con valores separados por comas (archivo cvs) para procesar los datos y clasificar las imagenes si son riñones sanos o con piedras, por lo que se sugiere realizar los siguentes pasos para ejecutar el programa correctamente con cualquier otra base de datos (ya sea con imagenes a color o blanco y negro o escala de grises):

<br> 1. Descargar la base de datos, descomprimir el ZIP y mover los archivos *olivetti_faces.npy* y *olivetti_faces_target.npy* a la carpeta principal donde se encuentra el programa *save_img_py.py*.

> [!IMPORTANT]
> La base de datos de imangenes utilizada para este proyecto pertenece a su resprectivo creador, SAHIL YAGNIK.
> <br><br>Link de la base de datos de las imagenes: https://www.kaggle.com/datasets/sahilyagnik/olivetti-faces

<br> 2. Ejecutar el programa *save_img_npy.py*, entra a la carpeta **imgs** que genera el programa y mueve la capeta **all** a la carpeta kmeans y despues mover la carpeta **imgs** a **knn**. (Este paso es opcional puesto que las imagenes ya estan en este repositorio).

Si es en la carpeta **knn**, seguir los siguentes pasos:

<br> 3. Ejecutar *date_4_angulos_separados.py*.

<br> 4. Ejecutar *PCA_Scree.py*.

<br> 5. Ejecutar *KNN.py*.

Si es en la carpeta **kmeans**, seguir los siguentes pasos:

<br> 3. Ejecutar *date_4_angulos_separados.py*.

<br> 4. Ejecutar *PCA_Scree.py*. (Opcional porque solo se ve las graficas)

<br> 5. Ejecutar *kmeans.py*.

> [!WARNING]
> Si hay problemas en ejecutar el *save_img_npy.py*, debes de tener los archivos *olivetti_faces.npy* y *olivetti_faces_target.npy* en la misma carpeta.
> <br> ![image](https://github.com/MrMike92/Practica3.1_VA/assets/93272523/3bbf10b6-69e6-40d5-bfe3-d34ebbffe282)


> [!NOTE]
> Siéntete libre de personalizar el código, ya sea cambiando el valor de k (número de vecinos más cercanos) o cambiando la distancia euclidiana por otra distancia para ver si se pueda mejorar la precisión del clasificador.
> <br><br>Este código no tiene dependencias externas y debería funcionar con cualquier entorno Python 3.x.

Si deseas contribuir a este proyecto, puedes enviar solicitudes de extracción (pull requests) con mejoras o características adicionales y si tienes alguna pregunta o problema, puedes contactarme a través de mi perfil de GitHub MrMike92, en un futuro planeo abrir un correo para poder contactarme. 🐢

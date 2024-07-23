import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Cargar el archivo CSV proporcionado con los nuevos datos
file_path_new = 'filtered_data_final_v3.csv'
new_data = pd.read_csv(file_path_new)

# Seleccionar solo las columnas de interés (power y heart_rate)
new_data = new_data[['power', 'heart_rate']]

# Cargar los datos originales
file_path_original = 'filtered_data_final_v3.csv'
data_original = pd.read_csv(file_path_original)

# Seleccionar solo las columnas de interés (power y heart_rate)
data_original = data_original[['power', 'heart_rate']]

# Función para generar datos sintéticos y aplicar clustering
def generate_synthetic_data(data, power_variation=5, heart_rate_variation=2, n_clusters=4, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    synthetic_data = data.copy()
    synthetic_data['power'] += np.random.choice([-power_variation, power_variation], size=len(synthetic_data))
    synthetic_data['heart_rate'] += np.random.choice([-heart_rate_variation, heart_rate_variation], size=len(synthetic_data))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    synthetic_data['cluster'] = kmeans.fit_predict(synthetic_data)
    centroids = kmeans.cluster_centers_
    
    return synthetic_data, centroids

# Generar 10 juegos de datos sintéticos
synthetic_datasets = []
centroids_list = []

for i in range(10):
    synthetic_data, centroids = generate_synthetic_data(new_data, seed=42+i)
    synthetic_datasets.append(synthetic_data)
    centroids_list.append(centroids)

# Aplicar clustering a los datos originales
kmeans_original = KMeans(n_clusters=4, random_state=42)
data_original['cluster'] = kmeans_original.fit_predict(data_original)
centroids_original = kmeans_original.cluster_centers_

# Calcular la curva polinomial para los datos originales
centroids_sorted_original = centroids_original[np.argsort(centroids_original[:, 0])]
polynomial_coefficients_original = np.polyfit(centroids_sorted_original[:, 0], centroids_sorted_original[:, 1], 3)
polynomial_curve_original = np.poly1d(polynomial_coefficients_original)
power_range_original = np.linspace(min(centroids_sorted_original[:, 0]), max(centroids_sorted_original[:, 0]), 100)

# Crear una figura y un eje
fig, ax = plt.subplots(figsize=(12, 8))

# Colores para los clusters
colors = ['blue', 'green', 'orange', 'purple']

# Graficar los datos originales
for i in range(4):
    cluster_data = data_original[data_original['cluster'] == i]
    ax.scatter(cluster_data['power'], cluster_data['heart_rate'], c=colors[i], alpha=0.3, label=f'Cluster {i} (Original)')

# Graficar los centroides de los datos originales
ax.scatter(centroids_original[:, 0], centroids_original[:, 1], c='red', marker='X', s=200, label='Centroids (Original)')

# Graficar la curva polinomial para los datos originales
ax.plot(power_range_original, polynomial_curve_original(power_range_original), color='black', linestyle='--', label='Polynomial Curve (Original)')

# Graficar los datos sintéticos
for j, (synthetic_data, centroids) in enumerate(zip(synthetic_datasets, centroids_list)):
    for i in range(4):
        cluster_data = synthetic_data[synthetic_data['cluster'] == i]
        ax.scatter(cluster_data['power'], cluster_data['heart_rate'], c=colors[i], alpha=0.3, edgecolor='k', label=f'Cluster {i} (Synthetic {j+1})' if j == 0 else "")
    ax.scatter(centroids[:, 0], centroids[:, 1], c='yellow', marker='X', s=100, label=f'Centroids (Synthetic {j+1})' if j == 0 else "")

    # Calcular y graficar la curva polinomial que pasa por los centroides de los datos sintéticos
    centroids_sorted = centroids[np.argsort(centroids[:, 0])]
    polynomial_coefficients = np.polyfit(centroids_sorted[:, 0], centroids_sorted[:, 1], 3)
    polynomial_curve = np.poly1d(polynomial_coefficients)
    power_range = np.linspace(min(centroids_sorted[:, 0]), max(centroids_sorted[:, 0]), 100)
    ax.plot(power_range, polynomial_curve(power_range), color='orange', linestyle='--', label=f'Polynomial Curve (Synthetic {j+1})' if j == 0 else "")

# Añadir título y etiquetas
ax.set_title('Original and Synthetic Data: Power vs Heart Rate with Clusters, Centroids, and Polynomial Curves')
ax.set_xlabel('Power')
ax.set_ylabel('Heart Rate')

# Añadir leyenda
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Ajustar el espacio para la leyenda
plt.tight_layout(rect=[0, 0, 0.75, 1])

# Mostrar la gráfica
plt.show()

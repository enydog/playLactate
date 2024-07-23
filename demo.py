import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Cargar el archivo CSV proporcionado
file_path = 'LTtest.csv'
data = pd.read_csv(file_path)

# Filtrar los datos excluyendo Power = 0
filtered_data_exclude_power_eq_0 = data[data['Power'] != 0]



# Seleccionar columnas relevantes para clustering
X = final_filtered_data[['Power', 'HRM']]

# Aplicar KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

# Obtener los centroides de los clusters
centroids = kmeans.cluster_centers_

# Añadir etiquetas de clusters a los datos
final_filtered_data['Cluster'] = kmeans.labels_

# Fit a single polynomial curve that passes through each centroid
centroid_powers = centroids[:, 0]
centroid_hrms = centroids[:, 1]

# Fit a polynomial curve of degree 2 through the centroids
poly_coeffs = np.polyfit(centroid_powers, centroid_hrms, 2)
x_range = np.linspace(final_filtered_data['Power'].min(), final_filtered_data['Power'].max(), 500)
poly_curve = np.polyval(poly_coeffs, x_range)

# Renombrar las columnas del archivo de lactato para facilitar el manejo
file_path_lactate = 'potencia_lactato.csv'
data_lactate = pd.read_csv(file_path_lactate)
data_lactate.rename(columns={'Potencia (W)': 'Power', 'Lactato (mmol/L)': 'Lactate'}, inplace=True)

# Plot Power vs HRM with clusters and centroids marked as crosses and add lactate data on secondary y-axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Scatter plot for each cluster on the primary y-axis
for cluster in range(3):
    cluster_data = final_filtered_data[final_filtered_data['Cluster'] == cluster]
    ax1.scatter(cluster_data['Power'], cluster_data['HRM'], alpha=0.5, label=f'Cluster {cluster}')

# Plot the centroids with a cross
ax1.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='+', label='Centroids')

# Plot the polynomial curve passing through the centroids
ax1.plot(x_range, poly_curve, color='black', linewidth=2, label='Polynomial Curve')

ax1.set_xlabel('Power')
ax1.set_ylabel('HRM')
ax1.legend(loc='upper left')
ax1.grid(True)

# Create secondary y-axis for lactate data
ax2 = ax1.twinx()
ax2.plot(data_lactate['Power'], data_lactate['Lactate'], color='blue', label='Lactate')
ax2.set_ylabel('Lactate', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Add legend for lactate data
fig.tight_layout()
fig.legend(loc='upper right')

plt.title('Polynomial Clusters -Centroids and Lactate curve')
plt.show()

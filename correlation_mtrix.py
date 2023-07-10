from osgeo import gdal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression

# Open the raster file and return a Dataset object
raster_file = 'correlation_matrix/MODIS_indices_20230526.tif'
raster_dataset = gdal.Open(raster_file)

# Access the raster data using the ReadAsArray() method
raster_data = raster_dataset.ReadAsArray()

# Get the dimensions of the raster data
num_bands, height, width = raster_data.shape

# Reshape the raster data to a 2D format, where each index is a separate column
raster_2d = raster_data.reshape(num_bands, -1)

# Convert the 2D array to a Pandas DataFrame
raster_df = pd.DataFrame(raster_2d.transpose())

# Remove NaN and infinite values
raster_df = raster_df.replace([np.inf, -np.inf], np.nan).dropna()

# Compute the correlation matrix of the raster data
corr_matrix = raster_df.corr()

# Apply feature selection using SelectKBest with f_regression
k = 5  # Number of features to select
feature_selector = SelectKBest(score_func=f_regression, k=k)
selected_features = feature_selector.fit_transform(raster_df, raster_df.iloc[:, -1])

# Get the indices of the selected features with lowest scores
selected_indices = np.argsort(feature_selector.scores_)[:k]

# Get the names of all features
feature_names = raster_df.columns

# Write the selected indices to a text file
output_file = 'selected_indices.txt'
with open(output_file, 'w') as file:
    for index in selected_indices:
        file.write(f'{feature_names[index]}\n')

print(f'Selected indices are written to {output_file}.')

# Plot the correlation matrix as a heatmap
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

# Add correlation coefficient numbers to the plot
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='black')

# Add all features to the graph
ax.set_xticks(np.arange(len(feature_names)))
ax.set_yticks(np.arange(len(feature_names)))
ax.set_xticklabels(feature_names, rotation=45, ha='right')
ax.set_yticklabels(feature_names)

# Mark the selected features by changing their color
for index in selected_indices:
    ax.get_xticklabels()[index].set_color('red')
    ax.get_yticklabels()[index].set_color('red')

# Add legend on the side
cbar = fig.colorbar(im, ax=ax, orientation='vertical', label='Correlation')
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Correlation', rotation=270)

# Add a title
ax.set_title("Correlation Matrix")

# Save the figure
output_image = 'correlation_heatmap.png'
plt.savefig(output_image)

# Show the plot
plt.show()

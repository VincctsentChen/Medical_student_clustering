# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:26:47 2023

@author: f006q7g
"""
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
import numpy as np

# Set the OMP_NUM_THREADS environment variable
os.environ['OMP_NUM_THREADS'] = '10'

def calculate_inertia(data, max_clusters):
    inertia = []
    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
    return inertia

# Custom function for Gap Statistic
def gap_statistic(X, B=10, max_clusters=20):
    gaps = []
    for cluster_count in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=cluster_count, random_state=42)
        kmeans.fit(X)
        inertia = kmeans.inertia_
        BWkbs = np.zeros(B)
        for i in range(B):
            X_random = np.random.random_sample(size=X.shape)
            kmeans_random = KMeans(n_clusters=cluster_count, random_state=42)
            kmeans_random.fit(X_random)
            BWkbs[i] = kmeans_random.inertia_
        gaps.append((1/B) * sum(np.log(BWkbs) - np.log(inertia)))
    return gaps

# Change the directory
path = r'C:\Users\f006q7g\OneDrive - Dartmouth College\RCode\ChatGPT\HMS'
os.chdir(path)

# Load the CSV file
filename = 'HMS 2016-2021combined_file_PHDonly.csv' # chnage to MD student file pr pHD students  HMS 2016-2021combined_file_MDonly.csv HMS 2016-2021combined_file_PHDonly.csv
data = pd.read_csv(filename, low_memory=False)



# Extend the columns_of_interest list
columns_of_interest = ['diener1', 'diener2', 'diener3', 'diener4', 'diener5', 'diener6', 'diener7', 'diener8', 'phq9_1', 'phq9_2', 'phq9_3', 'phq9_4', 'phq9_5', 'phq9_6', 'phq9_7', 'phq9_8', 'phq9_9', 'gad7_1', 'gad7_2', 'gad7_3', 'gad7_4', 'gad7_5', 'gad7_6', 'gad7_7']

# Update data_selected
data_selected = data[columns_of_interest]

# Remove rows with missing values
data_selected = data_selected.dropna()

# Scale the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_selected)

# Calculate inertia for different numbers of clusters
max_clusters = 15
inertia = calculate_inertia(data_scaled, max_clusters)

# Plot the inertia
plt.figure()
plt.plot(range(1, max_clusters + 1), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()


# calculate silhouette scores for different numbers of clusters
def calculate_silhouette(data, max_clusters):
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    return silhouette_scores

# Calculate silhouette scores for different numbers of clusters
max_clusters = 15
silhouette_scores = calculate_silhouette(data_scaled, max_clusters)

# Plot the silhouette scores
plt.figure()
plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.show()

# Choose the optimal number of clusters based on the highest Silhouette Score
n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2



# Choose the optimal number of clusters based on the Elbow Method plot
n_clusters = 3

# Perform cluster analysis with multiple initializations
kmeans = KMeans(n_clusters=n_clusters, n_init=20, init='random', random_state=0)
data.loc[data_selected.index, 'cluster'] = kmeans.fit_predict(data_scaled)

# Select the best solution based on within-cluster sum of squares
lowest_inertia = kmeans.inertia_
best_labels = kmeans.labels_
for i in range(1, 20):
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, init='random', random_state=i)
    data.loc[data_selected.index, 'cluster'] = kmeans.fit_predict(data_scaled)
    if kmeans.inertia_ < lowest_inertia:
        lowest_inertia = kmeans.inertia_
        best_labels = kmeans.labels_

# Assign the best cluster labels to the data
data.loc[data_selected.index, 'cluster'] = best_labels

# Create a pair plot
data_selected['cluster'] = data.loc[data_selected.index, 'cluster']
sns.pairplot(data_selected, hue='cluster', diag_kind='hist', corner=True)
plt.show()

# Update 3D plots to accommodate new columns
# Due to the increased number of columns, it's difficult to visualize all the dimensions simultaneously.
# You may need to choose the most relevant dimensions or use dimensionality reduction techniques like PCA or t-SNE.

# Save the results to a new CSV file
output_filename = 'clustered_data.csv'
data.to_csv(output_filename, index=False)

# Print the number of rows used
print(f"Number of rows used: {data_selected.shape[0]}")

import joblib

# Save the kmeans model to disk
kmeans_filename = 'kmeans_model.joblib'
joblib.dump(kmeans, kmeans_filename)

# Save the scaler to disk
scaler_filename = 'scaler.joblib'
joblib.dump(scaler, scaler_filename)


## Predict cluster membership

# def predict_cluster_membership(individual_values, kmeans_model, scaler):
#     # Assuming 'individual_values' is a list with values in the same order as the input features
#     individual_df = pd.DataFrame([individual_values])
    
#     # Scale the individual's values using the same scaler used for the original dataset
#     individual_scaled = scaler.transform(individual_df)
    
#     # Predict the cluster membership
#     cluster_membership = kmeans_model.predict(individual_scaled)
    
#     return cluster_membership[0]

# # Example usage
# # Replace 'individual_values' with actual values for the individual in the same order as the input features
# individual_values = [7, 7, 7, 7, 7, 7, 7, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# predicted_cluster = predict_cluster_membership(individual_values, kmeans, scaler)
# print(f"The predicted cluster m,embership for the individual is: {predicted_cluster}")

# Calculate inertia for different numbers of clusters
max_clusters = 15
inertia = calculate_inertia(data_scaled, max_clusters)

# Calculate gap statistic for different numbers of clusters
gaps = gap_statistic(data_scaled, B=10, max_clusters=max_clusters)

# Plot the inertia
plt.figure()
plt.plot(range(1, max_clusters + 1), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Plot the gap statistic
plt.figure()
plt.plot(range(1, max_clusters + 1), gaps, marker='o') 
plt.xlabel('Number of clusters')
plt.ylabel('Gap Statistic')
plt.title('Gap Statistic Method')
plt.show()



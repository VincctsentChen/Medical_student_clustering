# -*- coding: utf-8 -*-
"""
Created on Sat May  6 11:57:28 2023

@author: f006q7g
"""

import pandas as pd
import os
import matplotlib.pyplot as plt

# Change the directory
path = r'C:\Users\f006q7g\OneDrive - Dartmouth College\RCode\ChatGPT\HMS'
os.chdir(path)

# Load the clustered data
filename = 'clustered_data.csv'
data = pd.read_csv(filename, low_memory=False)

# Clean up data
data = data.dropna(subset=['sex_birth'])
data = data[data['sex_birth'].isin([1, 2])]
data['sex_birth'] = data['sex_birth'].map({1: 'Female', 2: 'Male'})

# Define the columns of interest
columns_of_interest = ['sex_birth']

# Calculate overall gender counts
total = len(data)
male_data = data.loc[data['sex_birth'] == 'Male']
female_data = data.loc[data['sex_birth'] == 'Female']

# Loop through each column of interest
for col in columns_of_interest:
    # Loop through each cluster
    for cluster in data['cluster'].unique():
        # Select data for current cluster and column
        cluster_data = data.loc[data['cluster'] == cluster, [col, 'sex_birth']]

        # Calculate percentages for each gender in the current cluster
        male_data_cluster = male_data.loc[male_data['cluster'] == cluster]
        female_data_cluster = female_data.loc[female_data['cluster'] == cluster]

        percentage_males_in_cluster = len(male_data_cluster) / len(male_data) * 100
        percentage_females_in_cluster = len(female_data_cluster) / len(female_data) * 100

        # Create plot
        fig, ax = plt.subplots()
        bars = ax.bar(['Female', 'Male'], [percentage_females_in_cluster, percentage_males_in_cluster], color=['red', 'blue'])
        ax.set_title(f"Cluster {cluster} - {col} vs Sex Birth")
        ax.set_ylabel('% of Total')

        # Add percentage values to the plot
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}%", ha='center', va='bottom')

        plt.tight_layout()
        plt.show()


# Summary statistics for numeric columns
summary = data.describe()

# Number of unique clusters
unique_clusters = data['cluster'].nunique()

# Gender distribution
gender_counts = data['sex_birth'].value_counts()

# Save the summary to a CSV file
summary.to_csv('summary_numeric.csv')

# Save unique_clusters and gender_counts to a CSV file
summary_counts = pd.DataFrame({
    'Unique Clusters': [unique_clusters],
    'Female Count': [gender_counts['Female']],
    'Male Count': [gender_counts['Male']]
})
summary_counts.to_csv('summary_counts.csv', index=False)



# create a csv file with the number and percentage of subjects in each cluster

# Initialize a dictionary to store the number of members and percentage in each cluster
cluster_info = {}

# Calculate the total number of members
total_members = len(data)

# Loop through each cluster
for cluster in data['cluster'].unique():
    # Count the number of members in the current cluster
    num_members = len(data.loc[data['cluster'] == cluster])

    # Calculate the percentage of members in the current cluster
    percentage_members = (num_members / total_members) * 100

    # Add the results to the dictionary
    cluster_info[f'Cluster {cluster}'] = {'Number of Members': num_members, 'Percentage': percentage_members}

# Convert the dictionary to a DataFrame
cluster_info_df = pd.DataFrame.from_dict(cluster_info, orient='index')

# Save the DataFrame to a CSV file
cluster_info_df.to_csv('cluster_members.csv')

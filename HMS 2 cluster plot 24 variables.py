import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import kruskal, sem
from scipy.stats import median_abs_deviation

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem

def bootstrap_median_ci(data, n_bootstraps=1000, ci=95):
    bootstrapped_medians = []
    for _ in range(n_bootstraps):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_medians.append(np.median(bootstrap_sample))
    return np.percentile(bootstrapped_medians, [(100-ci)/2, (100+ci)/2])


# Change the directory
path = r'C:\Users\f006q7g\OneDrive - Dartmouth College\RCode\ChatGPT\HMS'
os.chdir(path)


def load_and_extract_columns(csv_file, columns):
    df = pd.read_csv(csv_file, low_memory=False)
    extracted_df = df[columns]
    return extracted_df


def remove_missing_values(dataframe):
    cleaned_df = dataframe.dropna()
    return cleaned_df


def plot_median_values_per_cluster_with_error_bars(dataframe, cluster_col, columns_to_plot):
    grouped_medians = dataframe.groupby(cluster_col)[columns_to_plot].median()
    error_bars = []
    for col in columns_to_plot:
        col_error_bars = []
        for cluster in np.unique(dataframe[cluster_col]):
            cluster_data = dataframe[dataframe[cluster_col] == cluster][col]
            ci_low, ci_high = bootstrap_median_ci(cluster_data)
            col_error_bars.append((ci_high - ci_low) / 2)
        error_bars.append(col_error_bars)
    error_bars = np.array(error_bars)

    # Create labels for the legend
    legend_labels = ['Purpose','Social Support','Engagement','Making Others Happy','Competence','Good Person','Optimism','Respect','Anhedonia','Depressed mood','Sleep','Fatigue','Appetite','Self-criticism','Concentration','Psychomotor','Self-harm/suicidality','Nervousness','Rumination','Worrying','Tension','Agitation','Irritability','Apprehension']

    # Ensure the legend_labels length matches the number of columns to plot
    if len(columns_to_plot) != len(legend_labels):
        raise ValueError("Length of labels provided for the legend doesn't match the number of columns to plot.")

    # Replace dataframe column names with the provided labels
    rename_dict = dict(zip(columns_to_plot, legend_labels))
    grouped_medians = grouped_medians.rename(columns=rename_dict)

    ax = grouped_medians.plot(kind='bar', yerr=error_bars, figsize=(10, 6), rot=0, capsize=4)
    plt.xlabel('Cluster')
    plt.ylabel('Median Values')
    plt.title('Median Values per Cluster with Error Bars (95% CI)')
    plt.legend(title='Variables', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.show()



def plot_mean_values_per_cluster_with_std_error_bars(dataframe, cluster_col, columns_to_plot):
    grouped_means = dataframe.groupby(cluster_col)[columns_to_plot].mean()
    grouped_stds = dataframe.groupby(cluster_col)[columns_to_plot].std()

    ax = grouped_means.plot(kind='bar', yerr=grouped_stds, figsize=(10, 6), rot=0, capsize=4, legend=False)
    plt.xlabel('Cluster')
    plt.ylabel('Mean Values')
    plt.title('Mean Values per Cluster with Error Bars (Standard Deviation)')
    plt.legend(title='Variables', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.show()


def plot_mean_values_per_cluster_with_sem_error_bars(dataframe, cluster_col, columns_to_plot):
    grouped_means = dataframe.groupby(cluster_col)[columns_to_plot].mean()
    grouped_sems = dataframe.groupby(cluster_col)[columns_to_plot].agg(sem)

    # Create labels for the legend
    legend_labels = ['Purpose','Social Support','Engagement','Making Others Happy','Competence','Good Person','Optimism','Respect','Anhedonia','Depressed mood','Sleep','Fatigue','Appetite','Self-criticism','Concentration','Psychomotor','Self-harm/suicidality','Nervousness','Rumination','Worrying','Tension','Agitation','Irritability','Apprehension']

    # Ensure the legend_labels length matches the number of columns to plot
    if len(columns_to_plot) != len(legend_labels):
        raise ValueError("Length of labels provided for the legend doesn't match the number of columns to plot.")

    # Replace dataframe column names with the provided labels
    rename_dict = dict(zip(columns_to_plot, legend_labels))
    grouped_means = grouped_means.rename(columns=rename_dict)
    grouped_sems = grouped_sems.rename(columns=rename_dict)

    # Create a colormap
    cm = plt.get_cmap('gray')

    ax = grouped_means.plot(kind='bar', yerr=grouped_sems, figsize=(10, 6), rot=0, capsize=4, colormap=cm)
    plt.xlabel('Cluster')
    plt.ylabel('Mean Values')
    plt.title('Mean Values per Cluster with Error Bars (Standard Error of the Mean)')
    plt.legend(title='Variables', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.show()





if __name__ == '__main__':
    csv_file = 'clustered_data_MD.csv'
    columns_to_extract = ['diener1', 'diener2', 'diener3', 'diener4', 'diener5', 'diener6', 'diener7', 'diener8', 'phq9_1', 'phq9_2', 'phq9_3', 'phq9_4', 'phq9_5', 'phq9_6', 'phq9_7', 'phq9_8', 'phq9_9', 'gad7_1', 'gad7_2', 'gad7_3', 'gad7_4', 'gad7_5', 'gad7_6', 'gad7_7','cluster']
    extracted_data = load_and_extract_columns(csv_file, columns_to_extract)
    cleaned_data = remove_missing_values(extracted_data)

    plot_median_values_per_cluster_with_error_bars(cleaned_data, 'cluster', ['diener1', 'diener2', 'diener3', 'diener4', 'diener5', 'diener6', 'diener7', 'diener8', 'phq9_1', 'phq9_2', 'phq9_3', 'phq9_4', 'phq9_5', 'phq9_6', 'phq9_7', 'phq9_8', 'phq9_9', 'gad7_1', 'gad7_2', 'gad7_3', 'gad7_4', 'gad7_5', 'gad7_6', 'gad7_7'])
    plot_mean_values_per_cluster_with_sem_error_bars(cleaned_data, 'cluster', ['diener1', 'diener2', 'diener3', 'diener4', 'diener5', 'diener6', 'diener7', 'diener8', 'phq9_1', 'phq9_2', 'phq9_3', 'phq9_4', 'phq9_5', 'phq9_6', 'phq9_7', 'phq9_8', 'phq9_9', 'gad7_1', 'gad7_2', 'gad7_3', 'gad7_4', 'gad7_5', 'gad7_6', 'gad7_7'])
    plot_median_values_per_cluster_with_sem_error_bars(cleaned_data, 'cluster', ['diener1', 'diener2', 'diener3', 'diener4', 'diener5', 'diener6', 'diener7', 'diener8', 'phq9_1', 'phq9_2', 'phq9_3', 'phq9_4', 'phq9_5', 'phq9_6', 'phq9_7', 'phq9_8', 'phq9_9', 'gad7_1', 'gad7_2', 'gad7_3', 'gad7_4', 'gad7_5', 'gad7_6', 'gad7_7'])



# Perform Kruskal-Wallis test for each variable
variables = ['diener1', 'diener2', 'diener3', 'diener4', 'diener5', 'diener6', 'diener7', 'diener8', 'phq9_1', 'phq9_2', 'phq9_3', 'phq9_4', 'phq9_5', 'phq9_6', 'phq9_7', 'phq9_8', 'phq9_9', 'gad7_1', 'gad7_2', 'gad7_3', 'gad7_4', 'gad7_5', 'gad7_6', 'gad7_7']
clusters = np.unique(cleaned_data['cluster'])

for variable in variables:
    # Extract data for
    variables = ['diener1', 'diener2', 'diener3', 'diener4', 'diener5', 'diener6', 'diener7', 'diener8', 'phq9_1', 'phq9_2', 'phq9_3', 'phq9_4', 'phq9_5', 'phq9_6', 'phq9_7', 'phq9_8', 'phq9_9', 'gad7_1', 'gad7_2', 'gad7_3', 'gad7_4', 'gad7_5', 'gad7_6', 'gad7_7']
    clusters = np.unique(cleaned_data['cluster'])

for variable in variables:
# Extract data for each cluster
    data_by_cluster = [cleaned_data[cleaned_data['cluster'] == cluster][variable] for cluster in clusters]
    
    
    # Perform Kruskal-Wallis test
kruskal_wallis_statistic, p_value = kruskal(*data_by_cluster)

print(f"Kruskal-Wallis test for {variable}: H = {kruskal_wallis_statistic:.4f}, p = {p_value:.4f}")

    
    
def plot_median_values_per_cluster_with_error_bars_all_dimensions(dataframe, cluster_col):
    dimensions_to_plot = dataframe.columns.drop([cluster_col])
    grouped_medians = dataframe.groupby([cluster_col])[dimensions_to_plot].median()
    error_bars = []
    for col in dimensions_to_plot:
        col_error_bars = []
        for cluster in np.unique(dataframe[cluster_col]):
            cluster_data = dataframe[dataframe[cluster_col] == cluster][col]
            ci_low, ci_high = bootstrap_median_ci(cluster_data)
            col_error_bars.append((ci_high - ci_low) / 2)
        error_bars.append(col_error_bars)
    error_bars = np.array(error_bars)

    n_clusters = len(np.unique(dataframe[cluster_col]))
    n_cols = 3
    n_rows = np.ceil(len(dimensions_to_plot) / n_cols).astype(int)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for i, (dim, ax) in enumerate(zip(dimensions_to_plot, axes)):
        grouped_medians.loc[:, dim].plot(kind='bar', yerr=error_bars[i], ax=ax, rot=0, capsize=4)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Median Values')
        ax.set_title('Median Values for Dimension {}'.format(dim))

    plt.tight_layout()
    plt.show()

    
    plot_median_values_per_cluster_with_error_bars_all_dimensions(cleaned_data, 'cluster')

# create output file with summary statistics for each variable by cluster
def save_summary_statistics_to_csv(summary_statistics, filename):
    summary_statistics.to_csv(filename)

if __name__ == '__main__':

    
    def generate_extensive_summary_statistics_by_cluster(dataframe, cluster_col, variables):
        grouped = dataframe.groupby(cluster_col)[variables]
        summary_statistics = grouped.agg(['count', 'mean', 'median', 'std', 'min', 'max'])
        return summary_statistics

    # Generate extensive summary statistics for each variable by cluster
    extensive_summary_statistics = generate_extensive_summary_statistics_by_cluster(cleaned_data, 'cluster', variables)

    # Save the summary statistics to a CSV file
    save_summary_statistics_to_csv(extensive_summary_statistics, 'extensive_summary_statistics_by_cluster_MD.csv')


    
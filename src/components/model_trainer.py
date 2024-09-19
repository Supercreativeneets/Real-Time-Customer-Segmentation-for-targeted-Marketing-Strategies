import os
import sys

from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.cluster import KMeans
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.components.data_transformation import DataTransformation

@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts", "model.pkl")
    elbow_plot_path: str = os.path.join("artifacts", "elbow_plot.png")
    Fig3D_cluster_path: str = os.path.join("artifacts", "3D_cluster_plot.html")
    cluster_labels_path: str = os.path.join('artifacts', 'cluster_labels.csv')

class ModelTrainer:
    def __init__(self):
        self.model_config = ModelTrainerConfig()

    def find_optimal_clusters(self, data, max_k=10):
        wcss = []  # Within-cluster sum of squares

        for i in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)

        plt.figure(figsize=(8, 4))
        plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--')
        plt.title('Elbow Method For Optimal k')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.savefig(self.model_config.elbow_plot_path)
        plt.close()

        # Finding the elbow point (optimal number of clusters)
        deltas = np.diff(wcss)
        second_deltas = np.diff(deltas)
        optimal_k = np.argmax(second_deltas) + 2  # Adding 2 because np.diff reduces the array size by 1 each time

        return optimal_k

    def initiate_model_training(self, data_array):
        try:
            logging.info("Finding the optimal number of clusters")
            optimal_clusters = self.find_optimal_clusters(data_array)
            logging.info(f"Optimal number of clusters found: {optimal_clusters}")

            logging.info("Training K-Means model with optimal number of clusters")
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
            kmeans.fit(data_array)

            logging.info(f"Saving the trained model to {self.model_config.trained_model_path}")
            save_object(file_path=self.model_config.trained_model_path, obj=kmeans)

            x, y, z = data_array[:,:3].T
            fig = px.scatter_3d(x = x, 
                                y = y, 
                                z = z,
                    color = kmeans.labels_)
            
            logging.info(f"Saving the 3D figure to {self.model_config.Fig3D_cluster_path}")
            fig.write_html(self.model_config.Fig3D_cluster_path)

            
            # Save the cluster labels to a CSV file
            logging.info(f"Saving the Cluster labels to {self.model_config.cluster_labels_path}")
            np.savetxt(self.model_config.cluster_labels_path, kmeans.labels_, delimiter=',', fmt='%i')
                        
            return kmeans.labels_
        
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    pca_data_path = os.path.join('artifacts', 'pca_data.csv')
    pca_data = np.loadtxt(pca_data_path, delimiter=',',dtype=float)

    model_trainer = ModelTrainer()
    labels = model_trainer.initiate_model_training(pca_data)
    print("Clustering labels:", labels)

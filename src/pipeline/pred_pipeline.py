import sys
import os
import pandas as pd
import numpy as np
import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go
from dataclasses import dataclass
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

@dataclass
class Pred_Pipeline:
    def __init__(self):
        pass

    def predict(self, df):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            final_processor_path = os.path.join('artifacts', 'final_processor.pkl')
            pca_obj_path = os.path.join('artifacts', 'PCA_fitted.pkl')

            model = load_object(file_path=model_path)
            pre_processor = load_object(file_path=preprocessor_path)
            final_processor = load_object(file_path=final_processor_path)
            pca_obj = load_object(file_path=pca_obj_path)

            pre_process_data = pre_processor.transform(df)
            final_process_data = final_processor.transform(pre_process_data)
            pca_fitted_data = pca_obj.transform(final_process_data) 
                        
            # Predict the cluster
            predicted_cluster = model.predict(pca_fitted_data)

            return (
                pca_fitted_data,
                predicted_cluster
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        
            
    def plot_cluster_highlight(self,df):
        try:
            # Load the PCA-transformed data for all points
            pca_data_path = os.path.join('artifacts', 'pca_data.csv')
            pca_data_df = pd.read_csv(pca_data_path, header=None)
            pca_data = pca_data_df.values

            # The data contains the PCA-transformed points and their respective cluster labels
            cluster_labels_path = os.path.join('artifacts', 'cluster_labels.csv')
            cluster_labels_df = pd.read_csv(cluster_labels_path, header=None)
            cluster_labels = cluster_labels_df.values.flatten()  # Flatten to 1D array

            # Extract the first three PCA components
            x, y, z = pca_data[:,:3].T

            # Predict the cluster for the given input data
            pca_fitted_data, predicted_cluster = self.predict(df)
            predicted_cluster_label = predicted_cluster[0]

            # Create a color array where all clusters are gray by default
            colors = ['gray'] * len(cluster_labels)

            # Highlight the predicted cluster with a unique color
            for i, cluster_label in enumerate(cluster_labels):
                if cluster_label == predicted_cluster_label:
                    colors[i] = 'blue'  # Highlight the predicted cluster in blue

            # Create a 3D scatter plot for all points
            fig = go.Figure()

            # Add all points with color coding
            fig.add_trace(
                go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(size=5, color=colors),  # Apply the colors array
                name='Clusters'
                )
            )
            
            # Highlight the predicted point in red with a diamond symbol
            fig.add_trace(
                go.Scatter3d(
                    x=[pca_fitted_data[0][0]],  # x-coordinate of the predicted point
                    y=[pca_fitted_data[0][1]],  # y-coordinate
                    z=[pca_fitted_data[0][2]],  # z-coordinate
                    mode='markers',
                    marker=dict(size=10, color='red', symbol='diamond'),
                    name=f'Predicted Cluster: {predicted_cluster}'
                )
            )

            # Save the updated plot as an HTML file
            output_html_path = os.path.join('artifacts', 'cluster_prediction.html')
            pio.write_html(fig, file=output_html_path)
            logging.info(f'3D scatter plot with predicted cluster saved at {output_html_path}')

        except Exception as e:
            raise CustomException(e, sys)

class InputData:
    def __init__(self,
                 CUST_ID: str,  
                 BALANCE: float,
                 BALANCE_FREQUENCY: float,
                 PURCHASES: float,
                 ONEOFF_PURCHASES: float,
                 INSTALLMENTS_PURCHASES: float,
                 CASH_ADVANCE: float,
                 PURCHASES_FREQUENCY: float,
                 ONEOFF_PURCHASES_FREQUENCY: float,
                 PURCHASES_INSTALLMENTS_FREQUENCY: float,
                 CASH_ADVANCE_FREQUENCY: float,
                 CASH_ADVANCE_TRX: int,
                 PURCHASES_TRX: int,
                 CREDIT_LIMIT: float,
                 PAYMENTS: float,
                 MINIMUM_PAYMENTS: float,
                 PRC_FULL_PAYMENT: float,
                 TENURE: int):
        self.CUST_ID = CUST_ID
        self.BALANCE = BALANCE
        self.BALANCE_FREQUENCY = BALANCE_FREQUENCY
        self.PURCHASES = PURCHASES
        self.ONEOFF_PURCHASES = ONEOFF_PURCHASES
        self.INSTALLMENTS_PURCHASES = INSTALLMENTS_PURCHASES
        self.CASH_ADVANCE = CASH_ADVANCE
        self.PURCHASES_FREQUENCY = PURCHASES_FREQUENCY
        self.ONEOFF_PURCHASES_FREQUENCY = ONEOFF_PURCHASES_FREQUENCY
        self.PURCHASES_INSTALLMENTS_FREQUENCY = PURCHASES_INSTALLMENTS_FREQUENCY
        self.CASH_ADVANCE_FREQUENCY = CASH_ADVANCE_FREQUENCY
        self.CASH_ADVANCE_TRX = CASH_ADVANCE_TRX
        self.PURCHASES_TRX = PURCHASES_TRX
        self.CREDIT_LIMIT = CREDIT_LIMIT
        self.PAYMENTS = PAYMENTS
        self.MINIMUM_PAYMENTS = MINIMUM_PAYMENTS
        self.PRC_FULL_PAYMENT = PRC_FULL_PAYMENT
        self.TENURE = TENURE

    def transform_data_as_dataframe(self):
        try:
            user_input_data_dict = {
                "CUST_ID": [self.CUST_ID],
                "BALANCE": [self.BALANCE],
                "BALANCE_FREQUENCY": [self.BALANCE_FREQUENCY],
                "PURCHASES": [self.PURCHASES],
                "ONEOFF_PURCHASES": [self.ONEOFF_PURCHASES],
                "INSTALLMENTS_PURCHASES": [self.INSTALLMENTS_PURCHASES],
                "CASH_ADVANCE": [self.CASH_ADVANCE],
                "PURCHASES_FREQUENCY": [self.PURCHASES_FREQUENCY],
                "ONEOFF_PURCHASES_FREQUENCY": [self.ONEOFF_PURCHASES_FREQUENCY],
                "PURCHASES_INSTALLMENTS_FREQUENCY": [self.PURCHASES_INSTALLMENTS_FREQUENCY],
                "CASH_ADVANCE_FREQUENCY": [self.CASH_ADVANCE_FREQUENCY],
                "CASH_ADVANCE_TRX": [self.CASH_ADVANCE_TRX],
                "PURCHASES_TRX": [self.PURCHASES_TRX],
                "CREDIT_LIMIT": [self.CREDIT_LIMIT],
                "PAYMENTS": [self.PAYMENTS],
                "MINIMUM_PAYMENTS": [self.MINIMUM_PAYMENTS],
                "PRC_FULL_PAYMENT": [self.PRC_FULL_PAYMENT],
                "TENURE": [self.TENURE]
            }
            logging.info("Starting transformation...")
            logging.info(f"Data: {user_input_data_dict}")

            return pd.DataFrame(user_input_data_dict)
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    # Example usage:
    input_data = InputData(
        CUST_ID='C10029',
        BALANCE=7152.864372,
        BALANCE_FREQUENCY=1.0,
        PURCHASES=387.05,
        ONEOFF_PURCHASES=204.55,
        INSTALLMENTS_PURCHASES=182.5,
        CASH_ADVANCE=2236.145259,
        PURCHASES_FREQUENCY=0.666667,
        ONEOFF_PURCHASES_FREQUENCY=0.166667,
        PURCHASES_INSTALLMENTS_FREQUENCY=0.416667,
        CASH_ADVANCE_FREQUENCY=0.833333,
        CASH_ADVANCE_TRX=16,
        PURCHASES_TRX=8,
        CREDIT_LIMIT=10500.0,
        PAYMENTS=1601.448347,
        MINIMUM_PAYMENTS=1648.851345,
        PRC_FULL_PAYMENT=0.0,
        TENURE=12
    )

    pred_pipeline = Pred_Pipeline()
    input_df = input_data.transform_data_as_dataframe()
    _, predicted_cluster = pred_pipeline.predict(input_df)
    print("Cluster Label:", predicted_cluster)
    pred_pipeline.plot_cluster_highlight(input_df)

import os
import streamlit as st
import pandas as pd
import numpy as np
from src.utils import load_object
from src.exception import CustomException
from src.logger import logging
from src.pipeline.pred_pipeline import Pred_Pipeline, InputData
import plotly.io as pio

# Recommendations dictionary
CLUSTER_RECOMMENDATIONS = {
    0: "This group is mostly doing installments purchases, but monthly average purchases is low. They are paying dues on time and maintaining good credit score. We can give them rewards points on purchases using credit card. This will encourage them to make more purchases. Also, consider increasing their credit limit.",
    1: "They have poor credit score and taking only cash on advance. We can target them by providing less interest rate on purchase and cash advances. Through reducing interest we will put ourselves in a position over our competitors since we have our users' best interest. Again we hope to improve retention and frequency with this strategy.",
    2: "They are potential target customers who are paying dues and doing purchases and maintaining comparatively good credit score. The best marketing strategy for this cluster is to give credit points for every time they make a transaction using credit card. These points would incentivize the customers to be loyal to the bank and reduce churn rates.",
    3: "This group is using the card for just one-off purchases and some cash advance transactions. We can give them a higher One-Off credit limit to keep them using One-Off Purchases.",
}

# Streamlit app definition
def main():
    st.title("Customer Segmentation Prediction")
    st.markdown(
        """
        This application allows you to input customer financial data and predicts the cluster they belong to 
        using a pre-trained machine learning pipeline.
        """
    )

    # Input form for user data
    st.sidebar.header("Input Features")
    
    with st.sidebar.form(key="input_form"):
        # User inputs
        CUST_ID = st.text_input("Customer ID")
        BALANCE = st.slider("Balance", min_value=0.0, max_value=50000.0, value=5000.0, step=1.0)
        BALANCE_FREQUENCY = st.slider("Balance Frequency", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        PURCHASES = st.slider("Purchases", min_value=0.0, max_value=50000.0, value=1000.0, step=1.0)
        ONEOFF_PURCHASES = st.slider("One-off Purchases", min_value=0.0, max_value=50000.0, value=500.0, step=1.0)
        INSTALLMENTS_PURCHASES = st.slider("Installment Purchases", min_value=0.0, max_value=25000.0, value=300.0, step=1.0)
        CASH_ADVANCE = st.slider("Cash Advance", min_value=0.0, max_value=50000.0, value=2000.0, step=1.0)
        PURCHASES_FREQUENCY = st.slider("Purchases Frequency", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        ONEOFF_PURCHASES_FREQUENCY = st.slider("One-off Purchases Frequency", min_value=0.0, max_value=1.0, value=  0.2, step=0.01)
        PURCHASES_INSTALLMENTS_FREQUENCY = st.slider("Purchases Installments Frequency", min_value=0.0, max_value=1.0, value=0.4, step=0.01)
        CASH_ADVANCE_FREQUENCY = st.slider("Cash Advance Frequency", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        CASH_ADVANCE_TRX = st.slider("Cash Advance Transactions", min_value=0, max_value=500, value=10, step=1)
        PURCHASES_TRX = st.slider("Purchases Transactions", min_value=0, max_value=500, value=20, step=1)
        CREDIT_LIMIT = st.slider("Credit Limit", min_value=50.0, max_value=50000.0, value=10000.0, step=50.0)
        PAYMENTS = st.slider("Payments", min_value=0.0, max_value=100000.0, value=5000.0, step=25.0)
        MINIMUM_PAYMENTS = st.slider("Minimum Payments", min_value=0.0, max_value=100000.0, value=1000.0, step=50.0)
        PRC_FULL_PAYMENT = st.slider("Percentage Full Payment", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        TENURE = st.slider("Tenure", min_value=0, max_value=12, value=6, step=1)

        # Submit button
        submitted = st.form_submit_button("Predict Cluster")

    # Prediction and visualization
    if submitted:
        try:
            # Initialize InputData with user inputs
            input_data = InputData(
                CUST_ID=CUST_ID,
                BALANCE=BALANCE,
                BALANCE_FREQUENCY=BALANCE_FREQUENCY,
                PURCHASES=PURCHASES,
                ONEOFF_PURCHASES=ONEOFF_PURCHASES,
                INSTALLMENTS_PURCHASES=INSTALLMENTS_PURCHASES,
                CASH_ADVANCE=CASH_ADVANCE,
                PURCHASES_FREQUENCY=PURCHASES_FREQUENCY,
                ONEOFF_PURCHASES_FREQUENCY=ONEOFF_PURCHASES_FREQUENCY,
                PURCHASES_INSTALLMENTS_FREQUENCY=PURCHASES_INSTALLMENTS_FREQUENCY,
                CASH_ADVANCE_FREQUENCY=CASH_ADVANCE_FREQUENCY,
                CASH_ADVANCE_TRX=CASH_ADVANCE_TRX,
                PURCHASES_TRX=PURCHASES_TRX,
                CREDIT_LIMIT=CREDIT_LIMIT,
                PAYMENTS=PAYMENTS,
                MINIMUM_PAYMENTS=MINIMUM_PAYMENTS,
                PRC_FULL_PAYMENT=PRC_FULL_PAYMENT,
                TENURE=TENURE,
            )

            # Transform input data to DataFrame
            input_df = input_data.transform_data_as_dataframe()

            # Initialize Prediction Pipeline
            pred_pipeline = Pred_Pipeline()

            # Make prediction
            pca_fitted_data, predicted_cluster = pred_pipeline.predict(input_df)

            # Display the predicted cluster
            cluster = predicted_cluster[0]
            st.success(f"The customer belongs to Cluster: {cluster}")

            # Display the corresponding recommendation
            recommendation = CLUSTER_RECOMMENDATIONS.get(cluster, "No recommendation available for this cluster.")
            st.info(f"Recommendation: {recommendation}")

            # Generate and display the cluster visualization
            pca_data_path = "artifacts/cluster_prediction.html"
            pred_pipeline.plot_cluster_highlight(input_df)
            with open(pca_data_path, "r",encoding="utf-8") as f:  # (Explicitly use UTF-8 encoding)
                cluster_html = f.read()
            st.components.v1.html(cluster_html, height=900)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
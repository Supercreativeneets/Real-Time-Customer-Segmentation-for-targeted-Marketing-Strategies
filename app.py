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
        BALANCE = st.number_input("Balance", value=0.0, format="%.2f")
        BALANCE_FREQUENCY = st.number_input("Balance Frequency", value=0.0, format="%.2f")
        PURCHASES = st.number_input("Purchases", value=0.0, format="%.2f")
        ONEOFF_PURCHASES = st.number_input("One-off Purchases", value=0.0, format="%.2f")
        INSTALLMENTS_PURCHASES = st.number_input("Installment Purchases", value=0.0, format="%.2f")
        CASH_ADVANCE = st.number_input("Cash Advance", value=0.0, format="%.2f")
        PURCHASES_FREQUENCY = st.number_input("Purchases Frequency", value=0.0, format="%.2f")
        ONEOFF_PURCHASES_FREQUENCY = st.number_input("One-off Purchases Frequency", value=0.0, format="%.2f")
        PURCHASES_INSTALLMENTS_FREQUENCY = st.number_input("Purchases Installments Frequency", value=0.0, format="%.2f")
        CASH_ADVANCE_FREQUENCY = st.number_input("Cash Advance Frequency", value=0.0, format="%.2f")
        CASH_ADVANCE_TRX = st.number_input("Cash Advance Transactions", value=0, format="%d")
        PURCHASES_TRX = st.number_input("Purchases Transactions", value=0, format="%d")
        CREDIT_LIMIT = st.number_input("Credit Limit", value=0.0, format="%.2f")
        PAYMENTS = st.number_input("Payments", value=0.0, format="%.2f")
        MINIMUM_PAYMENTS = st.number_input("Minimum Payments", value=0.0, format="%.2f")
        PRC_FULL_PAYMENT = st.number_input("Percentage Full Payment", value=0.0, format="%.2f")
        TENURE = st.number_input("Tenure", value=0, format="%d")

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
            with open(pca_data_path, "r", encoding="utf-8") as f:  # Explicitly use UTF-8 encoding)
                cluster_html = f.read()
            st.components.v1.html(cluster_html, height=700)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
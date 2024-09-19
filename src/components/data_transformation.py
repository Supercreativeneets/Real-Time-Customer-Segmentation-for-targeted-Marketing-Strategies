import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from dataclasses import dataclass

@dataclass
class PreProcessingConfig:
    pre_processor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class FinalProcessingConfig:
    final_processing_obj_file_path=os.path.join('artifacts',"final_processor.pkl")

class PCAConfig:
    PCA_obj_file_path = os.path.join('artifacts', 'PCA_fitted.pkl')

class PCAdata:
    pca_data_path = os.path.join('artifacts', 'pca_data.csv')
    
class MinimumPaymentsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.payments_mean_ = None

    def fit(self, X, y=None):
        self.payments_mean_ = np.mean(X['PAYMENTS'])
        return self

    def transform(self, X):
        X = X.copy()
        minpay = X['MINIMUM_PAYMENTS'].copy()
        payments_mean = self.payments_mean_

        i = 0
        for payments, minpayments in zip(X['PAYMENTS'], X['MINIMUM_PAYMENTS'].isna()):
            if (payments == 0) and (minpayments == True):
                minpay.iloc[i] = 0
            elif (0 < payments < payments_mean) and (minpayments == True): 
                minpay.iloc[i] = payments
            elif minpayments == True: 
                minpay.iloc[i] = payments_mean
            i += 1
        
        X['MINIMUM_PAYMENTS'] = minpay.copy()
        return X
    
class MonthlyAvgPurchaseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, purchase_col='PURCHASES', tenure_col='TENURE'):
        self.purchase_col = purchase_col
        self.tenure_col = tenure_col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        # Avoid division by zero by replacing zero values in TENURE with NaN
        X[self.tenure_col].replace(0, float('nan'), inplace=True)
        # Compute Monthly_avg_purchase
        X['Monthly_avg_purchase'] = X[self.purchase_col] / X[self.tenure_col]
        # Replace NaN values with 0 in the resulting column
        X['Monthly_avg_purchase'].fillna(0, inplace=True)
        return X
    
class MonthlyCashAdvanceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cash_advance_col='CASH_ADVANCE', tenure_col='TENURE'):
        self.cash_advance_col = cash_advance_col
        self.tenure_col = tenure_col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        # Avoid division by zero by replacing zero values in TENURE with NaN
        X[self.tenure_col].replace(0, float('nan'), inplace=True)
        # Compute Monthly_cash_advance
        X['Monthly_cash_advance'] = X[self.cash_advance_col] / X[self.tenure_col]
        # Replace NaN values with 0 in the resulting column
        X['Monthly_cash_advance'].fillna(0, inplace=True)
        return X

class CreditLimitTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.median_credit_limit = None

    def fit(self, X, y=None):
        # Calculate the median of the CREDIT_LIMIT column
        self.median_credit_limit = X['CREDIT_LIMIT'].median()
        return self

    def transform(self, X):
        # Copy the dataframe to avoid changing the original
        X = X.copy()
        
        # Fill missing values in CREDIT_LIMIT with the median
        X['CREDIT_LIMIT'].fillna(self.median_credit_limit, inplace=True)
        
        # Create the limit_usage column
        X['limit_usage'] = X['BALANCE'] / X['CREDIT_LIMIT']
        
        return X

class PurchaseTypeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, oneoff_purchases_col='ONEOFF_PURCHASES', installments_purchases_col='INSTALLMENTS_PURCHASES'):
        self.oneoff_purchases_col = oneoff_purchases_col
        self.installments_purchases_col = installments_purchases_col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        # Apply the purchase type logic
        X['Purchase_Type'] = X.apply(self._purchase_type, axis=1)
        return X

    def _purchase_type(self, row):
        if (row[self.oneoff_purchases_col] > 0) and (row[self.installments_purchases_col] == 0):
            return 'ONE_OFF'
        if (row[self.oneoff_purchases_col] == 0) and (row[self.installments_purchases_col] > 0):
            return 'INSTALLMENTS'
        if (row[self.oneoff_purchases_col] > 0) and (row[self.installments_purchases_col] > 0):
            return 'BOTH'
        if (row[self.oneoff_purchases_col] == 0) and (row[self.installments_purchases_col] == 0):
            return 'NONE'
        
class OneHotEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column = 'Purchase_Type'):
        self.column = column
        self.categories_ = None

    def fit(self, X, y=None):
        # Find unique categories in the specified column
        self.categories_ = X[self.column].unique()
        return self

    def transform(self, X):
        # Create a copy of the dataframe to avoid modifying the original dataframe
        X = X.copy()
        # Perform one-hot encoding
        for category in self.categories_:
            X[f'{category}'] = (X[self.column] == category).astype(int)
        # Drop the original column
        X.drop(columns=[self.column], inplace=True)
        return X

num_cols = [
        'BALANCE_FREQUENCY', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES',
        'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
        'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
        'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'PAYMENTS', 'MINIMUM_PAYMENTS',
        'PRC_FULL_PAYMENT', 'Monthly_avg_purchase', 'Monthly_cash_advance',
        'limit_usage'
    ]

cat_cols = ['Purchase_Type']

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols):
        self.num_cols = num_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Apply logarithmic transformation to all numerical columns
        X = X.copy()
        numerical_cols = num_cols
        X[numerical_cols] = X[numerical_cols].applymap(lambda x: np.log(x + 1))
        return X

columns_to_drop = ['CUST_ID']

class DropIrrelevantColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X.drop(columns=self.columns_to_drop, axis=1, inplace=True)
        return X
    
    
class DataTransformation:
    def __init__(self):
        self.pre_processing_config = PreProcessingConfig()
        self.final_processing_config = FinalProcessingConfig()
        self.pca_config = PCAConfig()
        self.PCAdata = PCAdata()

    def get_pre_processor_object(self):
        try:
            ''' This function is responsible for transforming data '''
            # Create the preprocessor with ColumnTransformer
            preprocessor = Pipeline([
                ('min_payments', MinimumPaymentsTransformer()),
                ('monthly_avg_purchase', MonthlyAvgPurchaseTransformer()),
                ('monthly_cash_advance', MonthlyCashAdvanceTransformer()),
                ('limit_usage', CreditLimitTransformer()),
                ('purchase_type', PurchaseTypeTransformer()),
                ('drop_CUST_ID', DropIrrelevantColumnsTransformer(columns_to_drop))
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def get_final_processor_object(self):
        try:

            num_pipeline = Pipeline(steps = [
                                    ('log_transformer', LogTransformer(num_cols)),
                                    ('minmax_scaler',MinMaxScaler())
                                            ])

            cat_pipeline = Pipeline(steps = [
                                    ('one_hot_encoder', OneHotEncoderTransformer(column='Purchase_Type'))
                                            ])

            final_processor = ColumnTransformer([
                ("num_pipeline",num_pipeline,num_cols),
                ("cat_pipelines",cat_pipeline,cat_cols)
                ])
            
            return final_processor
                
        except Exception as e:
            raise CustomException(e, sys)
        
           
    def initiate_transformation(self, data_path):
        try: 
            df = pd.read_csv(data_path)
            logging.info("Data read successfully")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_pre_processor_object()

            logging.info(
                f"Applying preprocessing object on the dataframe."
            )
            df_pre = preprocessing_obj.fit_transform(df)

            logging.info("Obtaining finalprocessing object")
            final_processing_obj=self.get_final_processor_object()

            logging.info(
                f"Applying finalprocessing object on the dataframe."
            )
            df_transformed = final_processing_obj.fit_transform(df_pre)

            pca_obj = PCA(n_components=5)
            
            logging.info(
                f"Applying PCA object on the dataframe."
            )
            reduced_df = pca_obj.fit_transform(df_transformed)

            np.savetxt(self.PCAdata.pca_data_path, reduced_df, delimiter=',', fmt='%f')
            logging.info(f"Saved pca data for all points.")

            save_object(
                file_path=self.pre_processing_config.pre_processor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.final_processing_config.final_processing_obj_file_path,
                obj=final_processing_obj
            )
            logging.info(f"Saved finalprocessing object.")
            
            save_object(
                file_path=self.pca_config.PCA_obj_file_path,
                obj=pca_obj
            )
            logging.info(f"Saved PCA object.")

            return (
                df_pre, 
                df_transformed, 
                reduced_df,
                self.pre_processing_config.pre_processor_obj_file_path, 
                self.final_processing_config.final_processing_obj_file_path,
                self.pca_config.PCA_obj_file_path,
                self.PCAdata.pca_data_path
                )
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    data_path = '/Users/neeti/Documents/Customer-Segmentation-Kubernetes-CI-CD/notebook/data/Customer Data.csv'
    data_transformation = DataTransformation()
    df_pre, df_transformed,reduced_df,_,_,_,_ = data_transformation.initiate_transformation(data_path)

    # Assuming you have a ModelTrainer class for clustering
    # from src.components.model_trainer import ModelTrainer
    # model = ModelTrainer()
    # print(model.initiate_model_training(transformed_data))

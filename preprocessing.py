"""
Data Preprocessing Module for Loan Prediction
Handles missing values, encoding, feature engineering, and scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


class LoanDataPreprocessor:
    """Preprocessor for loan prediction data"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_data(self, filepath):
        """Load data from CSV file"""
        return pd.read_csv(filepath)
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        df = df.copy()
        
        # Fill categorical missing values with mode
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 
                           'Credit_History', 'Loan_Amount_Term']
        for col in categorical_cols:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Fill numerical missing values with median
        numerical_cols = ['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome']
        for col in numerical_cols:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        return df
    
    def feature_engineering(self, df):
        """Create new features from existing ones"""
        df = df.copy()
        
        # Total Income
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        
        # Loan Amount to Income Ratio
        df['LoanAmountToIncome'] = df['LoanAmount'] / (df['TotalIncome'] + 1)
        
        # Income per dependent
        df['Dependents_num'] = df['Dependents'].replace('3+', '3').astype(float)
        df['IncomePerDependent'] = df['TotalIncome'] / (df['Dependents_num'] + 1)
        
        # Log transformations for skewed features
        df['LoanAmount_log'] = np.log1p(df['LoanAmount'])
        df['TotalIncome_log'] = np.log1p(df['TotalIncome'])
        
        return df
    
    def encode_categorical(self, df, fit=True):
        """Encode categorical variables"""
        df = df.copy()
        
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 
                           'Self_Employed', 'Property_Area']
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def prepare_features(self, df, target_col='Loan_Status', fit=True):
        """Prepare features for modeling"""
        df = df.copy()
        
        # Drop Loan_ID if present
        if 'Loan_ID' in df.columns:
            df = df.drop('Loan_ID', axis=1)
        
        # Separate target if present
        y = None
        if target_col in df.columns:
            y = df[target_col].map({'Y': 1, 'N': 0})
            df = df.drop(target_col, axis=1)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Feature engineering
        df = self.feature_engineering(df)
        
        # Encode categorical variables
        df = self.encode_categorical(df, fit=fit)
        
        # Select final features
        feature_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                       'Credit_History', 'Property_Area', 'LoanAmount_log', 
                       'TotalIncome_log', 'Loan_Amount_Term', 'LoanAmountToIncome']
        
        X = df[feature_cols]
        
        if fit:
            self.feature_columns = feature_cols
        
        return X, y
    
    def fit_transform(self, df, target_col='Loan_Status'):
        """Fit preprocessor and transform data"""
        X, y = self.prepare_features(df, target_col, fit=True)
        return X, y
    
    def transform(self, df):
        """Transform new data using fitted preprocessor"""
        X, _ = self.prepare_features(df, target_col=None, fit=False)
        return X
    
    def save(self, filepath):
        """Save preprocessor to file"""
        joblib.dump(self, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load preprocessor from file"""
        return joblib.load(filepath)


def main():
    """Main function to test preprocessing"""
    # Load data
    preprocessor = LoanDataPreprocessor()
    df = preprocessor.load_data('data/raw/train.csv')
    
    print("Original data shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    
    # Preprocess
    X, y = preprocessor.fit_transform(df)
    
    print("\nProcessed features shape:", X.shape)
    print("\nFeature columns:", X.columns.tolist())
    print("\nTarget distribution:")
    print(y.value_counts())
    
    # Save preprocessor
    os.makedirs('models', exist_ok=True)
    preprocessor.save('models/preprocessor.joblib')
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    processed_df = X.copy()
    processed_df['Loan_Status'] = y
    processed_df.to_csv('data/processed/train_processed.csv', index=False)
    print("\nProcessed data saved to data/processed/train_processed.csv")


if __name__ == "__main__":
    main()

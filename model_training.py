"""
Model Training and Evaluation Module
Trains multiple classifiers and selects the best model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report, 
                             confusion_matrix)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class LoanPredictionModel:
    """Train and evaluate loan prediction models"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        
    def initialize_models(self):
        """Initialize different classification models"""
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'XGBoost': XGBClassifier(random_state=42, n_estimators=100, eval_metric='logloss'),
            'SVM': SVC(random_state=42, probability=True)
        }
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train all models and evaluate performance"""
        print("=" * 70)
        print("MODEL TRAINING AND EVALUATION")
        print("=" * 70)
        
        for name, model in self.models.items():
            print(f"\n{'=' * 70}")
            print(f"Training {name}...")
            print(f"{'=' * 70}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'model': model
            }
            
            # Print results
            print(f"\nTest Set Performance:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  ROC-AUC:   {roc_auc:.4f}")
            print(f"\nCross-Validation (5-fold):")
            print(f"  Mean Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")
            
        # Select best model based on F1-score
        self.best_model_name = max(self.results, key=lambda x: self.results[x]['f1_score'])
        self.best_model = self.results[self.best_model_name]['model']
        
        print("\n" + "=" * 70)
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"F1-Score: {self.results[self.best_model_name]['f1_score']:.4f}")
        print("=" * 70)
        
        return self.best_model
    
    def print_comparison(self):
        """Print comparison of all models"""
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)
        
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[m]['accuracy'] for m in self.results],
            'Precision': [self.results[m]['precision'] for m in self.results],
            'Recall': [self.results[m]['recall'] for m in self.results],
            'F1-Score': [self.results[m]['f1_score'] for m in self.results],
            'ROC-AUC': [self.results[m]['roc_auc'] for m in self.results],
        })
        
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        print("\n", comparison_df.to_string(index=False))
        
    def detailed_evaluation(self, X_test, y_test):
        """Print detailed evaluation of best model"""
        print("\n" + "=" * 70)
        print(f"DETAILED EVALUATION - {self.best_model_name}")
        print("=" * 70)
        
        y_pred = self.best_model.predict(X_test)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"                Predicted")
        print(f"                Rejected  Approved")
        print(f"Actual Rejected    {cm[0][0]:4d}      {cm[0][1]:4d}")
        print(f"       Approved    {cm[1][0]:4d}      {cm[1][1]:4d}")
        
    def save_model(self, filepath):
        """Save the best model"""
        joblib.dump(self.best_model, filepath)
        print(f"\nBest model ({self.best_model_name}) saved to {filepath}")
        
    @staticmethod
    def load_model(filepath):
        """Load a saved model"""
        return joblib.load(filepath)


def main():
    """Main function to train models"""
    # Load processed data
    print("Loading processed data...")
    df = pd.read_csv('data/processed/train_processed.csv')
    
    # Separate features and target
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Initialize and train models
    model_trainer = LoanPredictionModel()
    model_trainer.initialize_models()
    best_model = model_trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Print comparison
    model_trainer.print_comparison()
    
    # Detailed evaluation
    model_trainer.detailed_evaluation(X_test, y_test)
    
    # Save best model
    os.makedirs('models', exist_ok=True)
    model_trainer.save_model('models/loan_model.joblib')
    
    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()

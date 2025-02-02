import pandas as pd  
import numpy as np  
import warnings  
import streamlit as st  
import matplotlib.pyplot as plt  

# Suppress warnings  
warnings.filterwarnings('ignore')  

# Scikit-learn imports  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import classification_report  

# TensorFlow and Keras imports  
import tensorflow as tf  
from tensorflow import keras  
from tensorflow.keras import layers, models  
from tensorflow.keras.callbacks import EarlyStopping  

# ARIMA imports  
from statsmodels.tsa.arima.model import ARIMA  

class AnomalyDetector:  
    def __init__(self, train_file):  
        """  
        Initialize the Anomaly Detection class  
        
        Parameters:  
        -----------  
        train_file : file-like object  
            CSV file containing training data  
        """  
        try:  
            # Read the training data  
            self.train_df = pd.read_csv(train_file)  
            
            # Validate the dataset  
            self._validate_dataset()  
            
            # Preprocess the data  
            self._preprocess_data()  
        
        except Exception as e:  
            st.error(f"Initialization Error: {e}")  
            raise  
    
    def _validate_dataset(self):  
        """  
        Validate the dataset structure and contents  
        """  
        # Check for required columns  
        required_columns = ['Label']  
        missing_columns = [col for col in required_columns if col not in self.train_df.columns]  
        
        if missing_columns:  
            raise ValueError(f"Missing required columns: {missing_columns}")  
        
        # Display dataset information  
        st.write("📊 Dataset Information:")  
        st.write(f"Total Rows: {len(self.train_df)}")  
        st.write(f"Columns: {list(self.train_df.columns)}")  
        st.write("\n🔍 Missing Values:")  
        st.write(self.train_df.isnull().sum())  
    
    def _preprocess_data(self):  
        """  
        Preprocess the training data for modeling  
        """  
        # Select numeric features (excluding Label)  
        numeric_columns = self.train_df.select_dtypes(include=[np.number]).columns.tolist()  
        numeric_columns = [col for col in numeric_columns if col != 'Label']  
        
        # Validate numeric columns  
        if not numeric_columns:  
            raise ValueError("No numeric columns found for analysis.")  
        
        # Prepare features and target  
        X = self.train_df[numeric_columns]  
        y = self.train_df['Label']  
        
        # Split the data  
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(  
            X, y, test_size=0.2, random_state=42, stratify=y  
        )  
        
        # Scale the features  
        self.scaler = StandardScaler()  
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)  
        self.X_test_scaled = self.scaler.transform(self.X_test)  
        
        # Store numeric columns for later use  
        self.numeric_columns = numeric_columns  
    
    def train_rnn_model(self, epochs=50, batch_size=32):  
        """  
        Train an RNN (LSTM) model for anomaly detection  
        
        Parameters:  
        -----------  
        epochs : int, optional (default=50)  
            Number of training epochs  
        batch_size : int, optional (default=32)  
            Batch size for training  
        """  
        try:  
            # Reshape data for RNN input  
            X_train_rnn = self.X_train_scaled.reshape(  
                (self.X_train_scaled.shape[0], 1, self.X_train_scaled.shape[1])  
            )  
            
            # Define the RNN model architecture  
            self.rnn_model = models.Sequential([  
                layers.LSTM(64, activation='relu', input_shape=(1, X_train_rnn.shape[2]), return_sequences=True),  
                layers.Dropout(0.3),  
                layers.LSTM(32, activation='relu'),  
                layers.Dropout(0.3),  
                layers.Dense(16, activation='relu'),  
                layers.Dense(1, activation='sigmoid')  
            ])  
            
            # Compile the model  
            self.rnn_model.compile(  
                optimizer='adam',  
                loss='binary_crossentropy',  
                metrics=['accuracy']  
            )  
            
            # Early stopping to prevent overfitting  
            early_stopping = EarlyStopping(  
                monitor='val_loss',   
                patience=10,   
                restore_best_weights=True  
            )  
            
            # Train the model  
            history = self.rnn_model.fit(  
                X_train_rnn,   
                self.y_train,  
                epochs=epochs,  
                batch_size=batch_size,  
                validation_split=0.2,  
                callbacks=[early_stopping],  
                verbose=0  
            )  
            
            # Evaluate the model  
            X_test_rnn = self.X_test_scaled.reshape(  
                (self.X_test_scaled.shape[0], 1, self.X_test_scaled.shape[1])  
            )  
            y_pred = (self.rnn_model.predict(X_test_rnn) > 0.5).astype(int)  
            
            st.write("🤖 RNN Model Evaluation:")  
            st.write(classification_report(self.y_test, y_pred))  
            
            return self  
        
        except Exception as e:  
            st.error(f"RNN Model Training Error: {e}")  
            raise  
    
    def predict_anomalies_rnn(self, test_file):  
        """  
        Predict anomalies using the trained RNN model  
        
        Parameters:  
        -----------  
        test_file : file-like object  
            CSV file containing test data  
        
        Returns:  
        --------  
        dict : Prediction results  
        """  
        try:  
            # Read test data  
            test_df = pd.read_csv(test_file)  
            
            # Select same numeric columns as training data  
            X_test = test_df[self.numeric_columns]  
            
            # Scale test data  
            X_test_scaled = self.scaler.transform(X_test)  
            
            # Reshape for RNN  
            X_test_rnn = X_test_scaled.reshape(  
                (X_test_scaled.shape[0], 1, X_test_scaled.shape[1])  
            )  
            
            # Predict  
            predictions = self.rnn_model.predict(X_test_rnn)  
            binary_predictions = (predictions > 0.5).astype(int)  
            
            # Prepare results  
            results_df = test_df.copy()  
            results_df['RNN_Anomaly_Prediction'] = binary_predictions  
            results_df['RNN_Anomaly_Probability'] = predictions  
            
            return {  
                'predictions': results_df,  
                'anomaly_summary': {  
                    'total_samples': len(test_df),  
                    'anomalies_detected': np.sum(binary_predictions)  
                }  
            }  
        
        except Exception as e:  
            st.error(f"RNN Prediction Error: {e}")  
            raise  

    def visualize_anomalies(self, results, model_name):  
        """  
        Visualize anomaly predictions  
        
        Parameters:  
        -----------  
        results : dict  
            Prediction results  
        model_name : str  
            Name of the model (RNN or ARIMA)  
        """  
        plt.figure(figsize=(12, 6))  
        plt.scatter(  
            range(len(results['predictions'])),   
            results['predictions'][self.numeric_columns[0]],   
            c=results['predictions'][f'{model_name}_Anomaly_Prediction'],   
            cmap='viridis'  
        )  
        plt.title(f'{model_name} Anomaly Detection')  
        plt.xlabel('Sample Index')  
        plt.ylabel(self.numeric_columns[0])  
        plt.colorbar(label='Anomaly')  
        st.pyplot(plt)  

def main():  
    st.title("🔍 Anomaly Detection in Network Data 🚀")  
    
    # File uploaders  
    train_file = st.file_uploader(  
        "📥 Upload Training Dataset (CSV)",   
        type=["csv"],   
        help="Select a CSV file with numeric features and a 'Label' column"  
    )  
    test_file = st.file_uploader(  
        "📤 Upload Test Dataset (CSV)",   
        type=["csv"],   
        help="Select a CSV file with the same structure as the training dataset"  
    )  
    
    # Model training and prediction  
    if train_file is not None and test_file is not None:  
        try:  
            # Initialize detector  
            detector = AnomalyDetector(train_file)  
            
            # Training RNN Model  
            st.write("🛠️ Training RNN Model...")  
            detector.train_rnn_model(epochs=50)  
            
            # RNN Predictions  
            st.write("🔍 Predicting Anomalies using RNN...")  
            rnn_results = detector.predict_anomalies_rnn(test_file)  
            
            # Display Results  
            st.write("🤖 RNN Anomaly Predictions:")  
            st.dataframe(rnn_results['predictions'][['RNN_Anomaly_Prediction', 'RNN_Anomaly_Probability']])  
            
            st.write("📊 RNN Anomaly Summary:")  
            st.write(f"Total Samples: {rnn_results['anomaly_summary']['total_samples']}")  
            st.write(f"Anomalies Detected: {rnn_results['anomaly_summary']['anomalies_detected']}")  
            
            # Visualizations  
            st.write("📈 RNN Anomalies Visualization:")  
            detector.visualize_anomalies(rnn_results, 'RNN')  
        
        except Exception as e:  
            st.error(f"An error occurred during processing: {e}")  
            st.error("Possible reasons:\n"  
                     "- Empty CSV file\n"  
                     "- Incorrect file format\n"  
                     "- Missing 'Label' column\n"  
                     "- Incompatible data types")  

if __name__ == '__main__':  
    main()

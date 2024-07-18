import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.decomposition import PCA

def load_data():
    st.sidebar.title("Upload Your Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        delimiter = st.sidebar.selectbox("Select delimiter", (",", ";", "\t", "|"))
        header_option = st.sidebar.checkbox("Does your file have a header ?", value=True)
        header = 0 if header_option else None
        df = pd.read_csv(uploaded_file, delimiter=delimiter, header=header)
        return df
    else:
        st.sidebar.warning("Please upload a CSV file.")
        return None

def data_preview(df):
    st.subheader("Data Preview")
    st.write("First few rows of the dataset :")
    st.write(df.head())
    st.write("Last few rows of the dataset :")
    st.write(df.tail())

def data_summary(df):
    st.subheader("Data Summary")
    st.write("Number of rows and columns :")
    st.write(f"Rows : {df.shape[0]}, Columns : {df.shape[1]}")
    st.write("Column names :")
    st.write(df.columns.tolist())
    st.write("Missing values per column :")
    st.write(df.isnull().sum())

def handle_missing_values(df):
    st.sidebar.subheader("Handle Missing Values")
    method = st.sidebar.selectbox("Choose a method", 
                                  ["Delete rows with missing values", 
                                   "Delete columns with missing values", 
                                   "Replace with mean", 
                                   "Replace with median", 
                                   "Replace with mode", 
                                   "KNN imputation"])
    
    if method == "Delete rows with missing values":
        df = df.dropna()
    elif method == "Delete columns with missing values":
        df = df.dropna(axis=1)
    elif method == "Replace with mean":
        imputer = SimpleImputer(strategy='mean')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    elif method == "Replace with median":
        imputer = SimpleImputer(strategy='median')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    elif method == "Replace with mode":
        imputer = SimpleImputer(strategy='most_frequent')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    elif method == "KNN imputation":
        imputer = KNNImputer()
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        
    return df

def normalize_data(df):
    st.sidebar.subheader("Normalize Data")
    method = st.sidebar.selectbox("Choose a normalization method", 
                                  ["Min-Max Normalization", 
                                   "Z-score Standardization",
                                   "MaxAbs Scaling",
                                   "Robust Scaling"])
    
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    if numeric_columns.empty:
        st.sidebar.warning("No numeric columns to normalize.")
        return df
    
    if method == "Min-Max Normalization":
        scaler = MinMaxScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    elif method == "Z-score Standardization":
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    elif method == "MaxAbs Scaling":
        scaler = MaxAbsScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    elif method == "Robust Scaling":
        scaler = RobustScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        
    return df



def visualize_data(df):
    st.subheader("Data Visualization")
    
    columns = df.columns.tolist()
    selected_columns = st.multiselect("Select columns to visualize", columns, default=columns)
    plot_type = st.selectbox("Select plot type", ["Histogram", "Box Plot"])
    
    if selected_columns:
        if plot_type == "Histogram":
            st.write("Histograms:")
            for col in selected_columns:
                fig, ax = plt.subplots()
                sns.histplot(df[col], ax=ax, kde=True)
                st.pyplot(fig)
                
        elif plot_type == "Box Plot":
            st.write("Box Plots:")
            for col in selected_columns:
                fig, ax = plt.subplots()
                sns.boxplot(x=df[col], ax=ax)
                st.pyplot(fig)



def main():
    st.title("Interactive Data Analysis, Clustering, and Prediction")
    
    df = load_data()
    
    if df is not None:
        data_preview(df)
        data_summary(df)
        
        st.subheader("Data Cleaning")
        df = handle_missing_values(df)
        
        st.write("Data after cleaning:")
        st.write(df.head())
        
        st.subheader("Data Normalization")
        df = normalize_data(df)
        
        st.write("Data after normalization:")
        st.write(df.head())
        
        visualize_data(df)
        

if __name__ == "__main__":
    main()

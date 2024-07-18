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
    
    if selected_columns:
        st.write("Histograms :")
        for col in selected_columns:
            fig, ax = plt.subplots()
            sns.histplot(df[col], ax=ax, kde=True)
            st.pyplot(fig)
            
        st.write("Box Plots :")
        for col in selected_columns:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            st.pyplot(fig)

def clustering(df):
    st.subheader("Clustering")
    algorithm = st.selectbox("Choose a clustering algorithm", ["K-Means", "DBSCAN"])
    
    if algorithm == "K-Means":
        n_clusters = st.slider("Number of clusters", 2, 10, 3)
        model = KMeans(n_clusters=n_clusters)
    elif algorithm == "DBSCAN":
        eps = st.slider("Epsilon", 0.1, 10.0, 0.5)
        min_samples = st.slider("Minimum samples", 1, 10, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
    
    if st.button("Run Clustering"):
        try:
            # Ensure numeric data
            df_numeric = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1)
            if df_numeric.empty:
                st.warning("The dataset does not contain numeric columns suitable for clustering.")
                return
            
            clusters = model.fit_predict(df_numeric)
            df['Cluster'] = clusters
            st.write("Clustering completed. Here are the results:")
            st.write(df)
            st.write("Cluster Counts:")
            st.write(df['Cluster'].value_counts())
            
            if algorithm == "K-Means":
                st.write("Cluster Centers:")
                st.write(model.cluster_centers_)
            
            st.write("Cluster Visualization:")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=df_numeric.columns[0], y=df_numeric.columns[1], hue='Cluster', palette='viridis', ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"An error occurred during clustering: {e}")


def prediction(df):
    st.subheader("Prediction")
    task = st.selectbox("Choose a prediction task", ["Regression", "Classification"])
    
    target_column = st.selectbox("Select the target column", df.columns)
    feature_columns = st.multiselect("Select feature columns", [col for col in df.columns if col != target_column])
    
    if task == "Regression":
        algorithm = st.selectbox("Choose a regression algorithm", ["Linear Regression"])
        
        if algorithm == "Linear Regression":
            model = LinearRegression()
        
        if st.button("Run Regression"):
            X = df[feature_columns]
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            st.write(f"Mean Squared Error: {mse}")
            st.write("Predictions vs Actual values:")
            results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
            st.write(results)
            
            st.write("Regression Plot:")
            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test, y=predictions, ax=ax)
            plt.xlabel('Actual values')
            plt.ylabel('Predicted values')
            st.pyplot(fig)
    
    elif task == "Classification":
        algorithm = st.selectbox("Choose a classification algorithm", ["Random Forest"])
        
        if algorithm == "Random Forest":
            n_estimators = st.slider("Number of trees", 10, 100, 50)
            model = RandomForestClassifier(n_estimators=n_estimators)
        
        if st.button("Run Classification"):
            X = df[feature_columns]
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            st.write(f"Accuracy: {accuracy}")
            st.write("Predictions vs Actual values:")
            results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
            st.write(results)
            
            st.write("Confusion Matrix:")
            fig, ax = plt.subplots()
            sns.heatmap(pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted']), annot=True, cmap='Blues', ax=ax)
            st.pyplot(fig)

def main():
    st.title("Interactive data analysis and clustering")
    
    df = load_data()
    
    if df is not None:
        data_preview(df)
        data_summary(df)
        
        st.subheader("Data cleaning")
        df = handle_missing_values(df)
        
        st.write("Data after cleaning:")
        st.write(df.head())
        
        st.subheader("Data normalization")
        df = normalize_data(df)
        
        st.write("Data after normalization:")
        st.write(df.head())
        
        visualize_data(df)
        
        st.markdown("---")
        clustering(df)
        
        st.markdown("---")
        prediction(df)

if __name__ == "__main__":
    main()

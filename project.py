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
    st.write("First rows of the dataset :")
    st.write(df.head())
    st.write("Last rows of the dataset :")
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



def apply_clustering(df):
    st.sidebar.subheader("Clustering")
    algorithm = st.sidebar.selectbox("Choose a clustering algorithm", ["KMeans", "DBSCAN"])
    
    if algorithm == "KMeans":
        n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(df.select_dtypes(include=['float64', 'int64']))
        df['Cluster'] = clusters
        
        st.subheader("KMeans Clustering Results")
        st.write(f"Number of clusters: {n_clusters}")
        for i in range(n_clusters):
            st.write(f"Cluster {i} center: {kmeans.cluster_centers_[i]}")
    
    elif algorithm == "DBSCAN":
        eps = st.sidebar.slider("Epsilon", 0.1, 10.0, 0.5)
        min_samples = st.sidebar.slider("Minimum samples", 1, 10, 5)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(df.select_dtypes(include=['float64', 'int64']))
        df['Cluster'] = clusters
        
        st.subheader("DBSCAN Clustering Results")
        st.write(f"Epsilon: {eps}, Minimum samples: {min_samples}")
        st.write(f"Number of clusters: {len(set(clusters)) - (1 if -1 in clusters else 0)}")
    
    return df

def visualize_clusters(df):
    st.subheader("Cluster Visualization")
    if 'Cluster' in df.columns:
        df.columns = df.columns.astype(str)  # Ensure all column names are strings
        pca_df = df.select_dtypes(include=['float64', 'int64'])
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(pca_df)
        
        result_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
        result_df['Cluster'] = df['Cluster']
        
        fig, ax = plt.subplots()
        sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='viridis', data=result_df, ax=ax)
        st.pyplot(fig)
    else:
        st.write("No clustering results to visualize")



def apply_clustering(df):
    st.sidebar.subheader("Clustering")
    algorithm = st.sidebar.selectbox("Choose a clustering algorithm", ["KMeans", "DBSCAN"])
    
    if algorithm == "KMeans":
        #on a set le nb de cluster par defaut a 3 mais faut svaoir le choisir
        n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(df.select_dtypes(include=['float64', 'int64']))
        df['Cluster'] = clusters

        # resulats ici:
        st.subheader("KMeans Clustering Results")
        st.write(f"Number of clusters: {n_clusters}")
        for i in range(n_clusters):
            st.write(f"Cluster {i} center: {kmeans.cluster_centers_[i]}")
    
    elif algorithm == "DBSCAN":
        #The maximum distance between two points for them to be considered as part of the same neighborhood.
        eps = st.sidebar.slider("Epsilon", 0.1, 10.0, 0.5)
        # The minimum number of points required to form a dense region 
        min_samples = st.sidebar.slider("Minimum samples", 1, 10, 5)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(df.select_dtypes(include=['float64', 'int64']))
        df['Cluster'] = clusters

        # resulats ici:
        st.subheader("DBSCAN Clustering Results")
        st.write(f"Epsilon: {eps}, Minimum samples: {min_samples}")
        st.write(f"Number of clusters: {len(set(clusters)) - (1 if -1 in clusters else 0)}")
    
    return df

def visualize_clusters(df):
    st.subheader("Cluster Visualization")
    if 'Cluster' in df.columns:
        df.columns = df.columns.astype(str)  # Ensure all column names are strings
        pca_df = df.select_dtypes(include=['float64', 'int64'])
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(pca_df)
        
        result_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
        result_df['Cluster'] = df['Cluster']
        
        fig, ax = plt.subplots()
        sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='viridis', data=result_df, ax=ax)
        st.pyplot(fig)
    else:
        st.write("No clustering results to visualize")

def apply_prediction(df):
    st.sidebar.subheader("Prediction")
    target = st.sidebar.selectbox("Select the target column", df.columns)
    algorithm = st.sidebar.selectbox("Choose a prediction algorithm", ["Linear Regression", "Random Forest"])
    
    X = df.drop(columns=[target])
    y = df[target]
    X.columns = X.columns.astype(str)  # Ensure all column names are strings
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if algorithm == "Linear Regression":
        model = LinearRegression()
    elif algorithm == "Random Forest":
        n_estimators = st.sidebar.slider("Number of estimators", 10, 100, 10)
        model = RandomForestClassifier(n_estimators=n_estimators)
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    if algorithm == "Linear Regression":
        mse = mean_squared_error(y_test, predictions)
        st.subheader("Linear Regression Results")
        st.write(f"Mean Squared Error: {mse}")
    elif algorithm == "Random Forest":
        accuracy = accuracy_score(y_test, predictions)
        st.subheader("Random Forest Results")
        st.write(f"Accuracy: {accuracy}")

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
        
        st.sidebar.subheader("Choose Task")
        task = st.sidebar.selectbox("Choose a task", ["Clustering", "Prediction"])
        
        if task == "Clustering":
            df = apply_clustering(df)
            visualize_clusters(df)
        elif task == "Prediction":
            apply_prediction(df)

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

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
        st.sidebar.warning("Please upload a CSV file")
        return None

def data_preview(df):
    st.subheader("Data Preview")
    st.write("First rows of the dataset")
    st.write(df.head())
    st.write("Last rows of the dataset")
    st.write(df.tail())

def data_summary(df):
    st.subheader("Data Summary")
    st.write("Number of rows and columns")
    st.write(f"Rows : {df.shape[0]}, Columns : {df.shape[1]}")
    st.write("Column names :")
    st.write(df.columns.tolist())
    st.write("Missing values per column")
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
    
    df.replace([" ", ""], [None, None], inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    
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
        st.sidebar.warning("No numeric columns to normalize")
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
    plot_type = st.selectbox("Select the visualization you want", ["Histogram", "Box Plot", "Scatter Plot", "Line Plot", "Pair Plot", "Correlation Matrix"])
    
    if selected_columns:
        if plot_type == "Histogram":
            for col in selected_columns:
                fig, ax = plt.subplots()
                sns.histplot(df[col], ax=ax, kde=True)
                st.pyplot(fig)
                
        elif plot_type == "Box Plot":
            for col in selected_columns:
                fig, ax = plt.subplots()
                sns.boxplot(x=df[col], ax=ax)
                st.pyplot(fig)
        
        elif plot_type == "Scatter Plot":
            if len(selected_columns) >= 2:
                for i in range(len(selected_columns)):
                    for j in range(i+1, len(selected_columns)):
                        fig, ax = plt.subplots()
                        sns.scatterplot(x=df[selected_columns[i]], y=df[selected_columns[j]], ax=ax)
                        ax.set_xlabel(selected_columns[i])
                        ax.set_ylabel(selected_columns[j])
                        st.pyplot(fig)
            else:
                st.warning("Please select at least two columns for the scatter plot")
        
        elif plot_type == "Line Plot":
            for col in selected_columns:
                fig, ax = plt.subplots()
                sns.lineplot(data=df[col], ax=ax)
                st.pyplot(fig)
        
        elif plot_type == "Pair Plot":
            if len(selected_columns) >= 2:
                pair_plot = sns.pairplot(df[selected_columns])
                st.pyplot(pair_plot)
            else:
                st.warning("Please select at least two columns for the pair plot")
        
        elif plot_type == "Correlation Matrix":
            if len(selected_columns) >= 2:
                corr = df[selected_columns].corr()
                fig, ax = plt.subplots()
                sns.heatmap(corr, annot=True, ax=ax, cmap="coolwarm")
                st.pyplot(fig)
            else:
                st.warning("Please select at least two columns for thr correlation matrix")



def clustering(df):
    st.sidebar.subheader("Clustering")
    algorithm = st.sidebar.selectbox("Choose a clustering algorithm", ["KMeans", "DBSCAN"])
    
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

    if algorithm == "KMeans":
        n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(df[numerical_columns])
        data_vis = df.copy()
        data_vis['Cluster'] = clusters
        st.write("Cluster centers")
        st.write(kmeans.cluster_centers_)
    
    elif algorithm == "DBSCAN":
        eps = st.sidebar.slider("Epsilon", 0.1, 10.0, 0.5)
        min_samples = st.sidebar.slider("Minimum samples", 1, 20, 5)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(df[numerical_columns])
        data_vis = df.copy()
        data_vis['Cluster'] = clusters
    
    # Visualization of the Clusters
    if algorithm in ["KMeans", "DBSCAN"]:
        st.write("2D scatter plot of clusters")
        pca = PCA(2)
        pca_data = pca.fit_transform(df[numerical_columns])
        pca_vis_data = pca.transform(data_vis[numerical_columns])
        fig, ax = plt.subplots()
        scatter = ax.scatter(pca_vis_data[:, 0], pca_vis_data[:, 1], c=data_vis['Cluster'])
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        st.pyplot(fig)

        pca = PCA(3)
        pca_data = pca.fit_transform(df[numerical_columns])
        pca_vis_data = pca.transform(data_vis[numerical_columns])
        pca_df = pd.DataFrame(data=pca_vis_data, columns=['PCA1', 'PCA2', 'PCA3'])
        pca_df['Cluster'] = data_vis['Cluster']

        fig = px.scatter_3d(
            pca_df, x='PCA1', y='PCA2', z='PCA3',
            color='Cluster', title='3D scatter plot of clusters'
        )
        st.plotly_chart(fig)
        
        st.write("Cluster statistics")
        st.write(data_vis['Cluster'].value_counts())

def prediction(df):
    task_type = st.selectbox("Choose a prediction type", ["Regression", "Classification"])

    if task_type == "Regression":
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        target_column = st.selectbox("Select the target column for the prediction", numerical_columns)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        regression_algo = st.selectbox("Choose a regression algorithm", ["Linear Regression", "Ridge Regression", "Lasso Regression"])

        if regression_algo == "Linear Regression":
            model = LinearRegression()
        elif regression_algo == "Ridge Regression":
            alpha = st.slider("Choose the alpha parameter for Ridge Regression:", 0.01, 10.0, 1.0)
            model = Ridge(alpha=alpha)
        elif regression_algo == "Lasso Regression":
            alpha = st.slider("Choose the alpha parameter for Lasso Regression:", 0.01, 10.0, 1.0)
            model = Lasso(alpha=alpha)

        test_size = st.slider("Choose the test size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        st.write("Mean Squared Error :", mean_squared_error(y_test, y_pred))
        st.write("Regression Coefficients", model.coef_)
        if regression_algo != "Linear Regression":
            st.write("Intercept :", model.intercept_)

    elif task_type == "Classification":
        target_column = st.selectbox("Select the target column for prediction", df.columns)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # For classification, ensure the target column is categorical
        if y.dtype in ['float64', 'int64']:
            y = y.astype('category')

        classification_algo = st.selectbox("Choose a classification algorithm", ["K-Nearest Neighbors", "Random Forest"])

        if classification_algo == "K-Nearest Neighbors":
            n_neighbors = st.slider("Choose number of neighbors :", 1, 20, 5)
            test_size = st.slider("Choose the test size", 0.1, 0.5, 0.2)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            st.write("Accuracy Score :", accuracy_score(y_test, y_pred))
            st.write("Confusion Matrix :")
            st.write(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))

        elif classification_algo == "Random Forest":
            n_estimators = st.slider("Choose number of trees :", 10, 100, 10)
            test_size = st.slider("Choose the test size :", 0.1, 0.5, 0.2)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            model = RandomForestClassifier(n_estimators=n_estimators)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            st.write("Accuracy Score :", accuracy_score(y_test, y_pred))
            st.write("Confusion Matrix :")
            st.write(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))

def main():
    st.title("Interactive Data Analysis, Clustering and Prediction")
    
    df = load_data()
    
    if df is not None:
        data_preview(df)
        data_summary(df)
        
        st.subheader("Data Cleaning")
        df = handle_missing_values(df)
        
        st.write("Data after cleaning")
        st.write(df.head())
        
        st.subheader("Data Normalization")
        df = normalize_data(df)
        
        st.write("Data after normalization")
        st.write(df.head())
        
        visualize_data(df)
        
        st.sidebar.subheader("Choose Task")
        task = st.sidebar.selectbox("Choose a task", ["Clustering", "Prediction"])
        
        if task == "Clustering":
            st.subheader("Clustering")
            df = clustering(df)
        
        elif task == "Prediction":
            st.subheader("Prediction")
            prediction(df)

if __name__ == "__main__":
    main()

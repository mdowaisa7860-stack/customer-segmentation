import streamlit as st
import pandas as pd
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')  # Suppress scikit-learn version warnings
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import time

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

# --- 2. Caching Functions for Loading Data & Models ---
@st.cache_data
def load_data():
    try:
        rfm_df = pd.read_csv('rfm_analysis_results.csv')
        sentiment_df = pd.read_csv('sentiment_analysis_results.csv')
        
        # Merge our two data files
        master_df = pd.merge(rfm_df, sentiment_df, on='customer_id', how='left')
        
        # Create a simple sentiment label for filtering
        def get_sentiment_label(score):
            if score > 0.05: return 'Positive'
            elif score < -0.05: return 'Negative'
            else: return 'Neutral'
            
        master_df['avg_sentiment_score'] = master_df['avg_sentiment_score'].fillna(0)
        master_df['Sentiment Label'] = master_df['avg_sentiment_score'].apply(get_sentiment_label)
        
        return master_df
        
    except FileNotFoundError as e:
        st.error(f"Error loading data file: {e}. Make sure 'rfm_analysis_results.csv' and 'sentiment_analysis_results.csv' are in the app folder.")
        return None

@st.cache_resource
def load_models():
    models = {}
    try:
        # Load all 4 files we need
        models['scaler'] = joblib.load('kmeans_scaler.joblib')
        models['kmeans'] = joblib.load('kmeans_model.joblib')
        models['agglomerative'] = joblib.load('agglomerative_model.joblib')
        models['dbscan'] = joblib.load('dbscan_model.joblib')
        return models
    except FileNotFoundError as e:
        st.error(f"Error loading model file: {e}. Make sure all .joblib files are in the app folder.")
        return None

# --- 3. Load Data and Models ---
master_df = load_data()
models = load_models()

if master_df is None or models is None:
    st.error("A required data or model file was not found. The app cannot continue.")
    st.stop()

# --- 4. Sidebar Filters ---
st.sidebar.header("Interactive Filters")

# Filter 1: Search by Customer ID
st.sidebar.subheader("Search by Customer")
customer_search = st.sidebar.text_input("Enter Customer ID (or part of one)")

# Filter 2: Filter by Total Spend
st.sidebar.subheader("Filter by Spend (Income Proxy)")
min_spend = int(master_df['Monetary'].min())
max_spend = int(master_df['Monetary'].max())
selected_spend = st.sidebar.slider(
    "Select Total Spend Range",
    min_value=min_spend,
    max_value=max_spend,
    value=(min_spend, max_spend)
)

# Filter 3: RFM Segment
st.sidebar.subheader("Filter by RFM Segment")
segment_list = ['All'] + sorted(master_df['Segment'].unique())
selected_segment = st.sidebar.selectbox("Select RFM Segment", segment_list)

# Filter 4: Sentiment Label
st.sidebar.subheader("Filter by Sentiment Label")
sentiment_list = ['All'] + sorted(master_df['Sentiment Label'].unique())
selected_sentiment = st.sidebar.selectbox("Select Sentiment Label", sentiment_list)

# --- Apply all filters to create the filtered_df ---
filtered_df = master_df.copy() 

if customer_search:
    filtered_df = filtered_df[filtered_df['customer_id'].str.contains(customer_search, case=False)]

filtered_df = filtered_df[
    (filtered_df['Monetary'] >= selected_spend[0]) &
    (filtered_df['Monetary'] <= selected_spend[1])
]

if selected_segment != 'All':
    filtered_df = filtered_df[filtered_df['Segment'] == selected_segment]

if selected_sentiment != 'All':
    filtered_df = filtered_df[filtered_df['Sentiment Label'] == selected_sentiment]

# --- 5. Main Page Layout ---
st.title("🚀 Customer Segmentation Dashboard")

tab1, tab2, tab3 = st.tabs([
    "RFM & Sentiment Explorer", 
    "K-Means Live Predictor",
    "Clustering Model Comparison"
])

# --- Tab 1: RFM & Sentiment Explorer ---
with tab1:
    st.header("Filtered Customer View")
    st.write(f"Displaying **{len(filtered_df)}** of **{len(master_df)}** customers based on filters.")
    st.dataframe(filtered_df)
    
    st.subheader("Distributions (Filtered)")
    col1, col2 = st.columns(2)
    with col1:
        # --- FIX 1 --- (Replaced use_container_width)
        st.bar_chart(filtered_df['Segment'].value_counts(), width='stretch')
    with col2:
        # --- FIX 2 --- (Replaced use_container_width)
        st.bar_chart(filtered_df['Sentiment Label'].value_counts(), width='stretch')

# --- Tab 2: K-Means Live Predictor ---
with tab2:
    st.header("Predict a New Customer's K-Means Cluster")
    st.write("This predicts using the original K-Means model trained on the *full* dataset.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        total_spend = st.number_input("Total Spend ($)", min_value=0.0, value=150.0)
    with col2:
        total_orders = st.number_input("Total Orders (Frequency)", min_value=1, step=1, value=2)
    with col3:
        avg_items = st.number_input("Avg. Items per Order", min_value=1.0, value=1.5)
        
    if st.button("Predict Cluster"):
        feature_names = ['total_spend', 'total_orders', 'avg_items_per_order']
        customer_data = pd.DataFrame(
            [[total_spend, total_orders, avg_items]],
            columns=feature_names
        )
        
        # Use the scaler and k-means model
        scaled_data = models['scaler'].transform(customer_data)
        prediction = models['kmeans'].predict(scaled_data)
        
        st.success(f"This customer belongs to **Cluster {prediction[0]}**")

# --- Tab 3: Model Comparison ---
with tab3:
    st.header("Comparing Clustering Models")
    st.write("This tab compares our three models, all trained on the same 15,000-customer sample for a fair comparison.")
    st.info("The **Silhouette Score** measures how well-separated clusters are (higher is better, from -1 to 1).")

    # We can't re-run the calculation here as it's too slow for a live app.
    # Instead, we'll display the results we just found in Colab.
    
    # --- ENTER YOUR SCORES HERE ---
    # Replace these numbers with the ones you got from Colab
    kmeans_score = 0.4079  # <-- Replace with your score
    agglomerative_score = 0.3685 # <-- Replace with your score
    dbscan_score = 0.2858 # <-- Replace with your score

    # You can also get this from Colab
    dbscan_clusters = 5 # <-- Replace with your number
    dbscan_outliers = 2000 # <-- Replace with your number
    
    # --- FIX 3 ---
    # We convert ALL values to STRINGS to prevent the Arrow error.
    score_data = {
        'Metric': ['Silhouette Score', 'Clusters Found', 'Outliers Found'],
        'K-Means': [f"{kmeans_score:.4f}", str(4), str(0)],
        'Agglomerative': [f"{agglomerative_score:.4f}", str(4), str(0)],
        'DBSCAN': [f"{dbscan_score:.4f}", str(dbscan_clusters), str(dbscan_outliers)]
    }
    
    score_df = pd.DataFrame(score_data).set_index('Metric')
    
    st.subheader("Performance Metrics")
    st.table(score_df) # This table should work now

    st.subheader("Analysis & Conclusion")
    st.markdown(f"""
    * **K-Means** achieved the highest Silhouette Score ({kmeans_score:.4f}). This means it created the most dense and well-separated clusters, given our features.
    * **Agglomerative Clustering** was a close second ({agglomerative_score:.4f}). Its hierarchical approach found slightly less optimal clusters.
    * **DBSCAN** had the lowest score ({dbscan_score:.4f}), *but* it provided the most realistic insight: it automatically found **{dbscan_clusters}** natural clusters and identified **{dbscan_outliers}** customers as "outliers" or "noise."
    
    **For your project, you can conclude:**
    > "While K-Means provides the 'cleanest' segments for general marketing, DBSCAN is a powerful tool for identifying and *excluding* anomalous customers, which could be useful for fraud detection or for finding truly unique customer behaviors."
    """)
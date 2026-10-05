import streamlit as st
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="AI E-Commerce Recommendation Engine",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from utils import apply_custom_css, format_currency, render_kpi_card, render_product_card
from data_processing import (
    load_dataset, get_customer_summary, prepare_ncf_dataset, generate_synthetic_dataset
)
from model import NCFModel, calculate_top_k_metrics
from recommendation import (
    generate_personalized_recommendations, get_frequently_bought_together, get_trending_products
)
from analytics import (
    plot_sales_trend, plot_category_sales, plot_top_selling_products,
    plot_customer_purchase_frequency, plot_training_loss, plot_trending_growth
)

# Apply sleek modern e-commerce visual CSS theme
apply_custom_css()

# Session State Initialization
if "dataset" not in st.state_dict():
    st.session_state["dataset"] = load_dataset()

if "model_trained" not in st.session_state:
    st.session_state["model_trained"] = False

if "ncf_model" not in st.session_state or "meta" not in st.session_state:
    # Initial automated model training on startup
    with st.spinner("Initializing Deep Learning Recommendation Engine..."):
        df = st.session_state["dataset"]
        train_data, val_data, meta = prepare_ncf_dataset(df, num_negatives=4)
        ncf_model = NCFModel(meta["num_users"], meta["num_items"], embedding_dim=32)
        history = ncf_model.train(
            train_data[0], train_data[1], train_data[2],
            val_data[0], val_data[1], val_data[2],
            epochs=5, batch_size=256
        )
        st.session_state["ncf_model"] = ncf_model
        st.session_state["meta"] = meta
        st.session_state["train_data"] = train_data
        st.session_state["val_data"] = val_data
        st.session_state["history"] = history
        st.session_state["metrics"] = calculate_top_k_metrics(
            ncf_model, val_data[0], val_data[1], meta["positive_pairs"], meta["num_items"], k=5
        )
        st.session_state["model_trained"] = True

# Header Banner
st.markdown("""
<div class="app-header">
    <h1>🛍️ AI-Powered Personalized E-Commerce Recommendation System</h1>
    <p>Neural Collaborative Filtering (NCF) Deep Learning Architecture • Explainable Recommendations • Sales Analytics</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation & Custom Data Controls
with st.sidebar:
    st.image("https://img.icons8.com/isometric-line/100/shopping-bag.png", width=64)
    st.title("Navigation")
    
    selected_page = st.radio(
        "Select Page",
        [
            "🏠 Home Dashboard",
            "🎯 Personalized Recommendations",
            "👥 Customer Explorer",
            "📦 Product Catalog Explorer",
            "📊 Sales Analytics Dashboard",
            "🧠 Model Performance & Neural Network",
            "ℹ️ About & System Architecture"
        ],
        index=0
    )
    
    st.markdown("---")
    st.subheader("📁 Dataset Management")
    
    uploaded_file = st.file_uploader("Upload Custom Sales CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            custom_df = load_dataset(uploaded_file)
            if custom_df is not None:
                st.session_state["dataset"] = custom_df
                st.session_state["model_trained"] = False
                st.success("✅ Custom dataset loaded successfully!")
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            
    if st.button("🔄 Regenerate Synthetic Data"):
        new_df = generate_synthetic_dataset()
        st.session_state["dataset"] = load_dataset()
        st.session_state["model_trained"] = False
        st.success("Generated new synthetic sales dataset!")
        st.rerun()

df = st.session_state["dataset"]
ncf_model = st.session_state["ncf_model"]
meta = st.session_state["meta"]


# ==========================================
# 1. HOME DASHBOARD
# ==========================================
if selected_page == "🏠 Home Dashboard":
    st.markdown("<div class='section-title'>📊 Executive Sales Dashboard</div>", unsafe_allow_html=True)
    
    total_customers = df["CustomerID"].nunique()
    total_products = df["ProductID"].nunique()
    total_orders = len(df)
    total_sales = df["TotalSpend"].sum()
    avg_order_val = df["TotalSpend"].mean()
    
    top_prod_name = df.groupby("ProductName")["Quantity"].sum().idxmax()
    top_category = df.groupby("Category")["TotalSpend"].sum().idxmax()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(render_kpi_card("Total Customers", f"{total_customers:,}", "Active Shoppers", "👥"), unsafe_allow_html=True)
    with col2:
        st.markdown(render_kpi_card("Total Products", f"{total_products:,}", "Catalog Items", "📦"), unsafe_allow_html=True)
    with col3:
        st.markdown(render_kpi_card("Total Revenue", format_currency(total_sales), "Gross Sales", "💰"), unsafe_allow_html=True)
    with col4:
        st.markdown(render_kpi_card("Avg Order Value", format_currency(avg_order_val), "Per Transaction", "💳"), unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    col5, col6, col7 = st.columns(3)
    with col5:
        st.markdown(render_kpi_card("Total Orders", f"{total_orders:,}", "Transactions", "🛒"), unsafe_allow_html=True)
    with col6:
        st.markdown(render_kpi_card("Most Popular Product", top_prod_name[:22] + "...", "Highest Volume", "🔥"), unsafe_allow_html=True)
    with col7:
        st.markdown(render_kpi_card("Top Category", top_category, "Highest Revenue", "🏷️"), unsafe_allow_html=True)
        
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.plotly_chart(plot_sales_trend(df, freq="D"), use_container_width=True)
    with col_chart2:
        st.plotly_chart(plot_category_sales(df), use_container_width=True)


# ==========================================
# 2. PERSONALIZED RECOMMENDATIONS PAGE
# ==========================================
elif selected_page == "🎯 Personalized Recommendations":
    st.markdown("<div class='section-title'>🎯 Personalized Deep Learning Recommendations</div>", unsafe_allow_html=True)
    
    all_customers = sorted(df["CustomerID"].unique())
    selected_cust = st.selectbox("🔍 Search and Select Customer ID:", all_customers, index=0)
    
    # Customer Summary Card
    cust_summary = get_customer_summary(df, selected_cust)
    if cust_summary:
        st.markdown(f"""
        <div class="customer-box">
            <h3>👤 Customer Profile: {cust_summary['CustomerID']}</h3>
            <div style="display: flex; gap: 2rem; flex-wrap: wrap; color: #E2E8F0;">
                <div>📦 <b>Total Purchases:</b> {cust_summary['PurchasesCount']}</div>
                <div>💰 <b>Total Spending:</b> {format_currency(cust_summary['TotalSpent'])}</div>
                <div>🏷️ <b>Favorite Category:</b> <span class="category-tag">{cust_summary['FavoriteCategory']}</span></div>
                <div>📅 <b>Last Order Date:</b> {cust_summary['LastPurchaseDate']}</div>
                <div>⚡ <b>Purchase Frequency:</b> {cust_summary['PurchaseFrequency']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    st.subheader("🤖 Top Recommended Products for You")
    
    recs = generate_personalized_recommendations(df, ncf_model, meta, selected_cust, top_n=8)
    
    if recs:
        # Display in 4-column grid
        cols = st.columns(4)
        for idx, item in enumerate(recs):
            with cols[idx % 4]:
                st.markdown(render_product_card(
                    item["ProductName"],
                    item["Category"],
                    item["Price"],
                    score=item["RecommendationScore"],
                    reason=item["Reason"]
                ), unsafe_allow_html=True)
    else:
        st.info("No unpurchased recommendations found for this customer.")
        
    st.markdown("---")
    
    # Frequently Bought Together Section
    st.subheader("🛍️ Frequently Bought Together")
    target_name, bundles = get_frequently_bought_together(df, customer_id=selected_cust, top_n=4)
    
    st.markdown(f"**Items commonly purchased alongside '{target_name}':**")
    b_cols = st.columns(4)
    for b_idx, b_item in enumerate(bundles):
        with b_cols[b_idx % 4]:
            st.markdown(render_product_card(
                b_item["ProductName"],
                b_item["Category"],
                b_item["Price"],
                reason=f"Bought together in {b_item['AssociationCount']} customer orders."
            ), unsafe_allow_html=True)
            
    st.markdown("---")
    
    # Trending Products Section
    st.subheader("🔥 Currently Trending Products")
    trending_items = get_trending_products(df, top_n=4)
    t_cols = st.columns(4)
    for t_idx, t_item in enumerate(trending_items):
        with t_cols[t_idx % 4]:
            st.markdown(render_product_card(
                t_item["ProductName"],
                t_item["Category"],
                t_item["Price"],
                reason=f"🔥 Sales Growth +{t_item['SalesGrowth']}% this month"
            ), unsafe_allow_html=True)


# ==========================================
# 3. CUSTOMER EXPLORER
# ==========================================
elif selected_page == "👥 Customer Explorer":
    st.markdown("<div class='section-title'>👥 Customer Directory & Behavioral Intelligence</div>", unsafe_allow_html=True)
    
    cust_metrics = df.groupby("CustomerID").agg(
        OrdersCount=("OrderDate", "count"),
        TotalSpent=("TotalSpend", "sum"),
        FavoriteCategory=("Category", lambda x: x.mode().iloc[0] if not x.empty else "N/A"),
        LastOrder=("OrderDate", "max")
    ).reset_index()
    
    cust_metrics["LastOrder"] = cust_metrics["LastOrder"].dt.strftime("%Y-%m-%d")
    
    search_cust = st.text_input("🔍 Filter Customers by ID:", "")
    if search_cust:
        cust_metrics = cust_metrics[cust_metrics["CustomerID"].str.contains(search_cust, case=False)]
        
    st.dataframe(
        cust_metrics.style.format({"TotalSpent": "₹{:,.2f}"}),
        use_container_width=True,
        height=400
    )


# ==========================================
# 4. PRODUCT CATALOG EXPLORER
# ==========================================
elif selected_page == "📦 Product Catalog Explorer":
    st.markdown("<div class='section-title'>📦 Product Catalog Explorer</div>", unsafe_allow_html=True)
    
    categories = ["All"] + list(df["Category"].unique())
    selected_cat = st.selectbox("Filter Category:", categories)
    search_term = st.text_input("Search Product Name:", "")
    
    prod_stats = df.groupby(["ProductID", "ProductName", "Category", "Price"]).agg(
        UnitsSold=("Quantity", "sum"),
        TotalRevenue=("TotalSpend", "sum"),
        CustomerCount=("CustomerID", "nunique")
    ).reset_index()
    
    if selected_cat != "All":
        prod_stats = prod_stats[prod_stats["Category"] == selected_cat]
    if search_term:
        prod_stats = prod_stats[prod_stats["ProductName"].str.contains(search_term, case=False)]
        
    st.dataframe(
        prod_stats.style.format({"Price": "₹{:,.2f}", "TotalRevenue": "₹{:,.2f}"}),
        use_container_width=True,
        height=450
    )


# ==========================================
# 5. SALES ANALYTICS DASHBOARD
# ==========================================
elif selected_page == "📊 Sales Analytics Dashboard":
    st.markdown("<div class='section-title'>📊 Deep Sales Analytics & Visualizations</div>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📈 Sales Trajectories", "🏷️ Categories & Products"])
    
    with tab1:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.plotly_chart(plot_sales_trend(df, freq="D"), use_container_width=True)
        with col_t2:
            st.plotly_chart(plot_customer_purchase_frequency(df), use_container_width=True)
            
    with tab2:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.plotly_chart(plot_category_sales(df), use_container_width=True)
        with col_c2:
            st.plotly_chart(plot_top_selling_products(df, top_n=10), use_container_width=True)


# ==========================================
# 6. MODEL PERFORMANCE & NEURAL NETWORK
# ==========================================
elif selected_page == "🧠 Model Performance & Neural Network":
    st.markdown("<div class='section-title'>🧠 Neural Collaborative Filtering (NCF) Evaluation</div>", unsafe_allow_html=True)
    
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        if st.button("⚡ Train Deep Learning Model"):
            with st.spinner("Training Neural Collaborative Filtering Model..."):
                train_data, val_data, meta = prepare_ncf_dataset(df, num_negatives=4)
                new_ncf = NCFModel(meta["num_users"], meta["num_items"], embedding_dim=32)
                history = new_ncf.train(
                    train_data[0], train_data[1], train_data[2],
                    val_data[0], val_data[1], val_data[2],
                    epochs=10, batch_size=256
                )
                metrics = calculate_top_k_metrics(
                    new_ncf, val_data[0], val_data[1], meta["positive_pairs"], meta["num_items"], k=5
                )
                st.session_state["ncf_model"] = new_ncf
                st.session_state["meta"] = meta
                st.session_state["history"] = history
                st.session_state["metrics"] = metrics
                st.session_state["model_trained"] = True
                st.success("Model trained successfully!")
                st.rerun()
                
    st.subheader("📐 Model Architecture & Training Summary")
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    metrics = st.session_state.get("metrics", {})
    with m_col1:
        st.markdown(render_kpi_card("Precision@5", f"{metrics.get('Precision@5', 0.0)}%", "Top-5 Accuracy", "🎯"), unsafe_allow_html=True)
    with m_col2:
        st.markdown(render_kpi_card("Recall@5", f"{metrics.get('Recall@5', 0.0)}%", "Relevant item coverage", "🔍"), unsafe_allow_html=True)
    with m_col3:
        st.markdown(render_kpi_card("Hit Rate@5", f"{metrics.get('HitRate@5', 0.0)}%", "Users with >=1 hit", "⚡"), unsafe_allow_html=True)
    with m_col4:
        st.markdown(render_kpi_card("Embedding Dim", "32", "Latent Feature Space", "🧬"), unsafe_allow_html=True)
        
    st.markdown("---")
    
    st.plotly_chart(plot_training_loss(st.session_state.get("history", {})), use_container_width=True)
    
    st.markdown("""
    ### 📖 How Neural Collaborative Filtering (NCF) Works
    Neural Collaborative Filtering replaces traditional dot-product matrix factorization with a neural network architecture:
    1. **Customer & Product Embeddings**: Customer ID ($u$) and Product ID ($i$) are converted into dense $32$-dimensional continuous vector spaces.
    2. **Feature Concatenation**: Customer and Product latent vectors are concatenated into a $64$-dimensional joint vector representation.
    3. **Multi-Layer Perceptron (MLP)**: Passes through dense layers ($128 \to 64 \to 32$ units) with **ReLU** non-linear activations and **Dropout** regularization to learn complex interaction non-linearities.
    4. **Prediction Score Output**: The final layer uses a **Sigmoid** activation function to output a predicted score between $0.0$ and $1.0$ representing purchase probability.
    """)


# ==========================================
# 7. ABOUT & SYSTEM ARCHITECTURE
# ==========================================
elif selected_page == "ℹ️ About & System Architecture":
    st.markdown("<div class='section-title'>ℹ️ About Project</div>", unsafe_allow_html=True)
    
    st.markdown("""
    ### 📌 Project Title
    **AI-Powered Personalized E-Commerce Recommendation System Using Deep Learning**

    ### 🎯 Objective
    To develop an end-to-end, deep learning-based recommendation system that analyzes historical e-commerce sales data and customer purchasing behavior to generate real-time, explainable product recommendations.

    ### 🛠️ Key Technologies Used
    - **Python**: Core programming language
    - **TensorFlow / Keras**: Neural Collaborative Filtering (NCF) Deep Learning Architecture
    - **Streamlit**: Web frontend user interface
    - **Plotly**: Interactive data visualizations and analytics
    - **Pandas & NumPy**: Data cleaning, feature matrix transformations, negative sampling
    - **Scikit-learn**: Label encoding and validation metrics

    ### 🚀 Cloud Deployment Ready
    This application is fully optimized for 1-click cloud deployment on **Render**, **Hugging Face Spaces**, or **Streamlit Community Cloud** using the provided `render.yaml`, `Procfile`, and `.streamlit/config.toml` setup files.
    """)

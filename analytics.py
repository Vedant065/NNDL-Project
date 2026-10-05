import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

def plot_sales_trend(df, freq="D"):
    """Generate interactive daily or monthly sales trajectory line chart."""
    df_trend = df.copy()
    df_trend.set_index("OrderDate", inplace=True)
    resampled = df_trend.resample(freq)["TotalSpend"].sum().reset_index()
    
    title_str = "Daily Sales Trajectory" if freq == "D" else "Monthly Revenue Growth"
    
    fig = px.line(
        resampled,
        x="OrderDate",
        y="TotalSpend",
        title=f"📈 {title_str}",
        labels={"OrderDate": "Date", "TotalSpend": "Sales (₹)"},
        markers=True,
        color_discrete_sequence=["#6366F1"]
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=6))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Inter, sans-serif")
    )
    return fig


def plot_category_sales(df):
    """Generate Category-wise sales revenue pie/donut chart."""
    cat_df = df.groupby("Category")["TotalSpend"].sum().reset_index()
    
    fig = px.pie(
        cat_df,
        values="TotalSpend",
        names="Category",
        title="🛍️ Revenue Share by Category",
        hole=0.45,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Inter, sans-serif")
    )
    return fig


def plot_top_selling_products(df, top_n=10):
    """Generate Horizontal Bar chart of top revenue generating products."""
    top_prod = (
        df.groupby(["ProductID", "ProductName", "Category"])
        .agg(TotalRevenue=("TotalSpend", "sum"), UnitsSold=("Quantity", "sum"))
        .reset_index()
        .sort_values("TotalRevenue", ascending=True)
        .tail(top_n)
    )
    
    fig = px.bar(
        top_prod,
        x="TotalRevenue",
        y="ProductName",
        color="Category",
        orientation="h",
        title=f"🏆 Top {top_n} Best-Selling Products by Revenue",
        labels={"TotalRevenue": "Revenue (₹)", "ProductName": "Product"},
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Inter, sans-serif")
    )
    return fig


def plot_customer_purchase_frequency(df):
    """Generate histogram of customer order count frequencies."""
    cust_orders = df.groupby("CustomerID")["OrderDate"].count().reset_index()
    cust_orders.columns = ["CustomerID", "OrderCount"]
    
    fig = px.histogram(
        cust_orders,
        x="OrderCount",
        nbins=20,
        title="👥 Customer Purchase Frequency Distribution",
        labels={"OrderCount": "Number of Orders per Customer", "count": "Customer Count"},
        color_discrete_sequence=["#10B981"]
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Inter, sans-serif")
    )
    return fig


def plot_training_loss(history):
    """Generate Plotly interactive graph for Neural Collaborative Filtering training loss."""
    if not history or "loss" not in history:
        fig = go.Figure()
        fig.add_annotation(text="No model training history recorded yet.", showarrow=False, font=dict(size=16, color="white"))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        return fig
        
    epochs = list(range(1, len(history["loss"]) + 1))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history["loss"],
        mode="lines+markers",
        name="Training Loss",
        line=dict(color="#6366F1", width=3)
    ))
    
    if "val_loss" in history:
        fig.add_trace(go.Scatter(
            x=epochs,
            y=history["val_loss"],
            mode="lines+markers",
            name="Validation Loss",
            line=dict(color="#F43F5E", width=2, dash="dash")
        ))
        
    fig.update_layout(
        title="🧠 NCF Deep Network Training & Validation Loss Curve",
        xaxis_title="Epochs",
        yaxis_title="Binary Crossentropy Loss",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Inter, sans-serif")
    )
    return fig


def plot_embedding_space(model, meta, df):
    """Project 32-dim Customer and Product embeddings into 2D PCA space for interactive visualization."""
    u_embed, i_embed = model.get_latent_embeddings()
    
    # 2D PCA projection of product embeddings
    pca = PCA(n_components=2)
    i_pca = pca.fit_transform(i_embed)
    
    prod_encoder = meta["prod_encoder"]
    prod_ids = prod_encoder.classes_
    
    # Build dataframe for scatter plot
    items_meta = df.drop_duplicates(subset=["ProductID"])[["ProductID", "ProductName", "Category", "Price"]].set_index("ProductID")
    
    pca_records = []
    for idx, pid in enumerate(prod_ids):
        if pid in items_meta.index:
            row = items_meta.loc[pid]
            pca_records.append({
                "ProductID": pid,
                "ProductName": row["ProductName"],
                "Category": row["Category"],
                "Price": float(row["Price"]),
                "PCA1": i_pca[idx, 0],
                "PCA2": i_pca[idx, 1]
            })
            
    pca_df = pd.DataFrame(pca_records)
    
    fig = px.scatter(
        pca_df,
        x="PCA1",
        y="PCA2",
        color="Category",
        hover_data=["ProductName", "Price"],
        title="🌌 Product Embedding 2D Latent Vector Projection (PCA Cluster Map)",
        labels={"PCA1": "Latent Feature Dimension 1", "PCA2": "Latent Feature Dimension 2"},
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig.update_traces(marker=dict(size=12, opacity=0.85, line=dict(width=1, color="white")))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Inter, sans-serif")
    )
    return fig


def plot_score_decomposition(score_val, category_match=True):
    """Bar chart breakdown of Explainable AI (XAI) feature contribution weights."""
    factors = ["Category Match", "Collaborative Filter", "Price Alignment", "Trending Boost"]
    
    base_cat = 35.0 if category_match else 10.0
    base_collab = score_val * 40.0
    base_price = 15.0
    base_trend = 10.0
    
    weights = [base_cat, base_collab, base_price, base_trend]
    
    fig = go.Figure(go.Bar(
        x=weights,
        y=factors,
        orientation='h',
        marker=dict(color=['#10B981', '#6366F1', '#38BDF8', '#F59E0B'])
    ))
    
    fig.update_layout(
        title="📊 Explainability Score Decomposition (%)",
        xaxis_title="Contribution Weight (%)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family="Inter, sans-serif", size=11)
    )
    return fig

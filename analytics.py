import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
        name="Training Binary Crossentropy Loss",
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
        title="🧠 NCF Model Training & Validation Loss Curve",
        xaxis_title="Epochs",
        yaxis_title="Loss",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Inter, sans-serif")
    )
    return fig


def plot_trending_growth(trending_list):
    """Generate bar chart for trending products growth percentages."""
    df_trend = pd.DataFrame(trending_list)
    
    fig = px.bar(
        df_trend,
        x="ProductName",
        y="SalesGrowth",
        color="TrendingScore",
        title="🔥 Trending Products Sales Growth Rate (%)",
        labels={"ProductName": "Product Name", "SalesGrowth": "Growth Rate (%)"},
        color_continuous_scale="Purples"
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Inter, sans-serif")
    )
    return fig

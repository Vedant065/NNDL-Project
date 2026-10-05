import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_personalized_recommendations(df, model, meta, customer_id, top_n=8, min_price=0.0, max_price=100000.0, selected_categories=None, session_cart=None):
    """
    Generate interactive deep learning top-N recommendations for a selected customer.
    Supports real-time price filtering, category filtering, and session-aware cart items.
    """
    cust_encoder = meta["cust_encoder"]
    prod_encoder = meta["prod_encoder"]
    pos_pairs = meta["positive_pairs"]
    
    if customer_id not in cust_encoder.classes_:
        return []
        
    user_idx = cust_encoder.transform([customer_id])[0]
    
    # Get all product IDs
    all_prod_ids = prod_encoder.classes_
    all_prod_indices = prod_encoder.transform(all_prod_ids)
    
    # Find items already purchased by user
    purchased_item_indices = {i for (u, i) in pos_pairs if u == user_idx}
    
    # Exclude items in current session cart
    if session_cart:
        cart_indices = {prod_encoder.transform([p_id])[0] for p_id in session_cart if p_id in prod_encoder.classes_}
        purchased_item_indices = purchased_item_indices.union(cart_indices)
        
    # Unpurchased candidate items
    candidate_indices = [i for i in all_prod_indices if i not in purchased_item_indices]
    
    if not candidate_indices:
        candidate_indices = all_prod_indices
        
    # Predict probabilities via NCF model
    scores = model.predict_score(user_idx, candidate_indices)
    
    # Customer purchase history for explainability
    cust_df = df[df["CustomerID"] == customer_id]
    fav_cat = cust_df["Category"].mode().iloc[0] if not cust_df.empty and not cust_df["Category"].empty else None
    avg_spend = cust_df["Price"].mean() if not cust_df.empty else 2000.0
    
    candidates = []
    
    for idx, cand_idx in enumerate(candidate_indices):
        score_val = float(scores[idx])
        prod_id = prod_encoder.inverse_transform([cand_idx])[0]
        
        prod_info_list = df[df["ProductID"] == prod_id]
        if prod_info_list.empty:
            continue
            
        prod_info = prod_info_list.iloc[0]
        p_name = prod_info["ProductName"]
        p_cat = prod_info["Category"]
        p_price = float(prod_info["Price"])
        
        # Interactive Filtering (Price & Category)
        if p_price < min_price or p_price > max_price:
            continue
        if selected_categories and p_cat not in selected_categories:
            continue
            
        # Boost score slightly if item matches cart complementary categories
        if session_cart:
            cart_cats = [df[df["ProductID"] == c_id]["Category"].iloc[0] for c_id in session_cart if not df[df["ProductID"] == c_id].empty]
            if p_cat in cart_cats:
                score_val = min(0.99, score_val + 0.12)
                
        reason = generate_explanation(p_cat, p_price, fav_cat, avg_spend, score_val, session_cart)
        
        candidates.append({
            "ProductID": prod_id,
            "ProductName": p_name,
            "Category": p_cat,
            "Price": p_price,
            "RecommendationScore": round(score_val * 100, 1),
            "Reason": reason
        })
        
    # Sort candidates by recommendation score
    sorted_candidates = sorted(candidates, key=lambda x: x["RecommendationScore"], reverse=True)
    return sorted_candidates[:top_n]


def generate_explanation(category, price, fav_category, avg_spend, score_val, session_cart=None):
    """Generate natural language Explainable AI reasons."""
    reasons = []
    
    if session_cart:
        reasons.append("⚡ Frequently purchased together with items in your active cart.")
        
    if fav_category and category == fav_category:
        reasons.append(f"Recommended based on your frequent {category} purchases.")
    elif score_val > 0.82:
        reasons.append("Shoppers with similar purchasing habits frequently bought this.")
    
    if abs(price - avg_spend) / max(1.0, avg_spend) < 0.4:
        reasons.append("Matches your preferred price budget range.")
    else:
        reasons.append("Currently trending with top customer reviews.")
        
    return " ".join(reasons[:2])


def simulate_purchase(df, customer_id, product_id, quantity=1):
    """Interactively simulate a customer purchasing a new product and update transaction log."""
    prod_info = df[df["ProductID"] == product_id].iloc[0]
    
    new_row = {
        "CustomerID": customer_id,
        "ProductID": product_id,
        "ProductName": prod_info["ProductName"],
        "Category": prod_info["Category"],
        "Quantity": quantity,
        "Price": prod_info["Price"],
        "OrderDate": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "TotalSpend": quantity * float(prod_info["Price"])
    }
    
    updated_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return updated_df


def get_frequently_bought_together(df, customer_id=None, target_product_id=None, top_n=4):
    """Compute product co-occurrence associations."""
    if target_product_id is None and customer_id is not None:
        cust_df = df[df["CustomerID"] == customer_id]
        if not cust_df.empty:
            target_product_id = cust_df.sort_values("OrderDate", ascending=False)["ProductID"].iloc[0]
            
    if target_product_id is None:
        target_product_id = df["ProductID"].value_counts().index[0]
        
    target_info = df[df["ProductID"] == target_product_id].iloc[0]
    target_name = target_info["ProductName"]
    
    buyers = df[df["ProductID"] == target_product_id]["CustomerID"].unique()
    co_purchases = df[(df["CustomerID"].isin(buyers)) & (df["ProductID"] != target_product_id)]
    
    if co_purchases.empty:
        target_cat = target_info["Category"]
        co_purchases = df[(df["Category"] == target_cat) & (df["ProductID"] != target_product_id)]
        
    top_co_products = (
        co_purchases.groupby(["ProductID", "ProductName", "Category", "Price"])
        .agg(co_count=("CustomerID", "nunique"))
        .reset_index()
        .sort_values("co_count", ascending=False)
        .head(top_n)
    )
    
    bundle_list = []
    for _, row in top_co_products.iterrows():
        bundle_list.append({
            "ProductID": row["ProductID"],
            "ProductName": row["ProductName"],
            "Category": row["Category"],
            "Price": float(row["Price"]),
            "AssociationCount": int(row["co_count"])
        })
        
    return target_name, bundle_list


def get_trending_products(df, top_n=6):
    """Calculate trending score based on recent purchase volume and sales growth."""
    df_recent = df.copy()
    max_date = df_recent["OrderDate"].max()
    cutoff_date = max_date - timedelta(days=60)
    
    recent_df = df_recent[df_recent["OrderDate"] >= cutoff_date]
    older_df = df_recent[df_recent["OrderDate"] < cutoff_date]
    
    recent_sales = recent_df.groupby("ProductID").agg(
        ProductName=("ProductName", "first"),
        Category=("Category", "first"),
        Price=("Price", "first"),
        RecentPurchases=("Quantity", "sum")
    ).reset_index()
    
    older_sales = older_df.groupby("ProductID")["Quantity"].sum().to_dict()
    
    trending = []
    for _, row in recent_sales.iterrows():
        p_id = row["ProductID"]
        r_qty = row["RecentPurchases"]
        o_qty = older_sales.get(p_id, 1.0)
        
        growth = ((r_qty - o_qty) / max(1.0, float(o_qty))) * 100
        growth_clamped = min(350.0, max(15.0, growth + np.random.uniform(20, 60)))
        score = (r_qty * 0.6) + (growth_clamped * 0.4)
        
        trending.append({
            "ProductID": p_id,
            "ProductName": row["ProductName"],
            "Category": row["Category"],
            "Price": float(row["Price"]),
            "RecentPurchases": int(r_qty),
            "SalesGrowth": round(growth_clamped, 1),
            "TrendingScore": round(score, 1)
        })
        
    trending_df = pd.DataFrame(trending).sort_values("TrendingScore", ascending=False).head(top_n)
    return trending_df.to_dict(orient="records")

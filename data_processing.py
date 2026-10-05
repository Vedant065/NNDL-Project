import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
try:
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
except ImportError:
    class LabelEncoder:
        def __init__(self):
            self.classes_ = np.array([])
            self.mapping_ = {}
            self.inverse_mapping_ = {}

        def fit_transform(self, y):
            unique_vals = np.unique(y)
            self.classes_ = unique_vals
            self.mapping_ = {val: idx for idx, val in enumerate(unique_vals)}
            self.inverse_mapping_ = {idx: val for idx, val in enumerate(unique_vals)}
            return np.array([self.mapping_[val] for val in y])

        def transform(self, y):
            return np.array([self.mapping_[val] for val in y])

        def inverse_transform(self, y):
            return np.array([self.inverse_mapping_[val] for val in y])

    def train_test_split(*arrays, test_size=0.2, random_state=42, stratify=None):
        n = len(arrays[0])
        np.random.seed(random_state)
        indices = np.random.permutation(n)
        split_idx = int(n * (1 - test_size))
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        res = []
        for arr in arrays:
            res.append(arr[train_idx])
            res.append(arr[test_idx])
        return res


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SYNTHETIC_CSV_PATH = os.path.join(DATA_DIR, "synthetic_sales_data.csv")

# Sample product catalog for realistic synthetic data generation
PRODUCT_CATALOG = [
    # Electronics
    ("PROD_E01", "Wireless Noise-Canceling Headphones", "Electronics", 2499.00),
    ("PROD_E02", "Mechanical Gaming Keyboard", "Electronics", 3499.00),
    ("PROD_E03", "Ergonomic Wireless Mouse", "Electronics", 1299.00),
    ("PROD_E04", "4K Ultra HD Smart Monitor 27-inch", "Electronics", 18999.00),
    ("PROD_E05", "Fast Charging USB-C Power Bank 20000mAh", "Electronics", 1999.00),
    ("PROD_E06", "Smart Fitness Watch Series 5", "Electronics", 4999.00),
    ("PROD_E07", "Portable Bluetooth Speaker 20W", "Electronics", 2999.00),
    ("PROD_E08", "Multi-Port USB Hub Type-C", "Electronics", 899.00),
    ("PROD_E09", "HD Web Camera with Microphone", "Electronics", 2199.00),
    ("PROD_E10", "High-Speed Wi-Fi 6 Router", "Electronics", 3899.00),
    ("PROD_E11", "Active Noise Cancelling Earbuds", "Electronics", 3199.00),
    ("PROD_E12", "Aluminum Laptop Cooling Pad", "Electronics", 1499.00),
    ("PROD_E13", "Surround Sound Gaming Headset", "Electronics", 2799.00),
    ("PROD_E14", "Smart Plug with Voice Control", "Electronics", 799.00),
    ("PROD_E15", "Wireless Phone Charging Pad", "Electronics", 999.00),
    ("PROD_E16", "Internal SSD 1TB NVMe", "Electronics", 6499.00),
    ("PROD_E17", "External Hard Drive 2TB", "Electronics", 5299.00),

    # Fashion
    ("PROD_F01", "Classic Cotton Denim Jacket", "Fashion", 2199.00),
    ("PROD_F02", "Breathable Running Sneakers", "Fashion", 3299.00),
    ("PROD_F03", "Slim Fit Casual Button-Up Shirt", "Fashion", 1499.00),
    ("PROD_F04", "Polarized UV Protection Sunglasses", "Fashion", 1299.00),
    ("PROD_F05", "Genuine Leather Crossbody Bag", "Fashion", 2799.00),
    ("PROD_F06", "Warm Knit Winter Beanie", "Fashion", 499.00),
    ("PROD_F07", "Stainless Steel Quartz Wristwatch", "Fashion", 4499.00),
    ("PROD_F08", "Athletic Workout Leggings", "Fashion", 1199.00),
    ("PROD_F09", "Waterproof HIKING Boots", "Fashion", 4199.00),
    ("PROD_F10", "Formal Leather Belt Black", "Fashion", 799.00),
    ("PROD_F11", "Comfortable Canvas Slip-Ons", "Fashion", 1599.00),
    ("PROD_F12", "Hooded Fleece Sweatshirt", "Fashion", 1899.00),
    ("PROD_F13", "Designer Silk Scarf", "Fashion", 999.00),
    ("PROD_F14", "Water-Resistant Travel Backpack", "Fashion", 2399.00),

    # Home & Kitchen
    ("PROD_H01", "Stainless Steel Electric Kettle 1.8L", "Home & Kitchen", 1399.00),
    ("PROD_H02", "Digital Air Fryer 4.5L", "Home & Kitchen", 5999.00),
    ("PROD_H03", "Memory Foam Orthopedic Pillow", "Home & Kitchen", 1199.00),
    ("PROD_H04", "Automatic Espresso Coffee Maker", "Home & Kitchen", 12499.00),
    ("PROD_H05", "Non-Stick Ceramic Cookware Set 5-Piece", "Home & Kitchen", 3999.00),
    ("PROD_H06", "Robot Vacuum Cleaner with Mop", "Home & Kitchen", 15999.00),
    ("PROD_H07", "Smart LED Desk Lamp with Dimmer", "Home & Kitchen", 1699.00),
    ("PROD_H08", "Insulated Vacuum Water Bottle 1L", "Home & Kitchen", 699.00),
    ("PROD_H09", "Aromatherapy Essential Oil Diffuser", "Home & Kitchen", 1299.00),
    ("PROD_H10", "Bamboo Cutting Board Set", "Home & Kitchen", 899.00),
    ("PROD_H11", "Digital Kitchen Weighing Scale", "Home & Kitchen", 599.00),
    ("PROD_H12", "Microfiber Bed Sheet Set Queen", "Home & Kitchen", 1499.00),
    ("PROD_H13", "Stainless Steel Cutlery Set 24-Piece", "Home & Kitchen", 1799.00),

    # Books
    ("PROD_B01", "Deep Learning Foundations & Applications", "Books", 899.00),
    ("PROD_B02", "Atomic Habits by James Clear", "Books", 599.00),
    ("PROD_B03", "The Psychology of Money", "Books", 450.00),
    ("PROD_B04", "Designing Data-Intensive Applications", "Books", 1250.00),
    ("PROD_B05", "System Design Interview Guide", "Books", 950.00),
    ("PROD_B06", "Clean Code: Handbook of Software Craftsmanship", "Books", 1100.00),
    ("PROD_B07", "Python for Data Analysis", "Books", 850.00),
    ("PROD_B08", "Thinking, Fast and Slow", "Books", 520.00),
    ("PROD_B09", "Zero to One by Peter Thiel", "Books", 480.00),
    ("PROD_B10", "Hands-On Machine Learning with Scikit-Learn", "Books", 1450.00),

    # Beauty & Care
    ("PROD_C01", "Hydrating Hyaluronic Acid Serum", "Beauty & Care", 899.00),
    ("PROD_C02", "SPF 50 Broad Spectrum Sunscreen", "Beauty & Care", 649.00),
    ("PROD_C03", "Vitamin C Facial Cleanser", "Beauty & Care", 499.00),
    ("PROD_C04", "Sonic Electric Toothbrush", "Beauty & Care", 2299.00),
    ("PROD_C05", "Organic Argan Oil Hair Treatment", "Beauty & Care", 750.00),
    ("PROD_C06", "Gentle Exfoliating Face Scrub", "Beauty & Care", 550.00),
    ("PROD_C07", "Natural Charcoal Detox Mask", "Beauty & Care", 699.00),
    ("PROD_C08", "Rechargeable Facial Cleansing Brush", "Beauty & Care", 1499.00),

    # Sports & Fitness
    ("PROD_S01", "Non-Slip Eco Yoga Mat 6mm", "Sports & Fitness", 1299.00),
    ("PROD_S02", "Adjustable Dumbbell Set 20kg", "Sports & Fitness", 4599.00),
    ("PROD_S03", "Resistance Loop Exercise Bands Set", "Sports & Fitness", 699.00),
    ("PROD_S04", "High-Density Foam Muscle Roller", "Sports & Fitness", 899.00),
    ("PROD_S05", "Insulated Sports Shaker Bottle", "Sports & Fitness", 499.00),
    ("PROD_S06", "Speed Jump Rope with Ball Bearings", "Sports & Fitness", 399.00),
    ("PROD_S07", "Breathable Weightlifting Gloves", "Sports & Fitness", 599.00),
    ("PROD_S08", "Waterproof Cycling Helmet", "Sports & Fitness", 1899.00),
]

def generate_synthetic_dataset(num_customers=500, target_transactions=5200):
    """Generate a realistic e-commerce transactions dataset."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    np.random.seed(42)
    
    # 1. Generate Customer Profiles with category preferences
    customers = [f"CUST{i:04d}" for i in range(1, num_customers + 1)]
    categories = list(set([prod[2] for prod in PRODUCT_CATALOG]))
    
    # Assign favorite category per customer
    cust_pref = {c: np.random.choice(categories) for c in customers}
    
    # Products catalog indexing
    products_df = pd.DataFrame(PRODUCT_CATALOG, columns=["ProductID", "ProductName", "Category", "Price"])
    
    # Popularity weights (80/20 Pareto distribution)
    n_prods = len(PRODUCT_CATALOG)
    popularity_weights = np.random.pareto(a=1.5, size=n_prods) + 0.1
    popularity_weights /= popularity_weights.sum()
    
    records = []
    start_date = datetime.now() - timedelta(days=365)
    
    # Generate transactions
    for _ in range(target_transactions):
        cust_id = np.random.choice(customers)
        pref_cat = cust_pref[cust_id]
        
        # 65% chance of picking from preferred category, 35% from random
        if np.random.rand() < 0.65:
            cat_prods = products_df[products_df["Category"] == pref_cat]
            prod_idx = np.random.choice(cat_prods.index)
        else:
            prod_idx = np.random.choice(n_prods, p=popularity_weights)
            
        prod = products_df.iloc[prod_idx]
        
        quantity = np.random.choice([1, 1, 1, 2, 2, 3], p=[0.6, 0.2, 0.1, 0.05, 0.03, 0.02])
        days_offset = np.random.randint(0, 365)
        order_date = (start_date + timedelta(days=days_offset, hours=np.random.randint(0, 24))).strftime("%Y-%m-%d %H:%M:%S")
        
        records.append({
            "CustomerID": cust_id,
            "ProductID": prod["ProductID"],
            "ProductName": prod["ProductName"],
            "Category": prod["Category"],
            "Quantity": quantity,
            "Price": prod["Price"],
            "OrderDate": order_date
        })
        
    df = pd.DataFrame(records)
    df.to_csv(SYNTHETIC_CSV_PATH, index=False)
    print(f"Generated {len(df)} synthetic transactions and saved to {SYNTHETIC_CSV_PATH}")
    return df


def load_dataset(custom_file=None):
    """Load dataset from custom uploaded CSV or default synthetic dataset."""
    if custom_file is not None:
        try:
            df = pd.read_csv(custom_file)
            print("Loaded custom dataset uploaded by user.")
        except Exception as e:
            print(f"Error reading custom CSV: {e}")
            return None
    else:
        if not os.path.exists(SYNTHETIC_CSV_PATH):
            df = generate_synthetic_dataset()
        else:
            df = pd.read_csv(SYNTHETIC_CSV_PATH)
            
    df = preprocess_data(df)
    return df


def preprocess_data(df):
    """Standardize, clean, and enrich transaction dataframe."""
    df = df.copy()
    
    # Required columns check
    required_cols = ["CustomerID", "ProductID", "ProductName", "Category", "Quantity", "Price", "OrderDate"]
    for col in required_cols:
        if col not in df.columns:
            # Fallback column rename matching
            col_match = [c for c in df.columns if c.lower() == col.lower()]
            if col_match:
                df.rename(columns={col_match[0]: col}, inplace=True)
            else:
                raise ValueError(f"Missing required dataset column: {col}")
                
    # Data cleaning
    df.dropna(subset=["CustomerID", "ProductID", "Quantity", "Price"], inplace=True)
    df["CustomerID"] = df["CustomerID"].astype(str).str.strip()
    df["ProductID"] = df["ProductID"].astype(str).str.strip()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(1).astype(int)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(100.0).astype(float)
    df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce").fillna(pd.Timestamp.now())
    
    # Calculate Total Spend per transaction
    df["TotalSpend"] = df["Quantity"] * df["Price"]
    
    return df


def get_customer_summary(df, customer_id):
    """Calculate summary statistics for a given Customer ID."""
    cust_df = df[df["CustomerID"] == customer_id]
    if cust_df.empty:
        return None
        
    num_purchases = len(cust_df)
    total_spent = cust_df["TotalSpend"].sum()
    favorite_cat = cust_df["Category"].mode().iloc[0] if not cust_df["Category"].empty else "N/A"
    last_purchase = cust_df["OrderDate"].max().strftime("%Y-%m-%d")
    
    # Purchase frequency (orders per month)
    date_range = (cust_df["OrderDate"].max() - cust_df["OrderDate"].min()).days
    months = max(1, date_range / 30.0)
    freq = round(num_purchases / months, 1)
    
    return {
        "CustomerID": customer_id,
        "PurchasesCount": num_purchases,
        "TotalSpent": total_spent,
        "FavoriteCategory": favorite_cat,
        "LastPurchaseDate": last_purchase,
        "PurchaseFrequency": f"{freq} orders/month"
    }


def prepare_ncf_dataset(df, num_negatives=4):
    """
    Prepare customer-product interaction dataset for Neural Collaborative Filtering (NCF).
    Includes label encoding and negative sampling.
    """
    # Unique encoders
    cust_encoder = LabelEncoder()
    prod_encoder = LabelEncoder()
    
    df["user_idx"] = cust_encoder.fit_transform(df["CustomerID"])
    df["item_idx"] = prod_encoder.fit_transform(df["ProductID"])
    
    num_users = len(cust_encoder.classes_)
    num_items = len(prod_encoder.classes_)
    
    # Group positive interaction pairs (user, item)
    pos_pairs = set(zip(df["user_idx"], df["item_idx"]))
    
    users, items, labels = [], [], []
    
    # Positive samples (label = 1)
    for u, i in pos_pairs:
        users.append(u)
        items.append(i)
        labels.append(1.0)
        
        # Negative sampling (label = 0)
        for _ in range(num_negatives):
            neg_item = np.random.randint(0, num_items)
            while (u, neg_item) in pos_pairs:
                neg_item = np.random.randint(0, num_items)
            users.append(u)
            items.append(neg_item)
            labels.append(0.0)
            
    X_user = np.array(users, dtype=np.int32)
    X_item = np.array(items, dtype=np.int32)
    y = np.array(labels, dtype=np.float32)
    
    # Train / Test split
    X_user_train, X_user_test, X_item_train, X_item_test, y_train, y_test = train_test_split(
        X_user, X_item, y, test_size=0.2, random_state=42, stratify=y
    )
    
    meta = {
        "num_users": num_users,
        "num_items": num_items,
        "cust_encoder": cust_encoder,
        "prod_encoder": prod_encoder,
        "positive_pairs": pos_pairs
    }
    
    return (X_user_train, X_item_train, y_train), (X_user_test, X_item_test, y_test), meta

import streamlit as st

def apply_custom_css():
    """Inject modern e-commerce visual theme and responsive CSS styling."""
    st.markdown("""
    <style>
    /* Global Styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 95%;
    }
    
    /* Header Gradient Banner */
    .app-header {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .app-header h1 {
        color: #FFFFFF !important;
        font-weight: 800;
        font-size: 2.2rem;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .app-header p {
        color: #C7D2FE !important;
        margin-top: 0.4rem;
        font-size: 1.05rem;
        margin-bottom: 0;
    }
    
    /* KPI Card Component */
    .kpi-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .kpi-title {
        color: #94A3B8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        color: #F8FAFC;
        font-size: 1.75rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .kpi-subtext {
        color: #38BDF8;
        font-size: 0.85rem;
        margin-top: 0.3rem;
        font-weight: 500;
    }

    /* Product Card Component */
    .product-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
        transition: all 0.25s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .product-card:hover {
        border-color: #6366F1;
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.25);
        transform: translateY(-4px);
    }
    .product-badge {
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        font-weight: 700;
        font-size: 0.8rem;
        padding: 0.3rem 0.65rem;
        border-radius: 20px;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
    }
    .category-tag {
        background: #334155;
        color: #CBD5E1;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        display: inline-block;
        margin-top: 0.4rem;
    }
    .product-title {
        color: #F8FAFC;
        font-weight: 700;
        font-size: 1.1rem;
        margin-top: 0.6rem;
        margin-bottom: 0.4rem;
        line-height: 1.3;
    }
    .product-price {
        color: #38BDF8;
        font-weight: 700;
        font-size: 1.25rem;
        margin-bottom: 0.5rem;
    }
    .product-reason {
        background: rgba(99, 102, 241, 0.12);
        border-left: 3px solid #6366F1;
        color: #E0E7FF;
        font-size: 0.85rem;
        padding: 0.5rem 0.75rem;
        border-radius: 4px;
        margin-top: 0.6rem;
        line-height: 1.4;
    }
    
    /* Section Title */
    .section-title {
        color: #F8FAFC;
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border-bottom: 2px solid #334155;
        padding-bottom: 0.5rem;
    }
    
    /* Customer Info Box */
    .customer-box {
        background: linear-gradient(135deg, #0F172A, #1E293B);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
    }
    .customer-box h3 {
        color: #818CF8;
        margin-top: 0;
        font-size: 1.2rem;
    }
    
    /* Custom Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    </style>
    """, unsafe_allow_html=True)


def format_currency(value):
    """Format numeric values as Indian Rupees currency."""
    if value is None:
        return "₹0"
    return f"₹{value:,.2f}"


def render_kpi_card(title, value, subtext="", icon="📊"):
    """Render a modern dashboard KPI card."""
    return f"""
    <div class="kpi-card">
        <div class="kpi-title">{icon} {title}</div>
        <div class="kpi-value">{value}</div>
        {'<div class="kpi-subtext">' + subtext + '</div>' if subtext else ''}
    </div>
    """


def render_product_card(product_name, category, price, score=None, reason=None, image_icon="📦"):
    """Render an e-commerce product card with recommendation score and explanation."""
    score_html = f'<span class="product-badge">⚡ {score}% Match</span>' if score is not None else ''
    reason_html = f'<div class="product-reason">💡 <b>Why:</b> {reason}</div>' if reason else ''
    
    card_html = f"""
    <div class="product-card">
        <div>
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <span style="font-size: 2rem;">{image_icon}</span>
                {score_html}
            </div>
            <span class="category-tag">{category}</span>
            <div class="product-title">{product_name}</div>
            <div class="product-price">{format_currency(price)}</div>
        </div>
        {reason_html}
    </div>
    """
    return card_html

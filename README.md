# AI-Powered Personalized E-Commerce Recommendation System Using Deep Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.15+-FF6F00.svg)](https://tensorflow.org/)
[![Render Deployable](https://img.shields.io/badge/deploy-Render-46E3B7.svg)](https://render.com)

A deep learning-based e-commerce recommendation application built using **Neural Collaborative Filtering (NCF)** with **TensorFlow/Keras**, **Streamlit**, **Pandas**, **NumPy**, **Scikit-learn**, and **Plotly**.

---

## 🌟 Key Features

1. **Neural Collaborative Filtering (NCF) Model**:
   - Customer and Product embedding layers (32-dimensional continuous latent space).
   - Multi-Layer Perceptron (MLP) deep architecture (128 → 64 → 32 → 1 Sigmoid output).
   - Implicit feedback training with negative sampling.
2. **Personalized Product Recommendations**:
   - Predicts purchase probabilities for unpurchased items per customer.
   - Natural language Explainable AI (XAI) justifications ("Why recommended?").
3. **Frequently Bought Together (Co-occurrence Bundles)**:
   - Product association rules computed from historical order co-purchases.
4. **Trending Products Engine**:
   - Sales velocity and growth rate algorithm to identify currently trending catalog items.
5. **Interactive Executive Analytics**:
   - Daily/Monthly sales trajectories, category revenue distribution, top-selling products, and customer purchase frequency histograms using Plotly.
6. **Model Evaluation Suite**:
   - Interactive training loss curve, Precision@5, Recall@5, Hit Rate@5, and Top-K accuracy.
7. **Custom & Synthetic Data Handling**:
   - Includes automatic synthetic data generator (500+ customers, 100+ products, 5,000+ orders) and custom CSV upload support.

---

## 🏗️ Project Architecture

```
NNDL Project/
├── app.py                     # Main Streamlit UI & Navigation
├── data_processing.py         # Data generator, CSV cleaner, encoders & negative sampling
├── model.py                   # TensorFlow NCF Deep Learning Model & Evaluation Metrics
├── recommendation.py          # Candidate scoring, XAI explanations, bundles, trending
├── analytics.py               # Plotly interactive chart suite
├── utils.py                   # Custom CSS e-commerce card styling & metric formatting
├── requirements.txt           # Python dependency specifications
├── render.yaml                # Render Blueprint deployment configuration
├── Procfile                   # Cloud web service startup command
├── .streamlit/
│   └── config.toml            # Streamlit theme & port configuration
└── data/                      # Cached synthetic sales dataset
```

---

## 🧮 Neural Collaborative Filtering Formula

The prediction score $\hat{y}_{u,i}$ for Customer $u$ and Product $i$ is calculated as:

$$p_u = \text{Embedding}_{user}(u) \in \mathbb{R}^d$$

$$q_i = \text{Embedding}_{item}(i) \in \mathbb{R}^d$$

$$z_1 = [p_u \parallel q_i]$$

$$z_2 = \text{ReLU}(W_1 z_1 + b_1)$$

$$z_3 = \text{ReLU}(W_2 z_2 + b_2)$$

$$z_4 = \text{ReLU}(W_3 z_3 + b_3)$$

$$\hat{y}_{u,i} = \sigma(W_{out} z_4 + b_{out})$$

Where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the Sigmoid activation function yielding a probability score between $0.0$ and $1.0$.

---

## 💻 Local Setup Instructions

### 1. Clone or Open Workspace
```bash
cd "c:/Users/Hp/OneDrive/Desktop/NNDL Project"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit Application
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## ☁️ How to Deploy on Render (Step-by-Step)

1. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit for E-Commerce Recommendation System"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
   git push -u origin main
   ```

2. **Deploy on Render**:
   - Go to [Render Dashboard](https://dashboard.render.com/).
   - Click **New +** → **Web Service**.
   - Connect your GitHub repository.
   - Render will automatically pick up `render.yaml` or you can select:
     - **Environment**: Python
     - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
     - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
   - Click **Create Web Service**.

Your application will be live on Render with an automatic `.onrender.com` URL!

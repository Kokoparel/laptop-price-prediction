import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(
    page_title="Prediksi Harga Laptop",
    page_icon="💻",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    try:
        with open('laptop_price_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("Model file not found! Please run 'train_model.py' first.")
        st.stop()

model_data = load_model()
model = model_data['model']
encoders = model_data['encoders']
stats = model_data['stats']
feature_ranges = model_data['feature_ranges']

# Title
st.title("💻 Aplikasi Prediksi Harga Laptop")
st.markdown("### Menggunakan Linear Regression")

# Sidebar for navigation
page = st.sidebar.selectbox("Pilih Halaman", ["Prediksi", "Informasi Model"])

if page == "Prediksi":
    st.header("🔮 Prediksi Harga Laptop")
    st.markdown("Masukkan spesifikasi laptop untuk memprediksi harganya")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Spesifikasi Hardware")
        
        # RAM
        ram = st.slider(
            "RAM (GB)", 
            min_value=int(feature_ranges['ram_min']),
            max_value=int(feature_ranges['ram_max']),
            value=8,
            step=2
        )
        
        # Storage
        storage = st.slider(
            "Storage (GB)", 
            min_value=int(feature_ranges['storage_min']),
            max_value=int(feature_ranges['storage_max']),
            value=256,
            step=128
        )
        
        # Screen Size
        screen_size = st.slider(
            "Ukuran Layar (inches)", 
            min_value=float(feature_ranges['screen_min']),
            max_value=float(feature_ranges['screen_max']),
            value=15.6,
            step=0.1
        )
        
        # Weight
        weight = st.slider(
            "Berat (kg)", 
            min_value=float(feature_ranges['weight_min']),
            max_value=float(feature_ranges['weight_max']),
            value=2.0,
            step=0.1
        )
    
    with col2:
        st.subheader("Spesifikasi Lainnya")
        
        # Company
        companies = encoders['company'].classes_.tolist()
        company = st.selectbox("Merek", sorted(companies))
        company_encoded = encoders['company'].transform([company])[0]
        
        # Type
        types = encoders['typename'].classes_.tolist()
        typename = st.selectbox("Tipe Laptop", sorted(types))
        typename_encoded = encoders['typename'].transform([typename])[0]
        
        # CPU - Show ALL options
        cpus = encoders['cpu'].classes_.tolist()
        cpu = st.selectbox("Processor (CPU)", sorted(cpus))
        cpu_encoded = encoders['cpu'].transform([cpu])[0]
        
        # GPU - Show ALL options
        gpus = encoders['gpu'].classes_.tolist()
        gpu = st.selectbox("Graphics Card (GPU)", sorted(gpus))
        gpu_encoded = encoders['gpu'].transform([gpu])[0]
        
        # OS
        operating_systems = encoders['os'].classes_.tolist()
        os_sys = st.selectbox("Sistem Operasi", sorted(operating_systems))
        os_encoded = encoders['os'].transform([os_sys])[0]
    
    # Prediction button
    st.markdown("---")
    if st.button("🎯 Prediksi Harga", type="primary", use_container_width=True):
        # Prepare input data
        input_data = pd.DataFrame({
            'Ram_GB': [ram],
            'Storage_GB': [storage],
            'Screen_Size': [screen_size],
            'Weight_kg': [weight],
            'Company_Encoded': [company_encoded],
            'TypeName_Encoded': [typename_encoded],
            'Cpu_Encoded': [cpu_encoded],
            'Gpu_Encoded': [gpu_encoded],
            'OpSys_Encoded': [os_encoded]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display result
        st.markdown("---")
        st.success("✅ Prediksi Berhasil!")
        
        # Create 3 columns for results
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.metric("Harga Prediksi (Euro)", f"€{prediction:,.2f}")
        
        with res_col2:
            # Convert to USD (approximate rate)
            usd_price = prediction * 1.1
            st.metric("Harga Prediksi (USD)", f"${usd_price:,.2f}")
        
        with res_col3:
            # Convert to IDR (approximate rate)
            idr_price = prediction * 17000
            st.metric("Harga Prediksi (IDR)", f"Rp{idr_price:,.0f}")
        
        # Display input summary
        st.markdown("### 📋 Ringkasan Spesifikasi")
        specs_df = pd.DataFrame({
            'Spesifikasi': [
                'Merek', 'Tipe', 'RAM', 'Storage', 'Layar', 
                'Berat', 'Processor', 'GPU', 'OS'
            ],
            'Nilai': [
                company, typename, f"{ram} GB", f"{storage} GB", 
                f"{screen_size} inches", f"{weight} kg", cpu, gpu, os_sys
            ]
        })
        st.dataframe(specs_df, use_container_width=True, hide_index=True)
        
        # Visualization
        st.markdown("### 📊 Kontribusi Fitur terhadap Harga")
        
        # Calculate feature contributions
        coefficients = model.coef_
        contributions = input_data.values[0] * coefficients
        
        fig, ax = plt.subplots(figsize=(10, 6))
        feature_names = ['RAM', 'Storage', 'Screen', 'Weight', 'Brand', 'Type', 'CPU', 'GPU', 'OS']
        colors = ['#FF6B6B' if c < 0 else '#4ECDC4' for c in contributions]
        
        bars = ax.barh(feature_names, contributions, color=colors)
        ax.set_xlabel('Kontribusi terhadap Harga (€)', fontsize=12)
        ax.set_title('Pengaruh Setiap Fitur terhadap Prediksi Harga', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        st.pyplot(fig)

else:  # Informasi Model page
    st.header("📊 Informasi Model")
    
    # Model performance
    st.subheader("Performa Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Training Set")
        st.metric("R² Score", f"{stats['train_r2']:.4f}")
        st.metric("RMSE", f"€{stats['train_rmse']:,.2f}")
        st.metric("MAE", f"€{stats['train_mae']:,.2f}")
    
    with col2:
        st.markdown("#### Testing Set")
        st.metric("R² Score", f"{stats['test_r2']:.4f}")
        st.metric("RMSE", f"€{stats['test_rmse']:,.2f}")
        st.metric("MAE", f"€{stats['test_mae']:,.2f}")
    
    # Model explanation
    st.markdown("---")
    st.subheader("Tentang Model")
    
    st.markdown("""
    **Linear Regression** adalah algoritma machine learning yang digunakan untuk memprediksi 
    nilai kontinu berdasarkan hubungan linear antara variabel input dan output.
    
    #### Metrik Evaluasi:
    - **R² Score**: Mengukur seberapa baik model menjelaskan variasi data (0-1, semakin tinggi semakin baik)
    - **RMSE** (Root Mean Squared Error): Rata-rata kesalahan prediksi dalam satuan harga
    - **MAE** (Mean Absolute Error): Rata-rata absolut kesalahan prediksi
    
    #### Fitur yang Digunakan:
    1. RAM (GB)
    2. Storage (GB)
    3. Ukuran Layar (inches)
    4. Berat (kg)
    5. Merek Laptop
    6. Tipe Laptop
    7. Processor (CPU)
    8. Graphics Card (GPU)
    9. Sistem Operasi
    """)
    
    # Feature importance
    st.markdown("---")
    st.subheader("Koefisien Fitur")
    st.markdown("Nilai koefisien menunjukkan pengaruh setiap fitur terhadap harga:")
    
    feature_names = ['RAM', 'Storage', 'Layar', 'Berat', 'Merek', 'Tipe', 'CPU', 'GPU', 'OS']
    coef_df = pd.DataFrame({
        'Fitur': feature_names,
        'Koefisien': model.coef_
    }).sort_values('Koefisien', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#FF6B6B' if c < 0 else '#4ECDC4' for c in coef_df['Koefisien']]
    ax.barh(coef_df['Fitur'], coef_df['Koefisien'], color=colors)
    ax.set_xlabel('Koefisien', fontsize=12)
    ax.set_title('Koefisien Linear Regression untuk Setiap Fitur', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    st.pyplot(fig)
    
    st.dataframe(coef_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>💻 Aplikasi Prediksi Harga Laptop | Powered by Linear Regression</p>
</div>
""", unsafe_allow_html=True)
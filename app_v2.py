import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="Prediksi Harga Laptop v2.0",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        with open('laptop_price_model_v2.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("❌ Model file not found! Please run 'train_model_v2.py' first.")
        st.stop()

model_data = load_model()
model = model_data['model']
scaler = model_data['scaler']
feature_columns = model_data['feature_columns']
encoders = model_data['encoders']
feature_ranges = model_data['feature_ranges']
best_model_name = model_data['best_model_name']
all_results = model_data['all_results']

# Title
st.markdown('<p class="main-header">💻 Prediksi Harga Laptop v2.0</p>', unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: #666;'>Powered by <b>{best_model_name}</b> | "
            f"Akurasi: <b>{all_results[best_model_name]['test_r2']*100:.1f}%</b></p>", 
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/979/979585.png", width=100)
    st.title("Navigation")
    page = st.radio("Pilih Halaman:", ["🔮 Prediksi", "📊 Perbandingan Model", "ℹ️ Info Model"])
    
    st.markdown("---")
    st.markdown("### 🎯 Quick Stats")
    st.metric("Model Terbaik", best_model_name)
    st.metric("Test R² Score", f"{all_results[best_model_name]['test_r2']:.4f}")
    st.metric("Test RMSE", f"€{all_results[best_model_name]['test_rmse']:.2f}")

# Helper functions
def extract_cpu_info(cpu):
    cpu_lower = cpu.lower()
    if 'intel' in cpu_lower:
        brand = 1
    elif 'amd' in cpu_lower:
        brand = 2
    else:
        brand = 0
    
    if 'i7' in cpu_lower or 'ryzen 7' in cpu_lower:
        tier = 4
    elif 'i5' in cpu_lower or 'ryzen 5' in cpu_lower:
        tier = 3
    elif 'i3' in cpu_lower or 'ryzen 3' in cpu_lower:
        tier = 2
    elif 'celeron' in cpu_lower or 'pentium' in cpu_lower:
        tier = 1
    else:
        tier = 0
    return brand, tier

def categorize_gpu(gpu):
    gpu_lower = gpu.lower()
    if 'nvidia' in gpu_lower and 'gtx' in gpu_lower:
        return 3
    elif 'nvidia' in gpu_lower or 'amd radeon pro' in gpu_lower:
        return 2
    elif 'intel iris' in gpu_lower or 'intel uhd' in gpu_lower:
        return 1
    else:
        return 0

def categorize_brand(company):
    premium = ['Apple', 'Microsoft', 'Razer', 'MSI']
    if company in premium:
        return 2
    else:
        return 1

def categorize_os(os):
    os_lower = os.lower()
    if 'mac' in os_lower:
        return 3
    elif 'windows 10' in os_lower:
        return 2
    elif 'windows' in os_lower:
        return 1
    else:
        return 0

# PAGE 1: PREDICTION
if page == "🔮 Prediksi":
    st.header("🔮 Prediksi Harga Laptop")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("💾 Hardware Utama")
        ram = st.slider("RAM (GB)", 
                       min_value=int(feature_ranges['ram_min']),
                       max_value=int(feature_ranges['ram_max']),
                       value=8, step=2)
        
        storage = st.slider("Storage (GB)", 
                           min_value=int(feature_ranges['storage_min']),
                           max_value=int(feature_ranges['storage_max']),
                           value=256, step=128)
        
        is_ssd = st.selectbox("Tipe Storage", ["SSD", "HDD"])
        is_ssd_val = 1 if is_ssd == "SSD" else 0
    
    with col2:
        st.subheader("🖥️ Display & Desain")
        screen_size = st.slider("Ukuran Layar (inches)", 
                               min_value=float(feature_ranges['screen_min']),
                               max_value=float(feature_ranges['screen_max']),
                               value=15.6, step=0.1)
        
        screen_res = st.selectbox("Resolusi Layar", 
                                 ["HD (1366x768)", "Full HD (1920x1080)", 
                                  "QHD/Retina (2560x1600)", "4K (3840x2160)"])
        screen_res_map = {"HD (1366x768)": 1, "Full HD (1920x1080)": 2, 
                         "QHD/Retina (2560x1600)": 3, "4K (3840x2160)": 4}
        screen_res_val = screen_res_map[screen_res]
        
        weight = st.slider("Berat (kg)", 
                          min_value=float(feature_ranges['weight_min']),
                          max_value=float(feature_ranges['weight_max']),
                          value=2.0, step=0.1)
    
    with col3:
        st.subheader("⚙️ Spesifikasi Lainnya")
        companies = sorted(encoders['company'].classes_.tolist())
        company = st.selectbox("Merek", companies)
        
        types = sorted(encoders['typename'].classes_.tolist())
        typename = st.selectbox("Tipe Laptop", types)
        
        cpus = sorted(encoders['cpu'].classes_.tolist())
        cpu = st.selectbox("Processor", cpus)
        
        gpus = sorted(encoders['gpu'].classes_.tolist())
        gpu = st.selectbox("Graphics Card", gpus)
        
        oss = sorted(encoders['os'].classes_.tolist())
        os_sys = st.selectbox("Sistem Operasi", oss)
    
    # Prediction button
    if st.button("🎯 PREDIKSI HARGA", type="primary", use_container_width=True):
        with st.spinner("Memproses prediksi..."):
            # Extract features
            cpu_brand, cpu_tier = extract_cpu_info(cpu)
            gpu_type = categorize_gpu(gpu)
            brand_tier = categorize_brand(company)
            os_category = categorize_os(os_sys)
            
            # Calculate derived features
            ram_storage_ratio = ram / (storage + 1)
            performance_score = ram * cpu_tier * (is_ssd_val + 1)
            
            # Prepare input
            input_data = pd.DataFrame({
                'Ram_GB': [ram],
                'Storage_GB': [storage],
                'is_SSD': [is_ssd_val],
                'Screen_Size': [screen_size],
                'Screen_Resolution_Category': [screen_res_val],
                'Weight_kg': [weight],
                'CPU_Brand': [cpu_brand],
                'CPU_Tier': [cpu_tier],
                'GPU_Type': [gpu_type],
                'Brand_Tier': [brand_tier],
                'OS_Category': [os_category],
                'RAM_Storage_Ratio': [ram_storage_ratio],
                'Performance_Score': [performance_score]
            })
            
            # Scale if Linear Regression
            if best_model_name == 'Linear Regression':
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]
            else:
                prediction = model.predict(input_data)[0]
            
            # Calculate confidence interval (approximation)
            test_rmse = all_results[best_model_name]['test_rmse']
            lower_bound = max(0, prediction - test_rmse)
            upper_bound = prediction + test_rmse
            
            st.success("✅ Prediksi Berhasil!")
            
            # Display results
            st.markdown("---")
            col_r1, col_r2, col_r3 = st.columns(3)
            
            with col_r1:
                st.metric("💶 Harga (Euro)", f"€{prediction:,.2f}",
                         delta=f"±€{test_rmse:.0f}")
            with col_r2:
                usd_price = prediction * 1.1
                st.metric("💵 Harga (USD)", f"${usd_price:,.2f}",
                         delta=f"±${test_rmse*1.1:.0f}")
            with col_r3:
                idr_price = prediction * 17000
                st.metric("💴 Harga (IDR)", f"Rp{idr_price:,.0f}",
                         delta=f"±Rp{test_rmse*17000:,.0f}")
            
            # Confidence interval
            st.info(f"📊 **Rentang Prediksi (95% confidence):** €{lower_bound:,.2f} - €{upper_bound:,.2f}")
            
            # Specifications summary
            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                st.markdown("### 📋 Ringkasan Spesifikasi")
                specs_df = pd.DataFrame({
                    'Komponen': ['Merek', 'Tipe', 'Processor', 'RAM', 'Storage'],
                    'Detail': [company, typename, cpu, f"{ram} GB", f"{storage} GB {is_ssd}"]
                })
                st.dataframe(specs_df, use_container_width=True, hide_index=True)
            
            with col_s2:
                st.markdown("### 🎮 Detail Lainnya")
                other_df = pd.DataFrame({
                    'Komponen': ['Graphics', 'OS', 'Layar', 'Resolusi', 'Berat'],
                    'Detail': [gpu, os_sys, f"{screen_size}\"", screen_res, f"{weight} kg"]
                })
                st.dataframe(other_df, use_container_width=True, hide_index=True)
            
            # Feature importance visualization (for tree models)
            if best_model_name in ['Random Forest', 'Gradient Boosting']:
                st.markdown("### 📊 Feature Importance")
                
                importances = model.feature_importances_
                feat_imp_df = pd.DataFrame({
                    'Feature': feature_columns,
                    'Importance': importances
                }).sort_values('Importance', ascending=True)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = plt.cm.viridis(feat_imp_df['Importance'] / feat_imp_df['Importance'].max())
                ax.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color=colors)
                ax.set_xlabel('Importance Score', fontsize=12)
                ax.set_title(f'Feature Importance - {best_model_name}', 
                           fontsize=14, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                st.pyplot(fig)

# PAGE 2: MODEL COMPARISON
elif page == "📊 Perbandingan Model":
    st.header("📊 Perbandingan Performa Model")
    
    # Metrics comparison
    metrics_data = []
    for name, result in all_results.items():
        metrics_data.append({
            'Model': name,
            'Test R²': result['test_r2'],
            'Test RMSE': result['test_rmse'],
            'Test MAE': result['test_mae'],
            'CV R² Mean': result['cv_r2_mean']
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 R² Score Comparison")
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#2ecc71' if m == best_model_name else '#95a5a6' 
                 for m in metrics_df['Model']]
        bars = ax.barh(metrics_df['Model'], metrics_df['Test R²'], color=colors)
        ax.set_xlabel('R² Score', fontsize=12)
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, metrics_df['Test R²'])):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.4f}', va='center', fontweight='bold')
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("📉 RMSE Comparison")
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#e74c3c' if m == best_model_name else '#95a5a6' 
                 for m in metrics_df['Model']]
        bars = ax.barh(metrics_df['Model'], metrics_df['Test RMSE'], color=colors)
        ax.set_xlabel('RMSE (€)', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, metrics_df['Test RMSE'])):
            ax.text(val + 5, bar.get_y() + bar.get_height()/2, 
                   f'€{val:.2f}', va='center', fontweight='bold')
        
        st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("📋 Detailed Metrics Table")
    
    # Style the dataframe
    styled_df = metrics_df.style.highlight_max(
        subset=['Test R²', 'CV R² Mean'], 
        color='lightgreen'
    ).highlight_min(
        subset=['Test RMSE', 'Test MAE'], 
        color='lightgreen'
    ).format({
        'Test R²': '{:.4f}',
        'Test RMSE': '€{:.2f}',
        'Test MAE': '€{:.2f}',
        'CV R² Mean': '{:.4f}'
    })
    
    st.dataframe(styled_df, use_container_width=True)
    
    st.info(f"🏆 **Model Terbaik:** {best_model_name} dengan Test R² = {all_results[best_model_name]['test_r2']:.4f}")

# PAGE 3: MODEL INFO
else:
    st.header("ℹ️ Informasi Model")
    
    tab1, tab2, tab3 = st.tabs(["📖 Tentang Model", "🔧 Feature Engineering", "📚 Metodologi"])
    
    with tab1:
        st.markdown(f"""
        ### Model yang Digunakan: **{best_model_name}**
        
        Model ini dipilih setelah membandingkan 3 algoritma berbeda:
        - Linear Regression (baseline)
        - Random Forest
        - Gradient Boosting
        
        #### Performa Model Terpilih:
        - **Test R² Score:** {all_results[best_model_name]['test_r2']:.4f} 
          ({all_results[best_model_name]['test_r2']*100:.1f}% variasi harga dapat dijelaskan)
        - **Test RMSE:** €{all_results[best_model_name]['test_rmse']:.2f} 
          (rata-rata error prediksi)
        - **Test MAE:** €{all_results[best_model_name]['test_mae']:.2f} 
          (rata-rata absolut error)
        - **Cross-Validation R²:** {all_results[best_model_name]['cv_r2_mean']:.4f} ± {all_results[best_model_name]['cv_r2_std']:.4f}
        
        #### Interpretasi:
        Model dapat memprediksi harga laptop dengan tingkat akurasi **{all_results[best_model_name]['test_r2']*100:.1f}%**. 
        Rata-rata kesalahan prediksi adalah sekitar **€{all_results[best_model_name]['test_rmse']:.0f}**.
        """)
    
    with tab2:
        st.markdown("""
        ### Feature Engineering yang Diterapkan:
        
        Model ini menggunakan **13 fitur** yang telah diproses secara advanced:
        
        #### 1. **Hardware Features:**
        - `Ram_GB`: Kapasitas RAM dalam GB
        - `Storage_GB`: Kapasitas storage dalam GB
        - `is_SSD`: Binary indicator (1=SSD, 0=HDD)
        
        #### 2. **Display Features:**
        - `Screen_Size`: Ukuran layar dalam inches
        - `Screen_Resolution_Category`: Kategori resolusi (1=HD, 2=Full HD, 3=QHD, 4=4K)
        
        #### 3. **Physical Features:**
        - `Weight_kg`: Berat laptop dalam kilogram
        
        #### 4. **CPU Features:**
        - `CPU_Brand`: Brand processor (1=Intel, 2=AMD)
        - `CPU_Tier`: Tier processor (4=i7/Ryzen7, 3=i5/Ryzen5, 2=i3/Ryzen3, 1=Celeron)
        
        #### 5. **GPU Features:**
        - `GPU_Type`: Tipe GPU (3=High-end, 2=Dedicated, 1=Integrated Good, 0=Basic)
        
        #### 6. **Brand & OS:**
        - `Brand_Tier`: Tier merek (2=Premium, 1=Mid-range)
        - `OS_Category`: Kategori OS (3=macOS, 2=Win10, 1=Other Win/Linux, 0=No OS)
        
        #### 7. **Derived Features:**
        - `RAM_Storage_Ratio`: Rasio RAM terhadap Storage
        - `Performance_Score`: Skor performa = RAM × CPU_Tier × (is_SSD + 1)
        
        """)
    
    with tab3:
        st.markdown("""
        ### Metodologi Training:
        
        #### 1. **Data Preprocessing:**
        - Cleaning: Remove duplicates & handle missing values
        - Feature extraction dari text (CPU, GPU, Memory, dll)
        - Advanced feature engineering (13 features)
        
        #### 2. **Model Training:**
        - Train-Test Split: 80% training, 20% testing
        - Feature scaling untuk Linear Regression
        - 5-fold Cross-Validation untuk validasi robustness
        
        #### 3. **Model Selection:**
        - Compare 3 algorithms
        - Select best model based on Test R² score
        - Validate with cross-validation
        
        #### 4. **Evaluation Metrics:**
        - **R² Score**: Proportion of variance explained
        - **RMSE**: Root Mean Squared Error (in Euros)
        - **MAE**: Mean Absolute Error (in Euros)
        
        #### 5. **Dataset:**
        - Source: Kaggle Laptop Price Dataset
        - Total samples: 1,303 laptops
        - Price range: €174.99 - €6,099.00
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💻 Aplikasi Prediksi Harga Laptop v2.0 | Advanced ML Model</p>
    <p>Powered by {model} • Built with Streamlit</p>
</div>
""".format(model=best_model_name), unsafe_allow_html=True)
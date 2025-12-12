import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("LAPTOP PRICE PREDICTION - ADVANCED MODEL TRAINING")
print("="*70)

# Load dataset
print("\n[1/7] Loading dataset...")
df = pd.read_csv('laptop_price.csv', encoding='latin-1')
print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Data preprocessing
print("\n[2/7] Advanced Feature Engineering...")

# Remove duplicates and handle missing values
df = df.drop_duplicates()
df = df.dropna()

# 1. RAM extraction
df['Ram_GB'] = df['Ram'].str.extract(r'(\d+)').astype(float)

# 2. Storage - Detect SSD vs HDD and capacity
df['Storage_GB'] = df['Memory'].str.extract(r'(\d+)').astype(float)
df['is_SSD'] = df['Memory'].str.contains('SSD|Flash', case=False, na=False).astype(int)

# 3. Screen Size
df['Screen_Size'] = df['Inches']

# 4. Screen Resolution Category
def categorize_resolution(res):
    if pd.isna(res):
        return 0
    res_lower = str(res).lower()
    if '3840' in res_lower or '4k' in res_lower:
        return 4  # 4K
    elif '2880' in res_lower or '2560' in res_lower or 'retina' in res_lower:
        return 3  # QHD/Retina
    elif '1920' in res_lower or 'full hd' in res_lower:
        return 2  # Full HD
    elif '1366' in res_lower or '1440' in res_lower:
        return 1  # HD
    else:
        return 0  # Low res

df['Screen_Resolution_Category'] = df['ScreenResolution'].apply(categorize_resolution)

# 5. Weight
df['Weight_kg'] = df['Weight'].str.extract(r'(\d+\.?\d*)').astype(float)

# 6. CPU Brand and Tier
def extract_cpu_info(cpu):
    if pd.isna(cpu):
        return 0, 0
    cpu_lower = str(cpu).lower()
    
    # Brand
    if 'intel' in cpu_lower:
        brand = 1
    elif 'amd' in cpu_lower:
        brand = 2
    else:
        brand = 0
    
    # Tier
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

df['CPU_Brand'], df['CPU_Tier'] = zip(*df['Cpu'].apply(extract_cpu_info))

# 7. GPU Type
def categorize_gpu(gpu):
    if pd.isna(gpu):
        return 0
    gpu_lower = str(gpu).lower()
    
    if 'nvidia' in gpu_lower and 'gtx' in gpu_lower:
        return 3  # High-end dedicated
    elif 'nvidia' in gpu_lower or 'amd radeon pro' in gpu_lower:
        return 2  # Dedicated
    elif 'intel iris' in gpu_lower or 'intel uhd' in gpu_lower:
        return 1  # Integrated (better)
    else:
        return 0  # Basic integrated

df['GPU_Type'] = df['Gpu'].apply(categorize_gpu)

# 8. Brand Tier
def categorize_brand(company):
    premium = ['Apple', 'Microsoft', 'Razer', 'MSI']
    mid = ['Dell', 'HP', 'Lenovo', 'Asus', 'Acer']
    
    if company in premium:
        return 2
    elif company in mid:
        return 1
    else:
        return 0

df['Brand_Tier'] = df['Company'].apply(categorize_brand)

# 9. Operating System Category
def categorize_os(os):
    if pd.isna(os):
        return 0
    os_lower = str(os).lower()
    
    if 'mac' in os_lower:
        return 3  # macOS
    elif 'windows 10' in os_lower:
        return 2  # Windows 10
    elif 'windows' in os_lower:
        return 1  # Other Windows
    elif 'linux' in os_lower:
        return 1
    else:
        return 0  # No OS

df['OS_Category'] = df['OpSys'].apply(categorize_os)

# 10. Interaction Features (must be after all base features are created)
df['RAM_Storage_Ratio'] = df['Ram_GB'] / (df['Storage_GB'] + 1)
df['Performance_Score'] = df['Ram_GB'] * df['CPU_Tier'] * (df['is_SSD'] + 1)

# Label Encoding for original categorical features (for reference)
le_company = LabelEncoder()
le_typename = LabelEncoder()
le_cpu = LabelEncoder()
le_gpu = LabelEncoder()
le_os = LabelEncoder()

df['Company_Encoded'] = le_company.fit_transform(df['Company'])
df['TypeName_Encoded'] = le_typename.fit_transform(df['TypeName'])
df['Cpu_Encoded'] = le_cpu.fit_transform(df['Cpu'])
df['Gpu_Encoded'] = le_gpu.fit_transform(df['Gpu'])
df['OpSys_Encoded'] = le_os.fit_transform(df['OpSys'])

print(f"✓ Feature engineering completed: {df.shape[1]} total features")

# Select features for modeling
feature_columns = [
    'Ram_GB', 'Storage_GB', 'is_SSD', 'Screen_Size', 
    'Screen_Resolution_Category', 'Weight_kg',
    'CPU_Brand', 'CPU_Tier', 'GPU_Type', 'Brand_Tier', 'OS_Category',
    'RAM_Storage_Ratio', 'Performance_Score'
]

X = df[feature_columns]
y = df['Price_euros']

print(f"\n[3/7] Dataset Statistics:")
print(f"  Total samples: {len(X)}")
print(f"  Features used: {len(feature_columns)}")
print(f"  Price range: €{y.min():.2f} - €{y.max():.2f}")
print(f"  Price mean: €{y.mean():.2f}")
print(f"  Price median: €{y.median():.2f}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n[4/7] Data Split:")
print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Testing set: {X_test.shape[0]} samples")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
print(f"\n[5/7] Training Multiple Models...")

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=20, 
                                          min_samples_split=5, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                                   learning_rate=0.1, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n  Training {name}...")
    
    # Use scaled data for Linear Regression, original for tree-based
    if name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                scoring='r2', n_jobs=-1)
    
    results[name] = {
        'model': model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std()
    }
    
    print(f"    ✓ Test R²: {test_r2:.4f}")
    print(f"    ✓ Test RMSE: €{test_rmse:.2f}")
    print(f"    ✓ Test MAE: €{test_mae:.2f}")
    print(f"    ✓ CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Select best model based on test R2
print(f"\n[6/7] Model Comparison:")
print("\n" + "="*70)
print(f"{'Model':<25} {'Test R²':<12} {'Test RMSE':<15} {'Test MAE':<12}")
print("="*70)

for name, result in results.items():
    print(f"{name:<25} {result['test_r2']:<12.4f} €{result['test_rmse']:<14.2f} €{result['test_mae']:<11.2f}")

best_model_name = max(results.items(), key=lambda x: x[1]['test_r2'])[0]
best_model = results[best_model_name]['model']

print("="*70)
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Test R²: {results[best_model_name]['test_r2']:.4f}")
print(f"   Test RMSE: €{results[best_model_name]['test_rmse']:.2f}")
print(f"   Improvement over baseline: {(results[best_model_name]['test_r2'] - results['Linear Regression']['test_r2']) * 100:.2f}%")

# Feature importance (for tree-based models)
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print(f"\n[7/7] Feature Importance ({best_model_name}):")
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\n" + "="*50)
    for idx, row in feature_importance_df.head(10).iterrows():
        print(f"  {row['Feature']:<30} {row['Importance']:.4f}")
    print("="*50)

# Save model and all necessary data
print(f"\n[SAVING] Packaging model and metadata...")

model_data = {
    'best_model_name': best_model_name,
    'model': best_model,
    'scaler': scaler,
    'feature_columns': feature_columns,
    'all_results': results,
    'encoders': {
        'company': le_company,
        'typename': le_typename,
        'cpu': le_cpu,
        'gpu': le_gpu,
        'os': le_os
    },
    'feature_ranges': {
        'ram_min': X['Ram_GB'].min(),
        'ram_max': X['Ram_GB'].max(),
        'storage_min': X['Storage_GB'].min(),
        'storage_max': X['Storage_GB'].max(),
        'screen_min': X['Screen_Size'].min(),
        'screen_max': X['Screen_Size'].max(),
        'weight_min': X['Weight_kg'].min(),
        'weight_max': X['Weight_kg'].max()
    }
}

with open('laptop_price_model_v2.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("✓ Model saved successfully as 'laptop_price_model_v2.pkl'")

print("\n" + "="*70)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*70)
print("\n📊 Summary:")
print(f"  ✓ Best Model: {best_model_name}")
print(f"  ✓ Test R² Score: {results[best_model_name]['test_r2']:.4f}")
print(f"  ✓ Test RMSE: €{results[best_model_name]['test_rmse']:.2f}")
print(f"  ✓ Test MAE: €{results[best_model_name]['test_mae']:.2f}")
print(f"\n🚀 Run the Streamlit app: streamlit run app_v2.py")
print("="*70)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("Loading dataset...")
df = pd.read_csv('laptop_price.csv', encoding='latin-1')

print(f"Dataset shape: {df.shape}")
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Data preprocessing
print("\n" + "="*50)
print("PREPROCESSING DATA")
print("="*50)

# Menangani missing values
df = df.dropna()

# Extract features dari kolom yang ada
# Biasanya dataset laptop memiliki kolom seperti: Company, TypeName, Ram, Weight, etc.

# Feature Engineering untuk RAM (extract angka dari string)
if 'Ram' in df.columns:
    df['Ram_GB'] = df['Ram'].str.extract(r'(\d+)').astype(float)
else:
    print("Warning: Column 'Ram' not found")

# Feature Engineering untuk Storage/Memory
if 'Memory' in df.columns:
    # Extract primary storage capacity
    df['Storage_GB'] = df['Memory'].str.extract(r'(\d+)').astype(float)
else:
    print("Warning: Column 'Memory' not found")

# Feature Engineering untuk Screen Size
if 'Inches' in df.columns:
    df['Screen_Size'] = df['Inches']
elif 'ScreenResolution' in df.columns:
    # Extract dari resolution jika ada
    df['Screen_Size'] = 15.6  # default value
else:
    df['Screen_Size'] = 15.6  # default value

# Feature Engineering untuk Weight
if 'Weight' in df.columns:
    df['Weight_kg'] = df['Weight'].str.extract(r'(\d+\.?\d*)').astype(float)
else:
    df['Weight_kg'] = 2.0  # default value

# Label Encoding untuk categorical variables
le_company = LabelEncoder()
le_typename = LabelEncoder()
le_cpu = LabelEncoder()
le_gpu = LabelEncoder()
le_os = LabelEncoder()

if 'Company' in df.columns:
    df['Company_Encoded'] = le_company.fit_transform(df['Company'])
else:
    df['Company_Encoded'] = 0

if 'TypeName' in df.columns:
    df['TypeName_Encoded'] = le_typename.fit_transform(df['TypeName'])
else:
    df['TypeName_Encoded'] = 0

if 'Cpu' in df.columns:
    df['Cpu_Encoded'] = le_cpu.fit_transform(df['Cpu'])
else:
    df['Cpu_Encoded'] = 0

if 'Gpu' in df.columns:
    df['Gpu_Encoded'] = le_gpu.fit_transform(df['Gpu'])
else:
    df['Gpu_Encoded'] = 0

if 'OpSys' in df.columns:
    df['OpSys_Encoded'] = le_os.fit_transform(df['OpSys'])
else:
    df['OpSys_Encoded'] = 0

# Target variable (Price)
if 'Price_euros' in df.columns:
    y = df['Price_euros']
elif 'Price' in df.columns:
    y = df['Price']
else:
    print("Error: Price column not found!")
    exit()

# Select features
feature_columns = ['Ram_GB', 'Storage_GB', 'Screen_Size', 'Weight_kg', 
                   'Company_Encoded', 'TypeName_Encoded', 'Cpu_Encoded', 
                   'Gpu_Encoded', 'OpSys_Encoded']

X = df[feature_columns]

print(f"\nFeatures selected: {feature_columns}")
print(f"Target: Price")
print(f"\nData shape after preprocessing: {X.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")

# Train model
print("\n" + "="*50)
print("TRAINING MODEL")
print("="*50)

model = LinearRegression()
model.fit(X_train, y_train)

print("Model training completed!")

# Evaluate model
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)

print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)
print(f"\nTraining Set:")
print(f"  R² Score: {train_r2:.4f}")
print(f"  RMSE: {train_rmse:.2f}")
print(f"  MAE: {train_mae:.2f}")

print(f"\nTesting Set:")
print(f"  R² Score: {test_r2:.4f}")
print(f"  RMSE: {test_rmse:.2f}")
print(f"  MAE: {test_mae:.2f}")

# Feature importance (coefficients)
print("\n" + "="*50)
print("FEATURE IMPORTANCE (Coefficients)")
print("="*50)
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)
print(feature_importance)

# Save model and encoders
print("\n" + "="*50)
print("SAVING MODEL")
print("="*50)

model_data = {
    'model': model,
    'feature_columns': feature_columns,
    'encoders': {
        'company': le_company,
        'typename': le_typename,
        'cpu': le_cpu,
        'gpu': le_gpu,
        'os': le_os
    },
    'stats': {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae
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

with open('laptop_price_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model saved successfully as 'laptop_price_model.pkl'")
print("\nYou can now run the Streamlit app: streamlit run app.py")
import pandas as pd
import numpy as np

# Load the dataset
file_path = "apartment_for_rent_train.csv"
df = pd.read_csv(file_path)

# Print initial data inspection
print("\nInitial Data Inspection:")
print("-----------------------")
print("\nFirst few rows of raw data:")
print(df.head())

print("\nColumn dtypes before cleaning:")
for col in df.columns:
    values = df[col].head(3).values
    print(f"{col}: {df[col].dtype} - Sample values: {values}")

# Create a copy for cleaning
df_cleaned = df.copy()

# Step 1: Drop truly irrelevant columns if they exist
columns_to_drop = [col for col in df_cleaned.columns if col.startswith('Unnamed:')]
if columns_to_drop:
    df_cleaned = df_cleaned.drop(columns=columns_to_drop)

# Step 2: Define standardization functions
def standardize_construction_type(value):
    if pd.isna(value):
        return None
    value = str(value).lower().strip()
    mapping = {
        'stone': 'Stone',
        'monolith': 'Monolith',
        'panels': 'Panels',
        'bricks': 'Bricks',
        'cassette': 'Cassette'
    }
    return mapping.get(value, value.title())

def standardize_balcony(value):
    if pd.isna(value):
        return 'Not available'
    value = str(value).lower().strip()
    if value in ['0', 'none', 'not available', '']:
        return 'Not available'
    if value == 'multiple_balconies':
        return 'Multiple balconies'
    if value == 'closed_balcony':
        return 'Closed balcony'
    if value == 'open_balcony':
        return 'Open balcony'
    return value.title()

def standardize_renovation(value):
    if pd.isna(value):
        return None
    value = str(value).lower().strip()
    mapping = {
        'euro_renovation': 'Euro Renovation',
        'designer_renovation': 'Designer Renovation',
        'major_renovation': 'Major Renovation',
        'cosmetic_renovation': 'Cosmetic Renovation',
        'partial_renovation': 'Partial Renovation',
        'old_renovation': 'Old Renovation',
        'no_renovation': 'No Renovation'
    }
    return mapping.get(value, value.title())

print("\nColumn names before cleaning:", df_cleaned.columns.tolist())

# Step 3: Clean and standardize
df_cleaned['Construction_type'] = df_cleaned['Construction_type'].apply(standardize_construction_type)
df_cleaned['Balcony'] = df_cleaned['Balcony'].apply(standardize_balcony)
df_cleaned['Renovation'] = df_cleaned['Renovation'].apply(standardize_renovation)

# Step 4: Convert numeric fields
numeric_fields = {
    'Floor': lambda x: float(x) if pd.notna(x) else None,
    'New_construction': lambda x: 1 if x == 1 else 0,
    'Children_are_welcome': lambda x: 1 if x in [10, 11, 12] else 0,
    'Price': lambda x: float(x) if pd.notna(x) and float(x) >= 0 else None,
    'Ceiling_height': lambda x: float(x) if pd.notna(x) else None  # Removed incorrect division by 10
}

# Price validation function
def validate_price(row):
    if pd.isna(row['Price']) or pd.isna(row['Duration']) or pd.isna(row['Currency']):
        return None
    
    try:
        price = float(row['Price'])
        if price <= 0:  # Only validate that price is positive
            return None
        return price
    except (ValueError, TypeError):
        return None

# Apply price validation after numeric conversion
df_cleaned['Price'] = df_cleaned.apply(validate_price, axis=1)

# Handle missing values in amenity columns consistently
amenity_columns = ['Furniture', 'amenities', 'appliances', 'parking']
for col in amenity_columns:
    df_cleaned[col] = df_cleaned[col].fillna('Not available')
    df_cleaned[col] = df_cleaned[col].replace('none', 'Not available')

# Remove unnecessary columns if they exist
columns_to_remove = ['Original_price', 'Original_currency', 'Original_duration', 'Price_monthly']
df_cleaned = df_cleaned.drop(columns=[col for col in columns_to_remove if col in df_cleaned.columns])

for col, conversion in numeric_fields.items():
    if col in df_cleaned.columns:
        print(f"\nConverting {col}:")
        print("Before conversion (first 5 values):", df_cleaned[col].head().values)
        df_cleaned[col] = df_cleaned[col].apply(conversion)
        print("After conversion (first 5 values):", df_cleaned[col].head().values)

# Step 5: Data validation
print("\nData Validation:")
print("---------------")

# Check for invalid Floor values
invalid_floor = df_cleaned[(df_cleaned['Floor'] <= 0) | (df_cleaned['Floor'] > 50)]
print(f"Found {len(invalid_floor)} records with suspicious floor numbers")

# Check for invalid floor vs building floors
invalid_floors = df_cleaned[df_cleaned['Floor'] > df_cleaned['Floors_in_the_building']]
print(f"Found {len(invalid_floors)} records where floor number exceeds building floors")

# Check for invalid ceiling heights
invalid_height = df_cleaned[(df_cleaned['Ceiling_height'] < 2.0) | (df_cleaned['Ceiling_height'] > 5.0)]
print(f"Found {len(invalid_height)} records with suspicious ceiling height")

# Convert prices to USD (1 AMD = 0.0026 USD)
AMD_TO_USD = 0.0026

def convert_to_usd(row):
    if pd.isna(row['Price']) or pd.isna(row['Currency']) or pd.isna(row['Duration']):
        return None
    
    try:
        price = float(row['Price'])
        if price <= 0:
            return None
            
        # Convert daily prices to monthly
        if row['Duration'].lower() == 'daily':
            price = price * 30
            
        # Convert to USD if needed
        if row['Currency'] == 'AMD':
            price = price * AMD_TO_USD
        elif row['Currency'] == 'USD':
            price = price
        else:
            return None
            
        return price
    except (ValueError, TypeError):
        return None

# Convert all prices to monthly USD
print("\nStandardizing prices to USD:")
print("--------------------------")
df_cleaned['Price_USD'] = df_cleaned.apply(convert_to_usd, axis=1)
print(f"Valid USD prices: {df_cleaned['Price_USD'].notna().sum()} out of {len(df_cleaned)}")

# Print summary statistics
print("\nCleaning Summary:")
print("----------------")
missing = df_cleaned.isnull().sum()
print("\nMissing values by column:")
print(missing[missing > 0])

print("\nSample of cleaned data:")
print(df_cleaned.head())

# Save cleaned data
output_file = 'apartment_for_rent_train_cleaned.csv'
df_cleaned.to_csv(output_file, index=False)
print(f"\nCleaned data saved to '{output_file}'")

# Print some statistics about the cleaned dataset
print("\nDataset Statistics:")
print("-----------------")
print(f"Total number of records: {len(df_cleaned)}")
print(f"Number of columns: {len(df_cleaned.columns)}")
print("\nPrice statistics (USD):")
print(df_cleaned['Price_USD'].describe())

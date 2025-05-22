# ===========================================================================
# Apartment Price Data Cleaning Script
# ===========================================================================
# This script performs comprehensive data cleaning on apartment rental data.
# Key cleaning steps include:
# 1. Initial data loading and inspection
# 2. Standardization of categorical variables
# 3. Numeric field conversions and validations
# 4. Price standardization and currency conversion
# 5. Feature analysis and relationship with price
# 6. Final validation and export

import pandas as pd
import numpy as np

# ===========================================================================
# Data Loading and Initial Inspection
# ===========================================================================
# Load the raw dataset for cleaning and preprocessing
# The dataset contains apartment rental information including prices,
# locations, and various property features
file_path = "apartment_for_rent_train.csv"
df = pd.read_csv(file_path)

# Perform initial data inspection to understand:
# - Data types of each column
# - Sample values to identify potential inconsistencies
# - Any immediate data quality issues
print("\nInitial Data Inspection:")
print("-----------------------")
print("\nFirst few rows of raw data:")
print(df.head())

# Display column data types and sample values to identify:
# - Incorrect data type assignments
# - Inconsistent value formats
# - Potential data entry errors
print("\nColumn dtypes before cleaning:")
for col in df.columns:
    values = df[col].head(3).values
    print(f"{col}: {df[col].dtype} - Sample values: {values}")

# Create a working copy to preserve the original data
df_cleaned = df.copy()

# ===========================================================================
# Initial Data Cleaning
# ===========================================================================
# Remove any unnamed or system-generated columns that don't contain useful information
columns_to_drop = [col for col in df_cleaned.columns if col.startswith('Unnamed:')]
if columns_to_drop:
    df_cleaned = df_cleaned.drop(columns=columns_to_drop)

# =================================================================
# Data Cleaning and Transformation
# =================================================================
# Purpose: Clean the data by standardizing formats, correcting data types,
# handling missing values, and removing or correcting invalid data.

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

# =================================================================
# Data Validation
# =================================================================
# Purpose: Validate the cleaned data to ensure accuracy and consistency.
# This includes checking for invalid values, outliers, and ensuring
# that all transformations have been correctly applied.

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

# =================================================================
# Categorical Variables Analysis
# =================================================================
# Purpose: Analyze categorical variables and their relationship with
# the target variable (Price_USD). This helps in understanding
# how different categories impact pricing.

# After all the cleaning and before the discrete variables analysis
print("\nAnalyzing all categorical variables and their relationship with price:")
print("----------------------------------------------------------------")

# List all categorical variables
categorical_vars = [
    'Construction_type', 'Balcony', 'Renovation', 'New_construction', 'Elevator',
    'Furniture', 'Gender', 'amenities', 'appliances', 'parking'
]

# Calculate comprehensive statistics for each categorical variable
print("\nDetailed analysis of each categorical variable's impact on price:")
print("------------------------------------------------------------")

for var in categorical_vars:
    # Calculate statistics for each category
    stats = df_cleaned.groupby(var).agg({
        'Price_USD': ['count', 'mean', 'median', 'std',
                     lambda x: len(x[x > x.quantile(0.75)]) / len(x) * 100]  # % of high-priced properties
    })['Price_USD']
    
    stats.columns = ['count', 'mean', 'median', 'std', '% high_price']
    
    # Sort by mean price
    stats = stats.sort_values('mean', ascending=False)
    
    print(f"\n{var} impact on price:")
    print("------------------------")
    print(stats)
    print("\nNumber of unique values:", len(stats))
    
print("\nNow you can analyze these statistics and select the top 3 variables that you think")
print("are most likely to contribute to high-priced properties based on:")
print("1. The mean and median prices")
print("2. The number of properties in each category (count)")
print("3. The standard deviation (std) showing price spread")
print("4. The percentage of high-priced properties in each category (% high_price)")

# =================================================================
# Correlation Analysis
# =================================================================
# Purpose: Analyze the correlation between numeric variables to
# identify potential relationships and impacts on the target variable (Price_USD).

# Continue with correlation analysis
print("\nAnalyzing correlation between rooms, price, and duration:")
print("---------------------------------------------------")

# Create duration_numeric for correlation analysis (1 for daily, 30 for monthly)
df_cleaned['Duration_numeric'] = df_cleaned['Duration'].map({'daily': 1, 'monthly': 30})

# Calculate correlations
correlation_vars = ['Number_of_rooms', 'Price_USD', 'Duration_numeric']
correlations = df_cleaned[correlation_vars].corr()
print("\nCorrelation Matrix:")
print(correlations)

# Calculate average price by number of rooms and duration
price_by_rooms = df_cleaned.groupby(['Number_of_rooms', 'Duration'])[['Price_USD']].agg(['mean', 'count'])
print("\nAverage price by number of rooms and duration:")
print(price_by_rooms)

# Print insights
print("\nCorrelation Analysis Insights:")
print("-----------------------------")
rooms_price_corr = correlations.loc['Number_of_rooms', 'Price_USD']
rooms_duration_corr = correlations.loc['Number_of_rooms', 'Duration_numeric']
price_duration_corr = correlations.loc['Price_USD', 'Duration_numeric']

print(f"1. Rooms vs Price correlation: {rooms_price_corr:.3f}")
print(f"2. Rooms vs Duration correlation: {rooms_duration_corr:.3f}")
print(f"3. Price vs Duration correlation: {price_duration_corr:.3f}")

# =================================================================
# Discrete Variables Analysis
# =================================================================
# Purpose: Analyze discrete variables and their relationship with
# the target variable (Price_USD). This helps in understanding
# how different discrete factors impact pricing.

# After all the cleaning and before saving the file, analyze discrete variables relationship with price
print("\nAnalyzing discrete variables relationship with price:")
print("------------------------------------------------")

# List of discrete variables to analyze
discrete_vars = ['Construction_type', 'Balcony', 'Renovation', 'New_construction', 'Elevator']

# Calculate mean price for each category in each variable
for var in discrete_vars:
    print(f"\n{var} analysis:")
    price_by_category = df_cleaned.groupby(var)['Price_USD'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    print(price_by_category)
    
# Analyze relationship between discrete variables and price
print("\nAnalyzing impact on high-priced properties:")
print("---------------------------------------")

for var in discrete_vars:
    # Calculate statistics for each category
    stats = df_cleaned.groupby(var).agg({
        'Price_USD': ['mean', 'median', 'count', 
                     lambda x: len(x[x > x.quantile(0.75)]) / len(x) * 100]  # % of high-priced properties
    })['Price_USD']
    stats.columns = ['mean', 'median', 'count', '% high_price']
    
    # Sort by mean price
    stats = stats.sort_values('mean', ascending=False)
    
    print(f"\n{var} impact on price:")
    print(stats)
    
# Based on the analysis above, print the top 3 impactful variables
print("\nTop 3 variables most likely to contribute to high-priced properties:")
print("---------------------------------------------------------------")
print("1. Construction_type - Stone and Cassette types have significantly higher mean prices")
print("2. Renovation - Major Renovation shows the highest average price")
print("3. Balcony - Closed Balcony properties command premium prices")

# =================================================================
# Final Data Cleaning Summary and Export
# =================================================================
# Purpose: Provide a summary of the cleaning process, final checks,
# and export the cleaned dataset to a new file for further analysis
# or modeling.

# Original statistics continue below...
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

print("\nAnalyzing Address Impact on Prices:")
print("--------------------------------")

# Group by address and calculate statistics
address_stats = df_cleaned.groupby('Address').agg({
    'Price_USD': ['count', 'mean', 'median', 'std']
}).round(2)

address_stats.columns = ['count', 'mean_price', 'median_price', 'price_std']
address_stats = address_stats.sort_values('mean_price', ascending=False)

# Calculate the overall mean price for comparison
overall_mean_price = df_cleaned['Price_USD'].mean()

# Add price premium/discount compared to overall mean
address_stats['price_premium_pct'] = ((address_stats['mean_price'] - overall_mean_price) / overall_mean_price * 100).round(2)

# Filter to locations with significant sample size (more than 5 listings)
significant_locations = address_stats[address_stats['count'] > 5]

print("\nTop 10 Most Expensive Areas:")
print(significant_locations.head(10))

print("\nPrice Variation by Location:")
print(f"Number of unique locations: {len(address_stats)}")
print(f"Locations with significant samples (>5 listings): {len(significant_locations)}")
print(f"\nPrice range across locations:")
print(f"Minimum average price: ${significant_locations['mean_price'].min():,.2f}")
print(f"Maximum average price: ${significant_locations['mean_price'].max():,.2f}")
print(f"Price variation (std): ${significant_locations['mean_price'].std():,.2f}")

# Calculate coefficient of variation to measure relative price dispersion
cv = significant_locations['mean_price'].std() / significant_locations['mean_price'].mean()
print(f"\nCoefficient of variation: {cv:.2%}")

# Analyze top price predictors
print("\nAnalyzing Top Price Predictors:")
print("-----------------------------")

predictors = ['Renovation', 'Construction_type', 'Number_of_rooms', 'Ceiling_height', 
              'Furniture', 'New_construction', 'Elevator']

# Calculate comprehensive statistics for each predictor
predictor_impact = {}

for pred in predictors:
    if pred in df_cleaned.columns:
        # Drop rows with NaN values for this analysis
        pred_data = df_cleaned[[pred, 'Price_USD']].dropna()
        
        if pred_data[pred].dtype in ['int64', 'float64']:
            # For numeric predictors, calculate correlation
            correlation = pred_data[pred].corr(pred_data['Price_USD'])
            predictor_impact[pred] = {
                'correlation': correlation,
                'type': 'numeric'
            }
        else:
            # For categorical predictors, calculate eta squared
            categories = pred_data[pred].unique()
            category_means = pred_data.groupby(pred)['Price_USD'].mean()
            overall_mean = pred_data['Price_USD'].mean()
            
            # Calculate eta squared (measure of effect size)
            ss_between = sum(len(pred_data[pred_data[pred] == cat]) * 
                           (category_means[cat] - overall_mean) ** 2 
                           for cat in categories)
            ss_total = sum((pred_data['Price_USD'] - overall_mean) ** 2)
            eta_squared = ss_between / ss_total if ss_total != 0 else 0
            
            predictor_impact[pred] = {
                'eta_squared': eta_squared,
                'type': 'categorical'
            }

print("\nPredictor Impact Analysis:")
print("-------------------------")
for pred, impact in predictor_impact.items():
    if impact['type'] == 'numeric':
        print(f"{pred}: Correlation = {impact['correlation']:.3f}")
    else:
        print(f"{pred}: Eta Squared = {impact['eta_squared']:.3f}")

print("\nSaving cleaned data for model building...")
df_cleaned.to_csv('apartment_for_rent_train_cleaned.csv', index=False)

# ===========================================================================
# Feature Standardization Functions
# ===========================================================================

def standardize_area_values(area_str):
    """
    Standardize area values to square meters.
    Handles various input formats like:
    - Raw numbers (assumes sq meters)
    - Values with 'sqft' or 'sq ft' (converts to sq meters)
    - Values with 'm²' or 'sqm'
    
    Args:
        area_str: String containing the area value
        
    Returns:
        float: Standardized area in square meters
    """
    # Remove spaces and convert to lowercase
    area_str = area_str.strip().lower()
    
    # Handle m² or sqm directly as square meters
    if 'm²' in area_str or 'sqm' in area_str:
        area_value = area_str.replace('m²', '').replace('sqm', '').strip()
        return float(area_value) if area_value.replace('.', '', 1).isdigit() else None
    
    # Handle sqft or sq ft - convert to square meters (1 sqft = 0.092903 m²)
    if 'sqft' in area_str or 'sq ft' in area_str:
        area_value = area_str.replace('sqft', '').replace('sq ft', '').strip()
        return float(area_value) * 0.092903 if area_value.replace('.', '', 1).isdigit() else None
    
    # For raw numbers, assume it's already in square meters
    return float(area_str) if area_str.replace('.', '', 1).isdigit() else None

def standardize_price(price_str):
    """
    Standardize price values to a common currency (USD).
    Handles:
    - Different currency symbols ($, €, £, etc.)
    - Different formats (1,000 vs 1000)
    - Monthly vs yearly prices
    - Conversion of major currencies to USD
    
    Args:
        price_str: String containing the price value
        
    Returns:
        float: Standardized price in USD
    """
    # Remove spaces and convert to lowercase
    price_str = price_str.strip().lower()
    
    # Handle USD directly
    if '$' in price_str or 'usd' in price_str:
        price_value = price_str.replace('$', '').replace('usd', '').replace(',', '').strip()
        return float(price_value) if price_value.replace('.', '', 1).isdigit() else None
    
    # Handle AMD - convert to USD (1 AMD = 0.0026 USD)
    if 'amd' in price_str:
        price_value = price_str.replace('amd', '').replace(',', '').strip()
        return float(price_value) * 0.0026 if price_value.replace('.', '', 1).isdigit() else None
    
    # Handle EUR - convert to USD (assuming 1 EUR = 1.1 USD for example)
    if '€' in price_str or 'eur' in price_str:
        price_value = price_str.replace('€', '').replace('eur', '').replace(',', '').strip()
        return float(price_value) * 1.1 if price_value.replace('.', '', 1).isdigit() else None
    
    # Add more currency conversions as needed...
    
    # If no currency symbol matched, return None
    return None

# ===========================================================================
# Categorical Variable Standardization
# ===========================================================================

def standardize_categorical_features(df):
    """
    Standardize categorical variables to ensure consistency:
    - Normalize string cases (lower/upper)
    - Group similar categories
    - Handle misspellings and variations
    - Create meaningful category groupings for rare values
    
    Args:
        df: DataFrame containing categorical columns
        
    Returns:
        DataFrame: DataFrame with standardized categorical variables
    """
    # Property type standardization
    property_type_mapping = {
        'apt': 'apartment',
        'apartment': 'apartment',
        'flat': 'apartment',
        'condo': 'condominium',
        # ...existing mappings...
    }
    
    # Standardize property types using the mapping
    df['property_type'] = df['property_type'].str.lower().map(property_type_mapping)
    
    # Standardize other categorical variables following similar patterns
    # ...existing code...

# Now you can call these standardization functions as needed on your DataFrame columns
# For example:
# df['Area'] = df['Area'].apply(standardize_area_values)
# df['Price'] = df['Price'].apply(standardize_price)
# df = standardize_categorical_features(df)

# ===========================================================================
# Data Validation and Quality Checks
# ===========================================================================

def validate_numeric_ranges(df):
    """
    Validate numeric values are within reasonable ranges.
    Identifies and handles outliers, invalid values, and data entry errors.
    
    Args:
        df: DataFrame containing numeric columns
        
    Returns:
        DataFrame: DataFrame with validated numeric values
        dict: Summary of validation results and adjustments made
    """
    validation_summary = {}
    
    # Price validation
    # Check for unreasonable prices (too low or too high)
    price_stats = {
        'mean': df['price_usd'].mean(),
        'std': df['price_usd'].std(),
        'min': df['price_usd'].min(),
        'max': df['price_usd'].max()
    }
    
    # Define reasonable price ranges (e.g., 3 standard deviations from mean)
    price_lower_bound = max(100, price_stats['mean'] - 3 * price_stats['std'])
    price_upper_bound = price_stats['mean'] + 3 * price_stats['std']
    
    # Flag and handle price outliers
    price_outliers = df[
        (df['price_usd'] < price_lower_bound) | 
        (df['price_usd'] > price_upper_bound)
    ]
    validation_summary['price_outliers'] = len(price_outliers)
    
    # Similar validation for area and other numeric features
    # ...existing code...
    
    return df, validation_summary

def validate_categorical_consistency(df):
    """
    Validate consistency of categorical variables:
    - Check for unexpected categories
    - Ensure all categories are mapped
    - Verify categorical relationships make sense
    
    Args:
        df: DataFrame containing categorical columns
        
    Returns:
        DataFrame: DataFrame with validated categorical values
        dict: Summary of validation results
    """
    # ...existing code...

# ===========================================================================
# Final Data Cleaning and Export
# ===========================================================================

def perform_final_cleaning(df):
    """
    Perform final cleaning steps and prepare data for export:
    1. Remove any remaining invalid records
    2. Handle missing values according to column-specific strategies
    3. Verify data quality metrics are met
    4. Generate final cleaning report
    
    Args:
        df: DataFrame to be finalized
        
    Returns:
        DataFrame: Final cleaned DataFrame
        dict: Cleaning summary statistics
    """
    cleaning_summary = {}
    
    # Handle any remaining missing values
    missing_stats = df.isnull().sum()
    cleaning_summary['missing_values'] = missing_stats.to_dict()
    
    # Apply final cleaning steps
    df_final = df.copy()
    
    # Document all changes made
    cleaning_summary['rows_initial'] = len(df)
    cleaning_summary['rows_final'] = len(df_final)
    cleaning_summary['columns_modified'] = list(df.columns)
    
    return df_final, cleaning_summary

# Execute the cleaning pipeline
if __name__ == "__main__":
    # Load and clean the data
    df_cleaned, validation_results = validate_numeric_ranges(df_cleaned)
    df_cleaned = standardize_categorical_features(df_cleaned)
    df_final, cleaning_summary = perform_final_cleaning(df_cleaned)
    
    # Print cleaning summary
    print("\nData Cleaning Summary:")
    print("--------------------")
    print(f"Initial rows: {cleaning_summary['rows_initial']}")
    print(f"Final rows: {cleaning_summary['rows_final']}")
    print(f"Columns modified: {len(cleaning_summary['columns_modified'])}")
    print(f"Price outliers identified: {validation_results['price_outliers']}")
    
    # Export cleaned dataset
    output_path = "apartment_for_rent_train_cleaned.csv"
    df_final.to_csv(output_path, index=False)
    print(f"\nCleaned data exported to: {output_path}")

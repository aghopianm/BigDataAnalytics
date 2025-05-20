import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import matplotlib.pyplot as plt
import seaborn as sns
from transliterate import translit
from transliterate.base import TranslitLanguagePack, registry

# Load the cleaned data
df = pd.read_csv('apartment_for_rent_train_cleaned.csv')

# Remove rows where Price_USD is NaN
df = df.dropna(subset=['Price_USD'])

# Function to transliterate Armenian and Russian to English
def convert_to_english(text):
    try:
        # Try Armenian first
        text_en = translit(text, 'hy', reversed=True)
        # Then try Russian for any remaining Cyrillic
        text_en = translit(text_en, 'ru', reversed=True)
        return text_en
    except:
        return text

# Convert addresses to English and standardize
df['Address_standardized'] = df['Address'].apply(convert_to_english)

# Get address counts and create categories
address_counts = df['Address_standardized'].value_counts()
address_terciles = np.percentile(address_counts, [33.33, 66.66])

def get_count_category(address):
    count = address_counts[address]
    if count <= address_terciles[0]:
        return 'low_listing_area'
    elif count <= address_terciles[1]:
        return 'medium_listing_area'
    else:
        return 'high_listing_area'

# Create categorical feature for address counts
df['Address_count_category'] = df['Address_standardized'].apply(get_count_category)

# Select features including new address categorization
features = ['Renovation', 'Construction_type', 'Number_of_rooms', 'Ceiling_height',
           'Furniture', 'New_construction', 'Elevator', 'Address_count_category']

# Create label encoders for categorical variables
label_encoders = {}
X = df[features].copy()
for column in X.select_dtypes(include=['object']):
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column].fillna('Unknown'))

# Fill numeric missing values with median
for column in X.select_dtypes(include=['float64', 'int64']):
    X[column] = X[column].fillna(X[column].median())

# Prepare target variable
y = df['Price_USD']

print(f"Final dataset size: {len(X)}")
print("\nFeature missing values:")
print(X.isnull().sum())
print("\nTarget variable (Price_USD) missing values:", y.isnull().sum())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    print(f"\n{name} Performance:")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    ev_score = explained_variance_score(y_test, y_pred)
    
    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    # Print metrics
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"R-squared Score: {r2:.3f}")
    print(f"Explained Variance Score: {ev_score:.3f}")
    print("\nCross-validation Scores (R²):")
    print(f"Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    print("Individual CV Scores:", cv_scores)

# Use Random Forest for feature importance and further analysis
rf_model = models['Random Forest']

print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"R-squared Score: {r2:.3f}")
print(f"Explained Variance Score: {ev_score:.3f}")
print("\nCross-validation Scores (R²):")
print(f"Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
print("Individual CV Scores:", cv_scores)

# Calculate feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance in Predicting Rental Prices')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Print detailed analysis of top predictors
print("\nDetailed Analysis of Top Predictors:")
top_features = feature_importance['feature'].head(3).tolist()

for feature in top_features:
    if feature in label_encoders:
        # For categorical features
        original_categories = label_encoders[feature].classes_
        encoded_values = label_encoders[feature].transform(original_categories)
        avg_prices = []
        
        for cat, enc_val in zip(original_categories, encoded_values):
            mask = X[feature] == enc_val
            avg_price = y[mask].mean()
            avg_prices.append((cat, avg_price))
        
        print(f"\n{feature} Analysis:")
        for cat, avg_price in sorted(avg_prices, key=lambda x: x[1], reverse=True):
            print(f"{cat}: ${avg_price:,.2f}")
    else:
        # For numerical features
        correlation = df[feature].corr(df['Price_USD'])
        print(f"\n{feature} Analysis:")
        print(f"Correlation with price: {correlation:.3f}")
        print(f"Mean value: {df[feature].mean():.2f}")
        print(f"Median value: {df[feature].median():.2f}")

# Analyze price by address
address_price_analysis = df.groupby('Address')['Price_USD'].agg(['mean', 'count', 'std']).reset_index()
address_price_analysis = address_price_analysis.sort_values('mean', ascending=False)
print("\nTop 10 Most Expensive Areas:")
print(address_price_analysis.head(10))

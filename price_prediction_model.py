import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import matplotlib.pyplot as plt
import seaborn as sns
from transliterate import translit
from transliterate.base import TranslitLanguagePack, registry

# Set style for all plots
sns.set_style("darkgrid")  # Set seaborn style directly
plt.style.use('seaborn-v0_8-darkgrid')  # Use specific matplotlib style compatible with seaborn

def plot_feature_importance(model, feature_names, output_path='visualizations/feature_importance.png'):
    """Plot feature importance from the model"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        
        # Sort feature importances in descending order
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.title('Feature Importances')
        plt.bar(range(len(indices)), importance[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def plot_prediction_scatter(y_true, y_pred, output_path='visualizations/prediction_scatter.png'):
    """Plot scatter plot of predicted vs actual values"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Predicted vs Actual Prices')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_residuals(y_true, y_pred, output_path='visualizations/residuals.png'):
    """Plot residuals analysis"""
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Residuals distribution
    sns.histplot(residuals, kde=True, ax=ax1)
    ax1.set_title('Residuals Distribution')
    ax1.set_xlabel('Residual Value')
    
    # Residuals vs Predicted
    ax2.scatter(y_pred, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_title('Residuals vs Predicted Values')
    ax2.set_xlabel('Predicted Price')
    ax2.set_ylabel('Residual')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_model_comparison(model_scores, output_path='visualizations/model_comparison.png'):
    """Plot model comparison"""
    plt.figure(figsize=(12, 6))
    
    # Sort models by R² score
    model_scores = {k: v for k, v in sorted(model_scores.items(), key=lambda item: item[1]['R2'], reverse=True)}
    
    models = list(model_scores.keys())
    r2_scores = [scores['R2'] for scores in model_scores.values()]
    mse_scores = [scores['MSE'] for scores in model_scores.values()]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax2 = ax1.twinx()
    
    # Plot R² scores
    rects1 = ax1.bar(x - width/2, r2_scores, width, label='R² Score', color='skyblue')
    ax1.set_ylabel('R² Score')
    
    # Plot MSE scores
    rects2 = ax2.bar(x + width/2, mse_scores, width, label='MSE', color='lightcoral')
    ax2.set_ylabel('Mean Squared Error')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.title('Model Comparison')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# =====================================
# Data Loading and Initial Cleaning
# =====================================
# Load the previously cleaned training dataset
df = pd.read_csv('apartment_for_rent_train_cleaned.csv')

# Remove any rows where the target variable (Price_USD) is missing
# This is important as we can't train on rows without prices
df = df.dropna(subset=['Price_USD'])

# =====================================
# Address Standardization
# =====================================
# Function to convert Armenian and Russian text to English characters
# This helps standardize addresses and make them more processable
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

# =====================================
# Feature Engineering: Address Categories
# =====================================
# Count how many times each address appears in the dataset
address_counts = df['Address_standardized'].value_counts()

# Create terciles (33.33% and 66.66% points) to split addresses into three categories
# This helps identify high, medium, and low-density listing areas
address_terciles = np.percentile(address_counts, [33.33, 66.66])

# Function to categorize addresses based on their frequency
# This creates a new feature that captures the popularity/density of listings in each area
def get_count_category(address):
    count = address_counts[address]
    if count <= address_terciles[0]:
        return 'low_listing_area'      # Bottom third: areas with few listings
    elif count <= address_terciles[1]:
        return 'medium_listing_area'    # Middle third: areas with moderate listings
    else:
        return 'high_listing_area'      # Top third: areas with many listings

# Create categorical feature for address counts
df['Address_count_category'] = df['Address_standardized'].apply(get_count_category)

# Select features including new address categorization
features = ['Renovation', 'Construction_type', 'Number_of_rooms', 'Ceiling_height',
           'Furniture', 'New_construction', 'Elevator', 'Address_count_category']

# =====================================
# Feature Selection and Missing Value Treatment
# =====================================
# Create feature matrix X by selecting only the columns we want to use for prediction
X = df[features].copy()

# Automatically identify numeric and categorical columns based on their data types
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Handle missing values in numeric features
# We use median instead of mean because it's more robust to outliers
for column in numeric_features:
    X[column] = X[column].fillna(X[column].median())

# Handle missing values in categorical features
# We use 'Unknown' as it's interpretable and doesn't assume any specific category
for column in categorical_features:
    X[column] = X[column].fillna('Unknown')

# =====================================
# Feature Preprocessing Pipeline Setup
# =====================================
# Pipeline for numeric features:
# StandardScaler: Standardizes features by removing the mean and scaling to unit variance
# This is important because features are on different scales (e.g., rooms vs height)
numeric_transformer = Pipeline([
    ('scaler', StandardScaler())
])

# Pipeline for categorical features:
# OneHotEncoder: Converts categorical variables into binary (0/1) features
# drop='first': Drops first category to avoid multicollinearity
# sparse_output=False: Returns dense array instead of sparse matrix
# handle_unknown='ignore': Handles any new categories in test data by ignoring them
categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

# Combine both transformers into a single preprocessing pipeline
# This ensures that the correct transformation is applied to each feature type
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# =====================================
# Target Variable Preparation
# =====================================
# Extract the target variable (price)
y = df['Price_USD']

# Apply log transformation to the target variable because:
# 1. Real estate prices typically follow a log-normal distribution
# 2. Log transformation helps stabilize variance
# 3. Makes the target variable more normally distributed
# Using log1p instead of log because:
# - log1p(x) = log(1 + x)
# - Handles zero values gracefully
# - Makes it easier to interpret results (log1p(0) = 0)
y = np.log1p(y)

print(f"Final dataset size: {len(X)}")
print("\nFeature missing values:")
print(X.isnull().sum())
print("\nTarget variable (log-transformed Price_USD) missing values:", y.isnull().sum())
print("\nNumeric features:", list(numeric_features))
print("Categorical features:", list(categorical_features))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the preprocessing pipeline
preprocessor.fit(X_train)

# Transform the training and test data
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# =====================================
# Model Initialization with Optimized Parameters
# =====================================
models = {
    # Simple baseline model with no hyperparameters
    'Linear Regression': LinearRegression(),
    
    # Decision Tree with controlled depth to prevent overfitting
    # max_depth=10: Limits tree depth to prevent overfitting
    # min_samples_leaf=5: Ensures each leaf has at least 5 samples
    'Decision Tree': DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=42),
    
    # Random Forest: Ensemble of trees with controlled complexity
    # n_estimators=100: Uses 100 trees for robust predictions
    # max_depth=15: Deeper than single tree but still controlled
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=5, random_state=42),
    
    # Gradient Boosting: Advanced ensemble method that builds trees sequentially
    # learning_rate=0.1: Controls how much each tree contributes to the final prediction
    # max_depth=5: Shallow trees to prevent overfitting in boosting
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
    
    # AdaBoost: Adaptive Boosting, focuses on hard-to-predict cases
    # Fewer estimators than other ensembles as it can overfit easily
    'AdaBoost': AdaBoostRegressor(n_estimators=50, learning_rate=0.1, random_state=42),
    
    # Elastic Net: Linear regression with both L1 and L2 regularization
    # alpha=1.0: Controls overall regularization strength
    # l1_ratio=0.5: Equal mix of L1 (Lasso) and L2 (Ridge) regularization
    'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
    
    # K-Nearest Neighbors: Non-parametric method based on closest training examples
    # weights='distance': Closer neighbors have more influence
    'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5, weights='distance'),
    
    # Extra Trees: Similar to Random Forest but with random splits
    # Generally more random than Random Forest, which can reduce variance
    'Extra Trees': ExtraTreesRegressor(n_estimators=100, max_depth=15, min_samples_leaf=5, random_state=42),
    
    # Bayesian Ridge: Probabilistic linear regression
    # Automatically determines regularization parameters
    'Bayesian Ridge': BayesianRidge(),
    
    # Support Vector Regression (SVR) is commented out for the following reasons:
    # 1. Computational Complexity: O(n²) to O(n³), making it impractical for our dataset size of 32,563 samples
    # 2. Memory Usage: Requires storing the kernel matrix of size n×n, which would need ~8.5GB RAM
    # 3. Training Time: Would take several hours to train on this dataset size
    'Support Vector Regressor': SVR(kernel='rbf', C=1.0, epsilon=0.1),
    
    # Gaussian Naive Bayes is commented out for the following reasons:
    # 1. Not Suitable for Regression: GNB is a classification algorithm, not designed for continuous target variables
    # 2. Independence Assumption: Assumes features are independent, which is not true for real estate data
    # 3. Distribution Assumption: Assumes Gaussian distribution of features within each class, not applicable for regression
    # 'Gaussian Naive Bayes': GaussianNB(),
    
    # Kernel Ridge Regression is commented out for the following reasons:
    # 1. Memory Issues: Like SVR, requires storing a kernel matrix of size n×n (~8.5GB for our dataset)
    # 2. Computational Complexity: O(n³) complexity makes it impractical for large datasets
    # 3. Better Alternatives: Random Forest and Gradient Boosting can capture non-linear relationships more efficiently
    # 'Kernel Ridge': KernelRidge(kernel='rbf', alpha=1.0)
}

# Dictionary to store model performances
model_performances = {}

# =====================================
# Model Training and Evaluation Loop
# =====================================
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    # Train the model on the transformed training data
    # X_train_transformed: Feature matrix after preprocessing
    # y_train: Log-transformed target variable
    model.fit(X_train_transformed, y_train)
    
    # Generate predictions on the test set
    # These predictions will be in log-scale since we trained on log-transformed target
    y_pred = model.predict(X_test_transformed)
    
    # =====================================
    # Calculate Performance Metrics
    # =====================================
    # Mean Squared Error: Average squared difference between predictions and actual values
    # Lower is better, penalizes large errors more than small ones
    mse = mean_squared_error(y_test, y_pred)
    
    # Root Mean Squared Error: Square root of MSE, in same units as target variable
    # More interpretable than MSE, still in log-scale due to our transformation
    rmse = np.sqrt(mse)
    
    # R-squared: Proportion of variance in target that's predictable from features
    # Range: 0 to 1, higher is better, negative means worse than horizontal line
    r2 = r2_score(y_test, y_pred)
    
    # Mean Absolute Error: Average absolute difference between predictions and actual values
    # More robust to outliers than MSE/RMSE
    mae = mean_absolute_error(y_test, y_pred)
    
    # Explained Variance Score: Similar to R-squared but focuses on variance explanation
    # Range: 0 to 1, higher is better
    ev_score = explained_variance_score(y_test, y_pred)
    
    # Mean Absolute Percentage Error: Average percentage difference from actual values
    # More interpretable as it's scale-independent and shown as percentage
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train_transformed, y_train, cv=5, scoring='r2')
    
    # Store performance metrics
    model_performances[model_name] = {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAE': mae,
        'MAPE': mape,
        'Explained_Variance': ev_score,
        'CV_Mean': cv_scores.mean(),
        'CV_Std': cv_scores.std()
    }
    
    print(f"{model_name} Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Explained Variance Score: {ev_score:.3f}")
    print(f"Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    print("Individual CV Scores:", cv_scores)

# Save model comparison results to CSV
results_df = pd.DataFrame.from_dict(model_performances, orient='index')
results_df.to_csv('model_comparison_results.csv')
print("\nModel comparison results saved to 'model_comparison_results.csv'")

# Create visualizations directory if it doesn't exist
import os
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

print("\nGenerating model performance visualizations...")

# Plot model comparison
plot_model_comparison(model_performances)

# Get best model
best_model_name = results_df.sort_values('R2', ascending=False).index[0]
best_model = models[best_model_name]

# Get predictions from best model for visualization
y_pred_best = best_model.predict(X_test_transformed)

# Plot predictions vs actual values
plot_prediction_scatter(y_test, y_pred_best)

# Plot residuals analysis
plot_residuals(y_test, y_pred_best)

# Plot feature importance if available
if hasattr(best_model, 'feature_importances_'):
    plot_feature_importance(best_model, preprocessor.get_feature_names_out())
elif 'Random Forest' in models:
    # Fallback to Random Forest for feature importance
    rf_model = models['Random Forest']
    rf_model.fit(X_train_transformed, y_train)
    plot_feature_importance(rf_model, preprocessor.get_feature_names_out())

# Plot feature importance for the best model
if hasattr(best_model, 'feature_importances_'):
    plot_feature_importance(model_for_importance, preprocessor.get_feature_names_out())

# Plot predictions vs actual values
y_pred_best = best_model.predict(X_test_transformed)
plot_prediction_scatter(y_test, y_pred_best)

# Plot residuals analysis
plot_residuals(y_test, y_pred_best)

print("Visualizations saved in 'visualizations' directory")

# Save preprocessed data to CSV
preprocessed_train = pd.DataFrame(
    X_train_transformed,
    columns=preprocessor.get_feature_names_out()
)
preprocessed_train.to_csv('preprocessed_data.csv', index=False)
print("Preprocessed training data saved to 'preprocessed_data.csv'")

# =====================================
# Best Model Selection and Analysis
# =====================================
# Fix: Get the best model name from the index, not from a 'Model' column
best_model_name = results_df.sort_values('R2', ascending=False).index[0]
best_model = models[best_model_name]

# Use the best model for feature importance and further analysis
# Check if the best model has feature importance capability
if hasattr(best_model, 'feature_importances_'):
    model_for_importance = best_model
    model_name_for_importance = best_model_name
else:
    # Fallback to Random Forest if best model doesn't have feature_importances_
    model_for_importance = models['Random Forest']
    model_name_for_importance = 'Random Forest'

print(f"\nBest performing model: {best_model_name}")
print(f"Using {model_name_for_importance} for feature importance analysis")

# Get the performance metrics for the best model
best_performance = model_performances[best_model_name]

print(f"\n{best_model_name} Performance:")
print(f"Mean Squared error (MSE): ${best_performance['MSE']:.2f}")
print(f"Root Mean Squared Error (RMSE): ${best_performance['RMSE']:.2f}")
print(f"Mean Absolute Error (MAE): ${best_performance['MAE']:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {best_performance['MAPE']:.2f}%")
print(f"R-squared Score: {best_performance['R2']:.3f}")
print(f"Explained Variance Score: {best_performance['Explained_Variance']:.3f}")

print(f"\nCross-validation Scores (R²):")
print(f"Mean CV Score: {best_performance['CV_Mean']:.3f} (+/- {best_performance['CV_Std'] * 2:.3f})")

# =====================================
# Feature Importance Analysis
# =====================================
# Get feature names after preprocessing (includes one-hot encoded features)
feature_names = preprocessor.get_feature_names_out()

# Extract feature importance scores from the selected model
# Random Forest is used because:
# 1. It provides reliable importance scores
# 2. Can handle both numeric and categorical features
# 3. Accounts for both linear and non-linear relationships
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model_for_importance.feature_importances_
})

# Sort features by importance score in descending order
feature_importance = feature_importance.sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features (from {model_name_for_importance}):")
print(feature_importance.head(10))

# Plot feature importance (top 10 features)
plt.figure(figsize=(12, 6))
top_10_features = feature_importance.head(10)
sns.barplot(x='importance', y='feature', data=top_10_features)
plt.title(f'Top 10 Feature Importance in {model_name_for_importance} Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nFeature importance plot saved as 'feature_importance.png'")

# Print detailed analysis of original features
print("\nDetailed Analysis of Features:")

# Analyze numeric features
for feature in numeric_features:
    correlation = df[feature].corr(df['Price_USD'])
    print(f"\n{feature} Analysis:")
    print(f"Correlation with price: {correlation:.3f}")
    print(f"Mean value: {df[feature].mean():.2f}")
    print(f"Median value: {df[feature].median():.2f}")

# Analyze categorical features
for feature in categorical_features:
    print(f"\n{feature} Analysis:")
    value_counts = df[feature].value_counts()
    avg_prices = df.groupby(feature)['Price_USD'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    print("Category Distribution:")
    for category, count in value_counts.items():
        mean_price = avg_prices.loc[category, 'mean']
        print(f"{category}: {count} samples, Mean Price: ${mean_price:,.2f}")

# Analyze price by address category
address_price_analysis = df.groupby('Address_count_category')['Price_USD'].agg(['mean', 'count', 'std']).reset_index()
address_price_analysis = address_price_analysis.sort_values('mean', ascending=False)
print("\nPrice Analysis by Area Category:")
print(address_price_analysis)

# Final message
print(f"\nAnalysis complete! Best performing model: {best_model_name}")
print("Results have been saved to:")
print("- model_comparison_results.csv")
print("- preprocessed_data.csv")
print("- feature_importance.png")
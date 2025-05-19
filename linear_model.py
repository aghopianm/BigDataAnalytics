import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load and clean data
possum_data = pd.read_csv('possum.csv', delimiter=',', encoding='utf-8')
possum_data.rename(columns=lambda x: x.strip(), inplace=True)

# Define variables
independant_variable = possum_data[['skullW']]
dependant_variable = possum_data['headL']

# Create and fit the model
model = LinearRegression()
model.fit(independant_variable, dependant_variable)

# Predict on training data
train_predictions = model.predict(independant_variable)

# Compute metrics
mse = mean_squared_error(dependant_variable, train_predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(dependant_variable, train_predictions)
r2 = r2_score(dependant_variable, train_predictions)

# Print model parameters and evaluation metrics
print(f'Intercept: {round(model.intercept_, 2)}')
print(f'Coefficient: {round(model.coef_[0], 2)}')
print(f'Mean Squared Error (MSE): {round(mse, 2)}')
print(f'Root Mean Squared Error (RMSE): {round(rmse, 2)}')
print(f'Mean Absolute Error (MAE): {round(mae, 2)}')
print(f'R² Score: {round(r2, 2)}')

# Make new predictions
test = pd.DataFrame({'skullW': [70, 80, 85]})
prediction = model.predict(test)
print('Predictions:', prediction)

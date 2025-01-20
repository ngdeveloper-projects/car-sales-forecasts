import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Step 1: Generate Sample Data
models = [
    "Tata Nexon", "Tata Harrier", "Tata Tiago", "Tata Safari", 
    "Tata Altroz", "Tata Tigor", "Tata Punch", "Tata EV Max", "Tata Nexon EV"
]

data = []
for model in models:
    for month in range(1, 13):  # Monthly data for one year
        sales = random.randint(500, 5000)  # Random sales numbers
        inquiries = random.randint(600, 6000)  # Random inquiries
        advertising_spend = random.randint(100000, 1000000)  # Advertising spend in INR
        price = random.randint(600000, 2000000)  # Price range in INR
        data.append({
            "Month": month,
            "Car Model": model,
            "Sales": sales,
            "Inquiries": inquiries,
            "Advertising Spend (INR)": advertising_spend,
            "Price (INR)": price
        })

# Create a DataFrame
tata_car_data = pd.DataFrame(data)

# Save sample data to a CSV file for quick reference
sample_data_path = "tata_car_sales_sample_data.csv"
tata_car_data.to_csv(sample_data_path, index=False)
print(f"Sample data saved at: {os.path.abspath(sample_data_path)}")

# Step 2: Prepare Data for Modeling
le = LabelEncoder()
tata_car_data["Car Model Encoded"] = le.fit_transform(tata_car_data["Car Model"])

# Define features and target variable
features = tata_car_data[["Month", "Car Model Encoded", "Inquiries", "Advertising Spend (INR)", "Price (INR)"]]
target = tata_car_data["Sales"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Train a Random Forest Regressor
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train_scaled, y_train)

# Step 4: Generate Future Predictions
future_months = list(range(13, 16))  # Predicting for the next 3 months
future_predictions = []

for model_encoded in tata_car_data["Car Model Encoded"].unique():
    for month in future_months:
        sample_data = pd.DataFrame([{
            "Month": month,
            "Car Model Encoded": model_encoded,
            "Inquiries": random.randint(600, 6000),
            "Advertising Spend (INR)": random.randint(100000, 1000000),
            "Price (INR)": random.randint(600000, 2000000),
        }])
        sample_data_scaled = scaler.transform(sample_data)
        predicted_sales = model.predict(sample_data_scaled)
        future_predictions.append({
            "Month": month,
            "Car Model": le.inverse_transform([model_encoded])[0],
            "Predicted Sales": int(predicted_sales[0])
        })

# Create a DataFrame for the future predictions
future_forecast = pd.DataFrame(future_predictions)

# Filter for the next 3 months
next_three_months = future_forecast[future_forecast["Month"].isin([13, 14, 15])]

# Save forecasted sales data to a CSV file
forecast_output_path = "tata_car_sales_forecast_next_3_months.csv"
next_three_months.to_csv(forecast_output_path, index=False)
print(f"Forecast data saved at: {os.path.abspath(forecast_output_path)}")

# Step 5: Visualize the Predictions
plt.figure(figsize=(12, 8))
sns.barplot(data=next_three_months, x="Car Model", y="Predicted Sales", hue="Month")
plt.title("Forecasted Sales for Tata Car Models (Next 3 Months)")
plt.xlabel("Car Model")
plt.ylabel("Predicted Sales")
plt.legend(title="Month")
plt.xticks(rotation=45)
plt.show()

# Display the forecasted data
print(next_three_months)
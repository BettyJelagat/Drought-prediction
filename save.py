import pandas as pd
import numpy as np


import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


climatedata = pd.read_csv("1981-2024Daily summaries.csv", parse_dates=["DATE"],index_col="DATE")

Core_weather = climatedata[["T2M_MAX","T2M_MIN","PRECTOTCORR","QV2M","WS2M","GWETTOP"]].copy()
Core_weather.columns = ["temp_max","temp_min","precip","Humidity","w_speed","s_wetness"]

#convert date to pandas date time format
Core_weather.index = pd.to_datetime(Core_weather.index)

#Data cleaning
Core_weather.replace(-999, np.nan, inplace= True)
Core_weather.ffill(inplace=True)

#Compute the long-term average for each variable by day of the year
Core_weather['day_of_year'] = Core_weather.index.dayofyear

# Group by day of year to calculate long-term averages (e.g., average precipitation per day of year across all years)
long_term_avg = Core_weather.groupby('day_of_year')[['s_wetness', 'w_speed', 'precip', 'Humidity', 'temp_max', 'temp_min']].mean()

#Merge the long-term averages back to the original data
Core_weather = pd.merge(Core_weather, long_term_avg, on='day_of_year', suffixes=('', '_long_term'))

# Define thresholds for drought
precipitation_threshold = 0.75  # 75% of the long-term average
soil_moisture_threshold = 0.75  # 75% of the long-term average
temp_max_threshold = 2  # 2°C above the long-term average
def determine_drought(row):
    # Check drought conditions element-wise
    precip_drought = row['precip'] < (precipitation_threshold * row['precip_long_term'])
    temp_max_drought = row['temp_max'] > (row['temp_max_long_term'] + temp_max_threshold)
    soil_moisture_drought = row['s_wetness'] < (soil_moisture_threshold * row['s_wetness_long_term'])


    # Use element-wise logical AND (&) and check if all conditions are True
    if (precip_drought  &temp_max_drought & soil_moisture_drought).all():
        return 1  # Drought occurred
    else:
        return 0  # No drought

# Apply the function to each row
Core_weather['drought_occurred'] = Core_weather.apply(determine_drought, axis=1)


# Now you can separate features (X) and target (y)
X = Core_weather[["temp_max","temp_min","precip","Humidity","w_speed","s_wetness"]]  # Features
y = Core_weather['drought_occurred']  # Target: 0 = No Drought, 1 = Drought


#Standardize the data
scaler = StandardScaler()
X_transformed = scaler.fit_transform(Core_weather[["temp_max","temp_min","precip","Humidity","w_speed","s_wetness"]])

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)

## Initialize RandomForestClassifier with 100 estimators (trees)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the Random Forest model
rf_model.fit(X_train, y_train)

#save the model
with open('model/weather_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)

#save scaler
with open('model/scaler_model.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print('model and scaler saved successfully')
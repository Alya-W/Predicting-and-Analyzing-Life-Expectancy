import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import streamlit as st

file_path = 'life_exp.csv'
merged_data = pd.read_csv(file_path)

# Prepare the data
X = merged_data[[
'Region', 'Infant_deaths', 'Under_five_deaths', 'Adult_mortality', 
'BMI', 'Polio', 'Diphtheria', 'Schooling', 'Incidents_HIV', 
'GDP_per_capita', 'Economy_status_Developed', 'Economy_status_Developing'
]]

y = merged_data['Life_expectancy']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
regressor = RandomForestRegressor(
                              n_estimators=300,
                              random_state=42)
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

# Compute mean squared error (applicable for regression)
mse = mean_squared_error(y_test, y_pred)

# Compute R-squared score
r2 = r2_score(y_test, y_pred)

# streamlit start
st.title('Life Expectancy Prediction')
#st.dataframe(life_exp_data)
st.write('Using Random Forest Regressor')
st.write(f'Accuracy: {r2 * 100:.2f}%')


# User input
region_mapping = {'Africa': 0, 'European Union': 3, 'Asia': 1, 'Central America and Caribbean': 2,
                  'Rest of Europe': 7, 'Middle East': 4, 'South America': 8, 'Oceania': 6, 'North America': 5}

Region = st.selectbox('Region:', ('Africa', 'European Union', 'Asia', 'Central America and Caribbean',
                                  'Rest of Europe', 'Middle East', 'South America', 'Oceania', 'North America'))
Encoded_region = region_mapping[Region]

Infant_deaths = st.number_input('Infant Deaths:')
Under_five_deaths = st.number_input('Under Five Deaths:')
Adult_mortality = st.number_input('Adult Mortality:')
BMI = st.number_input('BMI:')
Polio = st.number_input('Polio:')
Diphtheria = st.number_input('Diphtheria:')
Schooling = st.number_input('Schooling:')
Incidents_HIV = st.number_input('HIV/AIDS Incidents:')
GDP_per_capita = st.number_input('GDP per Capita:')
Economy_Status = st.selectbox('Economy Status:', ('Developed', 'Developing'))

Economy_status_Developed = 1 if Economy_Status == 'Developed' else 0
Economy_status_Developing = 1 if Economy_Status == 'Developing' else 0


if st.button('Predict'):
  user_input = [[Encoded_region,
                 Infant_deaths,
                 Under_five_deaths,
                 Adult_mortality,
                 BMI,
                 Polio,
                 Diphtheria,
                 Schooling,
                 Incidents_HIV,
                 GDP_per_capita,
                 Economy_status_Developed,
                 Economy_status_Developing
                   ]]

  prediction = regressor.predict(user_input)
  st.write(f'The predicted life expectancy is: {round(prediction[0],2)}')
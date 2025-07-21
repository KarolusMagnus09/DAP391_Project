Tutorial: Running the Weather Prediction Project
This tutorial guides you through setting up and running a machine learning project for weather data processing and rainfall prediction, followed by uploading it to GitHub. The project includes data preprocessing, exploratory data analysis (EDA), and model training.

--Prerequisites
Python 3.8+: Ensure Python is installed
Dependencies: Install required Python libraries.
Data Files: Ensure the following files are in the vietnam-weather-data directory:
weather.csv: Raw weather data.
Final_Province_Coordinates.csv: Province coordinates.
Jupyter Notebook: For running the EDA notebook.
---Step 1: downloaded requirement.txt, you can copy this command"pip install -r requirements.txt"
---Step 2: Run the Data Preprocessing
Execute Transform_Data.py: This script loads raw data, performs feature engineering (e.g., adding lag features, seasonal indicators, wind components), handles outliers, and saves the processed data.
python Transform_Data.py
Input: vietnam-weather-data/weather.csv, vietnam-weather-data/Final_Province_Coordinates.csv
Output: vietnam-weather-data/added_feature_weather.csv (after renaming as per Step 1.3)
Notes:
Ensure utils.py is in the same directory, as it provides functions like convert_region, split_date, Clipping, and boxplots.
The script generates boxplots to visualize data before and after outlier clipping.
Check console output for missing data and duplicates (should be none).
---Step 3: Perform Exploratory Data Analysis:
Open Data_Understanding_Finding_Feature.ipynb and run all cells.
Dependencies: Requires utils.py for ols_metrics, anova_metrics, boxplots, and Clipping.
Input: vietnam-weather-data/added_feature_weather.csv
Output: Visualizations (e.g., bar plot of feature importance) and statistical analysis results.
---Step 4: Train Machine Learning Models
Run Model_Training.py: This script trains and evaluates Random Forest, KNN, and XGBoost models using hyperparameter tuning.
**python Model_Training.py**
Dependencies: Requires pandas, numpy, scikit-learn, xgboost, and time.
Input: final_weather_processed.csv
Output: Console output with model performance metrics (RMSE, R², training time, inference time) and best hyperparameters.
Notes:

Ensure final_weather_processed.csv exists (from Step 2.2).
The script splits data into training, validation, and test sets, applies log-transformation to the target (rain), and uses RandomizedSearchCV for hyperparameter tuning.
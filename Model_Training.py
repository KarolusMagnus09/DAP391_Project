model_df = pd.read_csv('final_weather_processed.csv')
features = [
    'humidi', 'cloud', 'pressure', 'cloud_humid',
    'is_rainy_season', 'is_dry_season',
    'max_temp', 'min_temp','range_temp',
    'rain_1d_ago', 'rain_2d_ago','rain_trend_3d','rain_intensity',         
    'wind_x', 'wind_y', 'Longitude', 'Latitude'                       
]
X =model_df[features]
y =model_df['rain']

# --- Chia dữ liệu ---
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# --- Log-transform target ---
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

# --- Hàm đánh giá mô hình ---
def evaluate_model(name, model, X_train, y_train_log, X_val, y_val):
    start_train = time.time()
    model.fit(X_train, y_train_log)
    train_time = time.time() - start_train

    start_infer = time.time()
    y_pred_log = model.predict(X_val)
    infer_time = (time.time() - start_infer) / len(X_val)

    y_pred = np.expm1(y_pred_log)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    return {
        'Model': name,
        'RMSE': round(rmse, 3),
        'R² Score (%)': round(r2 * 100, 2),
        'Train Time (s)': round(train_time, 2),
        'Inference Time (ms/sample)': round(infer_time * 1000, 4)
    }


#  1. Random Forest

param_dist_rf = {
    'n_estimators': [200, 300, 400, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
search_rf = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_dist_rf,
    n_iter=20, cv=3, verbose=1, random_state=42,
    scoring='neg_root_mean_squared_error', n_jobs=-1
)
search_rf.fit(X_train, y_train_log)
best_rf = search_rf.best_estimator_


#  2. KNN

param_dist_knn = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # 1: Manhattan, 2: Euclidean
}
search_knn = RandomizedSearchCV(
    KNeighborsRegressor(),
    param_distributions=param_dist_knn,
    n_iter=10, cv=3, verbose=1, random_state=42,
    scoring='neg_root_mean_squared_error', n_jobs=-1
)
search_knn.fit(X_train, y_train_log)
best_knn = search_knn.best_estimator_


#  3. XGBoost

param_dist_xgb = {
    'n_estimators': [200, 300, 400, 500],
    'max_depth': [3, 5, 10, 20],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}
search_xgb = RandomizedSearchCV(
    XGBRegressor(random_state=42, tree_method='hist', n_jobs=-1),
    param_distributions=param_dist_xgb,
    n_iter=30, cv=3, verbose=1, random_state=42,
    scoring='neg_root_mean_squared_error', n_jobs=-1
)
search_xgb.fit(X_train, y_train_log)
best_xgb = search_xgb.best_estimator_


#  Đánh giá tất cả mô hình

results = []
results.append(evaluate_model("Random Forest", best_rf, X_train, y_train_log, X_val, y_val))
results.append(evaluate_model("KNN", best_knn, X_train, y_train_log, X_val, y_val))
results.append(evaluate_model("XGBoost", best_xgb, X_train, y_train_log, X_val, y_val))

# --- In kết quả ---
df_results = pd.DataFrame(results)
print(df_results.sort_values(by='RMSE'))

# --- In tham số tốt nhất nếu cần ---
print("\nBest Params:")
print("Random Forest:", search_rf.best_params_)
print("KNN:", search_knn.best_params_)
print("XGBoost:", search_xgb.best_params_)
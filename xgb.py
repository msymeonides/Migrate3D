import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
# from multiprocessing import Pool
from itertools import product
import traceback


def signed_log_transformation(x):
    return np.sign(x) * np.log(np.abs(x) + 1)


def group_similar_features(df_features, correlation_threshold=0.9):
    """
    Groups features based on correlation threshold and returns a list of feature groups.
    """
    corr_matrix = df_features.corr().abs() # Calculate absolute correlation matrix
    feature_groups = []
    used_features = set()

    for feature in corr_matrix.columns:
        if feature not in used_features:
            # Find features that are highly correlated with the current feature
            correlated_features = set(corr_matrix.index[corr_matrix[feature] > correlation_threshold])
            feature_groups.append(correlated_features)
            used_features.update(correlated_features)
    return feature_groups


def aggregate_correlated_features(df, feature_groups):
    """
    Aggregates correlated features (mean) for each group of highly correlated features
    """
    aggregated_data = []
    feature_mapping = {}

    for i, group in enumerate(feature_groups):
        group_name = f"Group_{i + 1}"  # Create a name for each feature group
        group_features_label = ", ".join(group)  # Concatenate original names for labels
        aggregated_data.append(df[list(group)].mean(axis=1))  # Aggregate the group (mean)
        feature_mapping[group_features_label] = list(group)  # Store the original feature names for the group

    # Create aggregated dataframe with group names as column headers
    aggregated_df = pd.DataFrame(aggregated_data).T
    aggregated_df.columns = [", ".join(group) for group in feature_groups]
    #aggregated_df.columns = [f"Group_{i + 1}" for i in range(len(feature_groups))]
    print("Feature Mapping:", feature_mapping)

    return aggregated_df, feature_mapping
    #print("Feature Mapping:", feature_mapping)
    #aggregated_df = pd.DataFrame(aggregated_data).T
    #return aggregated_df, feature_mapping


def preprocess_features(df, parameters):
    """
    Preprocesses the feature data (log transformation and correlation grouping)
"""
    category_col = parameters['category_col']
    object_id_col = parameters['object_id_col_name']
    df_features = df.drop(['Category', 'Object ID'], axis=1)
    y = df['Category']

    # Apply signed log transformation and drop any rows with NaN values
    log_data = signed_log_transformation(df_features)
    log_data = log_data.dropna()

    # Ensure that we drop the corresponding rows from 'y' to maintain alignment with 'X'
    y = y.loc[log_data.index]

    # Group highly correlated features
    feature_groups = group_similar_features(log_data, correlation_threshold=0.9)
    aggregated_features, feature_mapping = aggregate_correlated_features(log_data, feature_groups)

    return aggregated_features, y, feature_mapping


def train_and_evaluate(X_train, y_train, X_test, y_test, params):
    """"
    Trains XGBoost model and evaluates performance
    """
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        objective='multi:softmax',
        num_class=len(np.unique(y_train)),
        **params
    )
    # TODO: change verbose to false
    # Fit model
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
    print('Confusion Matrix: ')
    print(cm_df)

    return accuracy, model


def cross_validate_model(X, y, params, n_splits=5):
    """
    Perform cross-validation and return average accuracy.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        accuracy, _ = train_and_evaluate(X_train, y_train, X_test, y_test, params)
        accuracies.append(accuracy)
    avg_accuracy = np.mean(accuracies)
    return avg_accuracy


# def generate_param_grids(param_space):
#     """
#     Generates all possible parameter combinations from the given parameter space.
#     """
#     keys = param_space.keys()
#     values = param_space.values()
#     return [dict(zip(keys, combo)) for combo in product(*values)]

def generate_param_grids(param_space):
    """
    Generates all possible parameter combinations from the given parameter space.
    """
    keys = param_space.keys()
    values = param_space.values()
    return [dict(zip(keys, combo)) for combo in product(*values)]


def optimize_hyperparameters(X_train, y_train, param_grid):
    """
    Uses GridSearchCV for hyperparameter tuning
    """
    model = xgb.XGBClassifier(use_label_encoder=False, objective='multi:softmax')
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

# def grid_search_worker(X_train, y_train, grid):
#     try:
#         accuracy = cross_validate_model(X_train, y_train, grid)
#         return grid, accuracy
#     except Exception as e:
#         print(f"Error for grid {grid}: {e}")
#         return grid, None
#
#
# def parallel_hyperparameter_tuning(X_train, y_train, param_space, n_jobs=4):
#     """
#     Parallelize hyperparameter tuning using multiprocessing.
#     """
#     param_grids = [dict(zip(param_space.keys(), combo)) for combo in product(*param_space.values())]
#     with Pool(processes=n_jobs) as pool:
#         results = pool.map(grid_search_worker, [(X_train, y_train, grid) for grid in param_grids])
#     return results
#

# def calculate_feature_importance(model, X_train):
#     """
#     Calculate feature importance
#     """
#     feature_importance = pd.DataFrame({
#         'feature': X_train.columns,
#         'importance': model.feature_importances_
#     }).sort_values(by='importance', ascending=False)
#
#     return feature_importance

def process_and_train_with_gridsearch(df_sum, param_spaces, output_file):
    """
    Main function for processing data, training model, and saving results
    """
    try:
        df_xgb = df_sum.copy()
        # df_xgb.drop(['Category', 'Duration'], axis=1, inplace=True)

        X, y, feature_mapping = preprocess_features(df_xgb)
        # y = df_xgb['Category']

        # Encode target labels as integers
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)  # Convert string labels to integers

        # Split data into training and test datasets
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.4, random_state=42)

        for i, param_space in enumerate(param_spaces):
            print(f"Starting GridSearch optimization round {i + 1}...")

            # GridSearchCV for the current param_space
            best_params = optimize_hyperparameters(X_train, y_train, param_space)
            print(f"Best parameters for round {i + 1}: {best_params}")

            # Evaluate model with the best parameters
            accuracy, model = train_and_evaluate(X_train, y_train, X_test, y_test, best_params)
            print(f"Accuracy after round {i + 1}: {accuracy * 100:.2f}%")

            if i == len(param_spaces) - 1:
                # Map aggregated features back to their original features
                feature_importance = pd.DataFrame({
                    'Features': X.columns.map(lambda col: ", ".join(map(str, feature_mapping.get(col, [col])))),
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)

                return feature_importance

                # # Final feature importance and save results
                # feature_importance = pd.DataFrame({
                #     'feature': X.columns,
                #     'importance': model.feature_importances_
                # }).sort_values(by='importance', ascending=False)
                # y_pred_decoded = label_encoder.inverse_transform(y_pred)
                # #feature_importance.to_csv(output_file, index=False)
                # #print(f"Feature importance saved to {output_file}.")
                # return feature_importance

        # # Optimize hyperparameters
        # best_params = optimize_hyperparameters(X_train, y_train, param_grid)
        # print(f'Best hyperparameters: {best_params}')
        #
        # # Train model with best hyperparameters
        # accuracy, model = train_and_evaluate(X_train, y_train, X_test, y_test, best_params)
        # print(f'Model Accuracy: {accuracy * 100:.2f}%')
        #
        # # Calculate and display feature importance
        # feature_importance = calculate_feature_importance(model, X_train)
        #
        # # Return results to main
        # return feature_importance
# def xgb_pipeline(df_sum, param_spaces):
#     """
#     Main function for processing data, training model, and saving results
#     """
#     try:
#         print(1)
#         #df_xgb = df_sum.copy()
#         #df_xgb.drop(['Category', 'Duration'], axis=1, inplace=True)
#         #df_xgb = df_sum.drop('Category', axis=1)
#         #y = df_sum['Category']
#
#         # Make sure any rows with missing values are dropped from both X and y
#         #df_xgb = df_xgb.dropna()
#         #y = y.loc[df_xgb.index]  # Make sure y is aligned with X
#
#         X, y = preprocess_features(df_sum)
#         #y = df_sum['Category']
#         print(f"Shape of X: {X.shape}")
#         print(f"Shape of y: {y.shape}")
#         print(2)
#         # Split data into training and test datasets
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
#
#         for i, param_space in enumerate(param_spaces):
#             print(f"Starting GridSearch optimization round {i + 1}...")
#             results = parallel_hyperparameter_tuning(X_train, y_train, param_space, n_jobs=4)
#             print(3)
#             # Find the best parameters
#             best_grid, best_score = max(results, key=lambda x: x[1] if x[1] is not None else -np.inf)
#             print(f"Best parameters for round {i + 1}: {best_grid} with score: {best_score:.4f}")
#             print(4)
#             # Train and evaluate with the best parameters
#             accuracy, model = train_and_evaluate(X_train, y_train, X_test, y_test, best_grid)
#             print(f"Accuracy for Grid {i + 1}: {accuracy:.2f}")
#             print(5)
#             if i == len(param_spaces) - 1:
#                 # Final feature importance and save results
#                 feature_importance = pd.DataFrame({
#                     'feature': X.columns,
#                     'importance': model.feature_importances_
#                 }).sort_values(by='importance', ascending=False)
#                 #feature_importance.to_csv(output_file, index=False)
#                 #print(f"Feature importance saved to {output_file}.")
#                 return feature_importance
#
#         # # Optimize hyperparameters
#         # best_params = optimize_hyperparameters(X_train, y_train, param_grid)
#         # print(f'Best hyperparameters: {best_params}')
#         #
#         # # Train model with best hyperparameters
#         # accuracy, model = train_and_evaluate(X_train, y_train, X_test, y_test, best_params)
#         # print(f'Model Accuracy: {accuracy * 100:.2f}%')
#         #
#         # # Calculate and display feature importance
#         # feature_importance = calculate_feature_importance(model, X_train)
#         #
#         # # Return results to main
#         # return feature_importance

    except Exception as e:
        print('Error: ', e)
        print('Stack trace: ', traceback.format_exc())


def xgboost(df_sum, parameters, output_file):
    print("Entered xgboost function.")
    param_spaces = [
        {'n_estimators': [300, 400, 500], 'learning_rate': [0.1, 0.2, 0.3]},
        {'max_depth': [3, 5, 7], 'gamma': [0.05, 0.1, 0.2]},
        {'subsample': [0.8, 0.9], 'colsample_bytree': [0.7, 0.8, 0.9, 1.0]},]

    feature_importance = process_and_train_with_gridsearch(df_sum, param_spaces, output_file)

    # Save or print the results
    saveXGB = output_file + '_XGB.xlsx'
    print('Saving XGB output to ' + saveXGB + '...')
    with pd.ExcelWriter(saveXGB, engine='xlsxwriter') as workbook:
        feature_importance.to_excel(workbook, sheet_name='Feature importance', index=False)


# def xgboost(df_sum, parameters, savefile):
#     param_spaces = [
#         {'n_estimators': [300, 400, 500], 'learning_rate': [0.1, 0.2, 0.3]},
#         {'max_depth': [3, 5, 7], 'gamma': [0.05, 0.1, 0.2]},
#         {'subsample': [0.8, 0.9], 'colsample_bytree': [0.7, 0.8, 0.9, 1.0]},
#     ]
#
#     # Preprocess features
#     X, _ = preprocess_features(df_sum)
#     y = df_sum['Object Type']
#
#     # Split into train-test data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
#
#     # Perform parallel hyperparameter tuning
#     results = parallel_hyperparameter_tuning(param_spaces, X_train, y_train)
#
#     # Evaluate models from the results
#     for i, (best_params, best_score) in enumerate(results):
#         print(f"Grid {i + 1}: Best Params: {best_params}, Score: {best_score:.4f}")
#
#         # Train and evaluate with best parameters
#         print(f"Training and evaluating model for Grid {i + 1}...")
#         accuracy, model = train_and_evaluate(X_train, y_train, X_test, y_test, best_params)
#         print(f"Accuracy for Grid {i + 1}: {accuracy:.2f}")
#
#     feature_importance = process_and_train_with_gridsearch(df_sum, param_spaces, output_file)



#if __name__ == "__main__":
   # datasets = []
# def xgboost(df_sum, parameters, savefile):
#     try:
#         # figure out feature correlation
#         df_xgb = df_sum.copy()
#         df_features = df_xgb.drop('Object Type', axis=1).copy()  # Data used to predict object type
#
#         # y = df_xgb['Object Type'].copy()  # Known classifications
#         features = df_xgb.columns
#
#         def signed_log_transformation(x):
#             return np.sign(x)*np.log(np.abs(x) + 1)
#
#         log_data = signed_log_transformation(df_features).dropna()
#
#         results = []
#
#         # Iterate over all unique feature pairs
#         for feature1, feature2 in combinations(log_data.columns, 2):
#             x = log_data[[feature1]]  # Note the double brackets to keep it 2D
#             y = log_data[feature2]
#
#             # Fit the linear regression model
#             model = LinearRegression().fit(x, y)
#             y_pred = model.predict(x)
#
#             # Calculate R-squared
#             r2 = r2_score(y, y_pred)
#
#             # Store the results
#             results.append({'X': feature1, 'Y': feature2, 'R_squared': r2})
#
#         # Convert the results list to a DataFrame
#         results_df = pd.DataFrame(results)
#
#         # Save results to a CSV file
#         results_df.to_csv('pairwise_regression_results.csv', index=False)
#
#         # Display the results
#         print(results_df)
#
#         df_xgb = df_sum.copy()
#         df_xgb.drop(['Object ID', 'Duration'],
#                     axis=1, inplace=True)
#
#         X = df_xgb.drop('Object Type', axis=1).copy()  # Data used to predict object type
#         y = df_xgb['Object Type'].copy()  # Known classifications

        # Check for super large/infinite values
        # inf_indices = np.isinf(X)
        # row_indices, col_indices = np.where(inf_indices)
        # large_values_indices = np.column_stack(np.where(X > 1e6))
        # Combine indices of inf values and large values (which i don't think should exist but whatevs)
        # all_indices = np.vstack((np.column_stack((row_indices, col_indices)), large_values_indices))
        # Remove duplicates
        # unique_indices = np.unique(all_indices, axis=0)
        # print(unique_indices)
        #
        # # Get number of different object types
        # num_classes = len(np.unique(y))
        #
        # # Split data into training and testing datasets
        # # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        # # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # type_xgb = xgb.XGBClassifier(
        #     use_label_encoder=False,
        #     objective='multi:softmax',
        #     num_class=num_classes,
        #     n_estimators=1000,
        #     learning_rate=0.1,
        #     max_depth=6,
        #     gamma=1,
        #     subsample=0.8,
        #     colsample_bytree=0.8,
        #     missing=np.inf
        # )
        #
        # # Perform cross-validation to find the best parameters
        # param_grid = {
        #     'learning_rate': [0.01, 0.1, 0.2],
        #     'max_depth': [4, 6, 8],
        #     'gamma': [0, 0.1, 0.5],
        #     'subsample': [0.8, 1.0],
        #     'colsample_bytree': [0.8, 1.0]
        # }
        #
        # grid_search = GridSearchCV(type_xgb, param_grid, scoring='accuracy', cv=5)
        # grid_search.fit(X_train, y_train)



        # Cross-validation
        # kfold = StratifiedKFold(n_splits=5)
        # results = cross_val_score(best_xgb, X, y, cv=kfold)
        # print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

        # eval_set = [(X_val, y_val)]
        # Build XGBoost model
        # type_xgb = xgb.XGBClassifier(
        #     use_label_encoder=False,
        #     objective='multi:softmax',
        #     num_class=num_classes,
        #     n_estimators=1000,
        #     learning_rate=0.1,
        #     max_depth=6,  # Initial setting
        #     gamma=0.1,  # Initial setting
        #     subsample=0.8,
        #     colsample_bytree=0.8,
        #     missing=np.inf,
        #     early_stopping_rounds=10,
        #     eval_metric='mlogloss')
        #
        # type_xgb.fit(X_train,
        #              y_train,
        #              verbose=False,
        #              eval_set=[(X_test, y_test)])
        #
        # y_pred = type_xgb.predict(X_test)
        #
        # cm = confusion_matrix(y_test, y_pred, labels=type_xgb.classes_)
        # cm_df = pd.DataFrame(cm, index=type_xgb.classes_, columns=type_xgb.classes_)
        # print("Confusion Matrix:")
        # print(cm_df)
        #
        # correctness = (y_test == y_pred).astype(int)
        # total_samples = len(y_test)
        # correct_predictions = np.sum(correctness)
        # incorrect_predictions = total_samples - correct_predictions
        #
        # correct_percentage = (correct_predictions / total_samples) * 100
        #
        # # Create a dataframe to display the percentages
        # print([f'{correct_percentage:.2f}%'])

       #
       #  # ROUND 1 optimizing parameters
       #  param_grid = {
       #      'n_estimators': [300, 400, 500, 600, 700, 800, 900, 1000],
       #      'learning_rate': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
       #      # optimal: 300, 0.25
       #
       #
       #  }
       #
       # # ROUND 2 optimizing parameters
       #  param_grid = {
       #      'n_estimators': [150, 200, 250, 300, 350],
       #      'learning_rate': [0.2, 0.2125, 0.225, 0.2375, 0.25, 0.2625, 0.275, 0.2875, 0.3]
       #      # optimal: 150, 0.25
       #  }
       #
       #  # ROUND 3 optimizing parameters
       #  param_grid = {
       #      'max_depth': [3, 4, 5, 6, 7, 8, 9],
       #      'gamma': [0.2, 0.1, 0.05, 0.01, 0.001, 0],
       #      # optimal: 5, 0.05
       #  }
       #
       #  # ROUND 4 optimizing parameters
       #  param_grid = {
       #      'subsample': [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
       #      'colsample_bytree': [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
       #      # optimal: 0.8, 0.8
       #  }


        # # Using random subset of data and features for speed/preventing overfitting
        # optimal_params = GridSearchCV(
        #     estimator=xgb.XGBClassifier(objective='multi:softmax',
        #                                 subsample=0.8,
        #                                 colsample_bytree=0.8,
        #                                 early_stopping_rounds=10,
        #                                 eval_metric='mlogloss',
        #                                 missing=np.inf,
        #                                 num_class=num_classes,
        #                                 n_estimators=150,
        #                                 max_depth=5,
        #                                 gamma=0.05,
        #                                 learning_rate=0.25),
        #
        #     param_grid=param_grid,
        #     scoring=make_scorer(accuracy_score),
        #     verbose=0,
        #     n_jobs=-1,
        #     cv=5)
        #
        # optimal_params.fit(X_train,
        #                    y_train,
        #                    eval_set=[(X_val, y_val)],
        #                    verbose=False)
        #
        # print(optimal_params.best_params_)
    #
    #
    #     # Updated model with optimized parameters
    #     type_xgb = xgb.XGBClassifier(
    #         use_label_encoder=False,
    #         objective='multi:softmax',
    #         num_class=num_classes,
    #         n_estimators=150,
    #         learning_rate=0.25,
    #         max_depth=5,
    #         gamma=0.05,
    #         subsample=0.8,
    #         colsample_bytree=0.8,
    #         missing=np.inf,
    #         early_stopping_rounds=10,
    #         eval_metric='mlogloss'
    #         )
    #
    #     # Set early stopping parameters after initialization
    #     type_xgb.set_params()
    #
    #     type_xgb.fit(X_train,
    #                  y_train,
    #                  verbose=False,
    #                  eval_set=[(X_test, y_test)])
    #
    #     y_pred = type_xgb.predict(X_test)
    #
    #     cm = confusion_matrix(y_test, y_pred, labels=type_xgb.classes_)
    #     cm_df = pd.DataFrame(cm, index=type_xgb.classes_, columns=type_xgb.classes_)
    #     print("Confusion Matrix:")
    #     print(cm_df)
    #
    #     correctness = (y_test == y_pred).astype(int)
    #     total_samples = len(y_test)
    #     correct_predictions = np.sum(correctness)
    #     incorrect_predictions = total_samples - correct_predictions
    #
    #     correct_percentage = (correct_predictions / total_samples) * 100
    #
    #     # Create a dataframe to display the percentages
    #     print([f'{correct_percentage:.2f}%'])
    #
    #     # kfold = StratifiedKFold(n_splits=5)
    #     # results = cross_val_score(type_xgb, X, y, cv=kfold, error_score='raise')
    #
    #     # print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    #
    #     # feature_important = type_xgb.get_booster().get_score(importance_type='weight')
    #     # keys = list(feature_important.keys())
    #     # values = list(feature_important.values())
    #     #
    #     # data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
    #     # data.nlargest(40, columns="score").plot(kind='barh', figsize=(20, 10))  ## plot top 40 features
    #     # pyplot.show()
    #
    #
    #     sorted_idx = np.argsort(type_xgb.feature_importances_)[::-1]
    #     for index in sorted_idx:
    #         print([X_train.columns[index], type_xgb.feature_importances_[index]])
    #
    #     feature_importance = pd.DataFrame({'feature': X_train.columns, 'importance': type_xgb.feature_importances_})
    #     feature_importance = feature_importance.sort_values(by='importance', ascending=True)
    #
    #     fig = px.bar(feature_importance, x='importance', y='feature', orientation='h')
    #     #fig.update_layout(title='Feature Importance<br><sup>Migrate3D ' + '{:%Y_%m_%d}'.format(date.today()) +
    #                             #'<br><sup>' + os.path.basename(parameters['infile_segments']).split('/')[-1]])
    #     fig.show()
    #
    #
    #
    # except Exception as e:
    #     print('Error:', e)
    #     print('Stack trace:', traceback.format_exc())
    #
    #
    #

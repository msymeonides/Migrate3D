import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score  # cross validation
from sklearn.metrics import confusion_matrix, r2_score
from sklearn.preprocessing import StandardScaler
import traceback
import plotly.express as px
from sklearn.linear_model import LinearRegression
from itertools import combinations
from sklearn.metrics import mean_squared_error
from datetime import date
import os


def xgboost(df_sum, parameters, savefile):
    try:
        # figure out feature correlation
        df_xgb = df_sum.copy()
        df_features = df_xgb.drop('Object Type', axis=1).copy()  # Data used to predict object type

        # y = df_xgb['Object Type'].copy()  # Known classifications
        features = df_xgb.columns

        def signed_log_transformation(x):
            return np.sign(x)*np.log(np.abs(x) + 1)

        log_data = signed_log_transformation(df_features).dropna()

        results = []

        # Iterate over all unique feature pairs
        for feature1, feature2 in combinations(log_data.columns, 2):
            x = log_data[[feature1]]  # Note the double brackets to keep it 2D
            y = log_data[feature2]

            # Fit the linear regression model
            model = LinearRegression().fit(x, y)
            y_pred = model.predict(x)

            # Calculate R-squared
            r2 = r2_score(y, y_pred)

            # Store the results
            results.append({'X': feature1, 'Y': feature2, 'R_squared': r2})

        # Convert the results list to a DataFrame
        results_df = pd.DataFrame(results)

        # Save results to a CSV file
        results_df.to_csv('pairwise_regression_results.csv', index=False)

        # Display the results
        print(results_df)

        df_xgb = df_sum.copy()
        df_xgb.drop(['Object ID', 'Duration'],
                    axis=1, inplace=True)

        X = df_xgb.drop('Object Type', axis=1).copy()  # Data used to predict object type
        y = df_xgb['Object Type'].copy()  # Known classifications

        # Check for super large/infinite values
        # inf_indices = np.isinf(X)
        # row_indices, col_indices = np.where(inf_indices)
        # large_values_indices = np.column_stack(np.where(X > 1e6))
        # Combine indices of inf values and large values (which i don't think should exist but whatevs)
        # all_indices = np.vstack((np.column_stack((row_indices, col_indices)), large_values_indices))
        # Remove duplicates
        # unique_indices = np.unique(all_indices, axis=0)
        # print(unique_indices)

        # Get number of different object types
        num_classes = len(np.unique(y))

        # Split data into training and testing datasets
        # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

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


        # ROUND 1 optimizing parameters
        param_grid = {
            'n_estimators': [300, 400, 500, 600, 700, 800, 900, 1000],
            'learning_rate': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
            # optimal: 300, 0.25


        }

       # ROUND 2 optimizing parameters
        param_grid = {
            'n_estimators': [150, 200, 250, 300, 350],
            'learning_rate': [0.2, 0.2125, 0.225, 0.2375, 0.25, 0.2625, 0.275, 0.2875, 0.3]
            # optimal: 150, 0.25
        }

        # ROUND 3 optimizing parameters
        param_grid = {
            'max_depth': [3, 4, 5, 6, 7, 8, 9],
            'gamma': [0.2, 0.1, 0.05, 0.01, 0.001, 0],
            # optimal: 5, 0.05
        }

        # ROUND 4 optimizing parameters
        param_grid = {
            'subsample': [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
            'colsample_bytree': [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
            # optimal: 0.8, 0.8
        }


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


        # Updated model with optimized parameters
        type_xgb = xgb.XGBClassifier(
            use_label_encoder=False,
            objective='multi:softmax',
            num_class=num_classes,
            n_estimators=150,
            learning_rate=0.25,
            max_depth=5,
            gamma=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            missing=np.inf,
            early_stopping_rounds=10,
            eval_metric='mlogloss'
            )

        # Set early stopping parameters after initialization
        type_xgb.set_params()

        type_xgb.fit(X_train,
                     y_train,
                     verbose=False,
                     eval_set=[(X_test, y_test)])

        y_pred = type_xgb.predict(X_test)

        cm = confusion_matrix(y_test, y_pred, labels=type_xgb.classes_)
        cm_df = pd.DataFrame(cm, index=type_xgb.classes_, columns=type_xgb.classes_)
        print("Confusion Matrix:")
        print(cm_df)

        correctness = (y_test == y_pred).astype(int)
        total_samples = len(y_test)
        correct_predictions = np.sum(correctness)
        incorrect_predictions = total_samples - correct_predictions

        correct_percentage = (correct_predictions / total_samples) * 100

        # Create a dataframe to display the percentages
        print([f'{correct_percentage:.2f}%'])

        # kfold = StratifiedKFold(n_splits=5)
        # results = cross_val_score(type_xgb, X, y, cv=kfold, error_score='raise')

        # print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

        # feature_important = type_xgb.get_booster().get_score(importance_type='weight')
        # keys = list(feature_important.keys())
        # values = list(feature_important.values())
        #
        # data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
        # data.nlargest(40, columns="score").plot(kind='barh', figsize=(20, 10))  ## plot top 40 features
        # pyplot.show()


        sorted_idx = np.argsort(type_xgb.feature_importances_)[::-1]
        for index in sorted_idx:
            print([X_train.columns[index], type_xgb.feature_importances_[index]])

        feature_importance = pd.DataFrame({'feature': X_train.columns, 'importance': type_xgb.feature_importances_})
        feature_importance = feature_importance.sort_values(by='importance', ascending=True)

        fig = px.bar(feature_importance, x='importance', y='feature', orientation='h')
        #fig.update_layout(title='Feature Importance<br><sup>Migrate3D ' + '{:%Y_%m_%d}'.format(date.today()) +
                                #'<br><sup>' + os.path.basename(parameters['infile_segments']).split('/')[-1]])
        fig.show()



    except Exception as e:
        print('Error:', e)
        print('Stack trace:', traceback.format_exc())




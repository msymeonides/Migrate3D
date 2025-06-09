import numpy as np
import pandas as pd
import xgboost.sklearn as xgb
import itertools
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif,SelectKBest
from itertools import product
import traceback
import shap
import matplotlib.pyplot as plt

from shared_state import messages, thread_lock


def signed_log_transformation(x):
    return np.sign(x) * np.log(np.abs(x) + 1)

def select_categories(df, parameters):
    # Filter if specific categories are given
    filter_ = parameters['pca_filter']
    if filter_ is not None:
        filter_ = [int(x) for x in filter_]
        df = df[df['Category'].isin(filter_)]
    with thread_lock:
        messages.append('Starting XGB...')
    df = df.dropna()
    df = df.drop(
        labels=['Duration', 'Path Length', 'Final Euclidean', 'Straightness', 'Velocity filtered Mean',
                'Velocity Mean', 'Velocity Median', 'Acceleration Filtered Mean', 'Acceleration Mean',
                'Absolute Acceleration Mean', 'Absolute Acceleration Median', 'Acceleration Filtered Mean',
                'Acceleration Filtered Median', 'Acceleration Filtered Standard Deviation',
                'Acceleration Median', 'Overall Euclidean Median', 'Convex Hull Volume'], axis=1)
    return df

# def select_important_features(X, y, k=20):
#     """
#     Selects the top k most informative features using mutual information.
#     """
#     selector = SelectKBest(score_func=mutual_info_classif, k=k)
#     X_new = selector.fit_transform(X, y)
#     selected_features = X.columns[selector.get_support()]
#     return pd.DataFrame(X_new, columns=selected_features), selected_features
#

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
        group_features_label = ", ".join(group)  # Concatenate original names for labels
        aggregated_data.append(df[list(group)].mean(axis=1))  # Aggregate the group (mean)
        feature_mapping[group_features_label] = list(group)  # Store the original feature names for the group

    # Create aggregated dataframe with group names as column headers
    aggregated_df = pd.DataFrame(aggregated_data).T
    aggregated_df.columns = [", ".join(group) for group in feature_groups]

    return aggregated_df, feature_mapping


def preprocess_features(df, parameters, k=20):
    """
    Preprocesses the feature data (log transformation and correlation grouping)
    """
    df_features = df.drop(['Object ID', 'Category'], axis=1)
    y = df['Category']

    # Apply signed log transformation and drop any rows with NaN values
    log_data = signed_log_transformation(df_features)
    log_data = log_data.dropna()

    # Ensure that we drop the corresponding rows from 'y' to maintain alignment with 'X'
    y = y.loc[log_data.index]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(log_data)
    scaled_df = pd.DataFrame(scaled_data, columns=log_data.columns)

    # Group highly correlated features
    #selected_features, feature_names = aggregate_correlated_features(scaled_df, y, k=k)
    feature_groups = group_similar_features(scaled_df, correlation_threshold=0.9)
    aggregated_features, feature_mapping = aggregate_correlated_features(scaled_df, feature_groups)

    return aggregated_features, y, feature_mapping


def train_and_evaluate(X_train, y_train, X_test, y_test, params):
    """
    Trains XGBoost model and evaluates performance
    """
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        eval_metric='mlogloss',
        num_class=len(np.unique(y_train)),
        early_stopping_rounds=15,
        n_jobs=-1,
        **params
    )

    eval_set = [(X_train, y_train), (X_test, y_test)]

    # Fit model
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Confusion matrix
    # cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    # cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Plot summary plot
    #shap.summary_plot(shap_values, X_test, plot_type="bar")
    #plt.show()

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


def generate_param_grids(param_space):
    """
    Generates all possible parameter combinations from the given parameter space.
    """
    keys = param_space.keys()
    values = param_space.values()
    return [dict(zip(keys, combo)) for combo in product(*values)]


def optimize_hyperparameters(X_train, y_train):
    """
    Uses RandomizedSearchCV for hyperparameter tuning
    """
    param_grid = {
        'n_estimators': [50, 100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 5, 7, 9, 11],
        'gamma': [0.0, 0.05, 0.1, 0.2, 0.3],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'reg_alpha': [0.0, 0.1, 0.5],
        'reg_lambda': [0.0, 0.1, 0.5]
    }
    model = xgb.XGBClassifier(objective='multi:softmax',
                              eval_metric='mlogloss',
                              num_class=len(np.unique(y_train)),
                              n_jobs=-1)

    random_search = RandomizedSearchCV(
        model, param_distributions=param_grid,
        #n_iter=25,  # Reduce the number of combinations
        cv=3, scoring='accuracy', verbose=0, n_jobs=-1, random_state=42
    )

    random_search.fit(X_train, y_train)
    return random_search.best_params_
    #grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    #grid_search.fit(X_train, y_train)
    #return grid_search.best_params_


def process_and_train_with_gridsearch(df_xgb, param_spaces, parameters):
    """
    Main function for processing data, training model, and saving results
    """
    try:
        df_xgb = select_categories(df_xgb, parameters)

        X, y, feature_mapping = preprocess_features(df_xgb, parameters)


        # Encode target labels as integers
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)  # Convert string labels to integers

        # Split data into training and test datasets
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.4, random_state=42)

        # for i, param_space in enumerate(param_spaces):
            #    with thread_lock:
            #       messages.append(f"Starting GridSearch optimization round {i + 1}...")

        best_params = optimize_hyperparameters(X_train, y_train)

            # GridSearchCV for the current param_space
            # best_params = optimize_hyperparameters(X_train, y_train, param_space)
            # with thread_lock:
            #   messages.append(f"Best parameters for round {i + 1}: {best_params}")

            # Evaluate model with the best parameters
        accuracy, model = train_and_evaluate(X_train, y_train, X_test, y_test, best_params)

        feature_importance = pd.DataFrame({
            'Features': [",".join(map(str, feature_mapping[col])) for col in X.columns],
            'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)

        return feature_importance

    except Exception as e:
        print('Error: ', e)
        print('Stack trace: ', traceback.format_exc())


def perform_xgboost_comparisons(data, category_col, feature_cols, savefile, category_filter=None):
    # Extract unique categories
    unique_categories = data[category_col].unique()

    # Apply category filter if provided
    if category_filter:
        unique_categories = [cat for cat in unique_categories if cat in category_filter]

    # Generate all two-way combinations of categories
    category_pairs = list(itertools.combinations(unique_categories, 2))

    # Initialize Excel writer
    with pd.ExcelWriter(savefile, engine='xlsxwriter') as writer:
        for cat1, cat2 in category_pairs:
            # Filter data for the two categories
            pair_data = data[data[category_col].isin([cat1, cat2])]

            # Encode categories as binary labels
            label_encoder = LabelEncoder()
            pair_data['label'] = label_encoder.fit_transform(pair_data[category_col])

            # Split data into features and labels
            X = pair_data[feature_cols]
            y = pair_data['label']

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Train XGBoost model
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            model.fit(X_train, y_train)

            # Evaluate model
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()

            # Save classification report and feature importance
            report_df.to_excel(writer, sheet_name=f'{cat1}_vs_{cat2}_Report')
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            feature_importance.to_excel(writer, sheet_name=f'{cat1}_vs_{cat2}_Features', index=False)

def xgboost(df_sum, parameters, output_file):
                try:
                    # Step 1: Hyperparameter Optimization on the Entire Dataset
                    param_spaces = [
                        {'n_estimators': [100, 200, 300, 400, 500], 'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3]},
                        {'max_depth': [3, 5, 7, 9], 'gamma': [0.0, 0.05, 0.1, 0.2]},
                        {'subsample': [0.6, 0.7, 0.8, 0.9, 1.0], 'colsample_bytree': [0.7, 0.8, 0.9, 1.0]},
                        {'min_child_weight': [1, 3, 5], 'reg_alpha': [0.0, 0.1, 0.5], 'reg_lambda': [0.0, 0.1, 0.5, 1.0]}
                    ]

                    # Perform grid search and train the model
                    feature_importance = process_and_train_with_gridsearch(df_sum, param_spaces, parameters)

                    # Save feature importance results
                    saveXGB = output_file + '_XGB.xlsx'
                    with pd.ExcelWriter(saveXGB, engine='xlsxwriter') as workbook:
                        feature_importance.to_excel(workbook, sheet_name='Feature importance', index=False)
                    with thread_lock:
                        messages.append(f"Grid search results saved to {saveXGB}")

                    # Step 2: Pairwise Comparisons
                    category_col = 'Category'  # Replace with the actual category column name
                    feature_cols = [col for col in df_sum.columns if col not in ['Object ID', 'Category']]  # Adjust as needed
                    savefile = output_file + '_XGB_Comparisons.xlsx'

                    perform_xgboost_comparisons(
                        data=df_sum,
                        category_col=category_col,
                        feature_cols=feature_cols,
                        savefile=savefile,
                        category_filter=parameters.get('pca_filter')  # Pass PCA filter if available
                    )
                    with thread_lock:
                        messages.append(f"Pairwise comparisons saved to {savefile}")

                except Exception as e:
                    with thread_lock:
                        messages.append(f"Error during XGBoost analysis: {str(e)}")

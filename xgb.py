import numpy as np
import pandas as pd
import xgboost.sklearn as xgb
import itertools
from xgboost import XGBClassifier
from xgboost.core import XGBoostError
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from itertools import product

from shared_state import messages, thread_lock, complete_progress_step
from PCA import apply_category_filter

class XGBAbortException(Exception):
    pass

def error():
    with thread_lock:
        messages.append('Error in XGBoost, bypassing...')
        messages.append('Usually these errors are due to too low N within one of the object categories. Use the category filter to exclude these low-N categories.')
        messages.append('')
    complete_progress_step("XGB")
    raise XGBAbortException()

def signed_log_transformation(x):
    return np.sign(x) * np.log(np.abs(x) + 1)

def select_categories(df, parameters):
    df = apply_category_filter(df, parameters.get('pca_filter'))
    with thread_lock:
        messages.append('Starting XGBoost...')
    df = df.dropna()
    df = df.drop(
        labels=['Duration', 'Path Length', 'Final Euclidean', 'Straightness', 'Velocity filtered Mean',
                'Velocity Mean', 'Velocity Median', 'Acceleration Filtered Mean', 'Acceleration Mean',
                'Absolute Acceleration Mean', 'Absolute Acceleration Median', 'Acceleration Filtered Mean',
                'Acceleration Filtered Median', 'Acceleration Filtered Standard Deviation',
                'Acceleration Median', 'Overall Euclidean Median', 'Convex Hull Volume'], axis=1)
    return df

def group_similar_features(df_features, correlation_threshold=0.9):
    corr_matrix = df_features.corr().abs()
    feature_groups = []
    used_features = set()

    for feature in corr_matrix.columns:
        if feature not in used_features:
            correlated_features = set(corr_matrix.index[corr_matrix[feature] > correlation_threshold])
            feature_groups.append(correlated_features)
            used_features.update(correlated_features)
    return feature_groups


def aggregate_correlated_features(df, feature_groups):
    aggregated_data = []
    feature_mapping = {}

    for i, group in enumerate(feature_groups):
        group_features_label = ", ".join(group)
        aggregated_data.append(df[list(group)].mean(axis=1))
        feature_mapping[group_features_label] = list(group)

    aggregated_df = pd.DataFrame(aggregated_data).T
    aggregated_df.columns = [", ".join(group) for group in feature_groups]

    return aggregated_df, feature_mapping


def preprocess_features(df):
    try:
        df_features = df.drop(['Object ID', 'Category'], axis=1)
        y = df['Category']

        log_data = signed_log_transformation(df_features)
        log_data = log_data.dropna()

        y = y.loc[log_data.index]

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(log_data)
        scaled_df = pd.DataFrame(scaled_data, columns=log_data.columns)

        feature_groups = group_similar_features(scaled_df, correlation_threshold=0.9)
        aggregated_features, feature_mapping = aggregate_correlated_features(scaled_df, feature_groups)

        return aggregated_features, y, feature_mapping
    except XGBAbortException:
        raise
    except KeyError:
        error()
    except Exception:
        error()


def train_and_evaluate(X_train, y_train, X_test, y_test, params):
    try:
        model = xgb.XGBClassifier(
            objective='multi:softmax',
            eval_metric='mlogloss',
            num_class=len(np.unique(y_train)),
            early_stopping_rounds=15,
            n_jobs=-1,
            **params
        )

        eval_set = [(X_train, y_train), (X_test, y_test)]
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy, model

    except XGBAbortException:
        raise
    except XGBoostError:
        error()
    except Exception:
        error()


def cross_validate_model(X, y, params, n_splits=5):
    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        accuracies = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            accuracy, _ = train_and_evaluate(X_train, y_train, X_test, y_test, params)
            accuracies.append(accuracy)
        return np.mean(accuracies)
    except XGBAbortException:
        raise
    except ValueError:
        error()
    except Exception:
        error()


def generate_param_grids(param_space):
    keys = param_space.keys()
    values = param_space.values()
    return [dict(zip(keys, combo)) for combo in product(*values)]


def optimize_hyperparameters(X_train, y_train):
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

    try:
        random_search = RandomizedSearchCV(
            model, param_distributions=param_grid,
            cv=3, scoring='accuracy', verbose=0, n_jobs=-1, random_state=42
        )
        random_search.fit(X_train, y_train)
        return random_search.best_params_

    except Exception:
        return {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'gamma': 0.0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0
        }

def process_and_train_with_gridsearch(df_xgb, parameters):
    try:
        df_xgb = select_categories(df_xgb, parameters)
        X, y, feature_mapping = preprocess_features(df_xgb)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        if len(X) < 2:
            with thread_lock:
                messages.append("Not enough samples to split data for training/testing. Skipping XGBoost analysis.")
                messages.append('')
            complete_progress_step("XGB")
            return None, X, None, None

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.4, random_state=42)
        best_params = optimize_hyperparameters(X_train, y_train)
        accuracy, model = train_and_evaluate(X_train, y_train, X_test, y_test, best_params)
        y_pred = model.predict(X_test)

        feature_importance = pd.DataFrame({
            'Features': [",".join(map(str, feature_mapping[col])) for col in X.columns],
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        return feature_importance, X, y_test, y_pred

    except XGBAbortException:
        raise
    except Exception:
        error()


def perform_xgboost_comparisons(data, category_col, aggregated_features, writer, parameters):
    if category_col not in data.columns:
        raise KeyError(f"'{category_col}' column is missing from the DataFrame.")

    data = apply_category_filter(data, parameters.get('pca_filter'))
    aggregated_features = aggregated_features.reindex(data.index).dropna()
    labels = data[category_col]
    unique_categories = data[category_col].unique()
    category_pairs = list(itertools.combinations(unique_categories, 2))

    for cat1, cat2 in category_pairs:
        pair_indices = labels.isin([cat1, cat2])
        pair_data = aggregated_features.reindex(labels.index)[pair_indices]
        pair_labels = labels[pair_indices]

        if pair_data.empty or pair_labels.empty:
            continue

        label_encoder = LabelEncoder()
        pair_labels_encoded = label_encoder.fit_transform(pair_labels)
        X_train, X_test, y_train, y_test = train_test_split(
            pair_data, pair_labels_encoded, test_size=0.3, random_state=42
        )
        model = XGBClassifier(eval_metric='logloss')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        feature_importance = pd.DataFrame({
            'Feature': pair_data.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        feature_importance.to_excel(writer, sheet_name=f'{cat1}_vs_{cat2}_Features', index=False)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_excel(writer, sheet_name=f'{cat1}_vs_{cat2}_Report')


def xgboost(df_sum, parameters, output_file):
    try:
        feature_importance, aggregated_features, y_test, y_pred = process_and_train_with_gridsearch(df_sum, parameters)
        saveXGB = output_file + '_XGB.xlsx'
        category_col = 'Category'

        with pd.ExcelWriter(saveXGB, engine='xlsxwriter') as writer:
            feature_importance.to_excel(writer, sheet_name='Dataset Features', index=False)

            if y_test is not None and y_pred is not None:
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose()
                report_df.to_excel(writer, sheet_name='Dataset Report')

            perform_xgboost_comparisons(
                data=df_sum,
                category_col=category_col,
                aggregated_features=aggregated_features,
                writer=writer,
                parameters=parameters
            )

        with thread_lock:
            msg = " XGBoost done."
            messages[-1] += msg
            messages.append('')
        complete_progress_step("XGB")
    except XGBAbortException:
        raise
    except Exception:
        error()
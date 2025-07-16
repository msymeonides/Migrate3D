import itertools
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import scikit_posthocs as sp
from scipy import stats
import time
import xgboost.sklearn as xgb
from xgboost import XGBClassifier

from shared_state import messages, thread_lock, complete_progress_step

class XGBAbortException(Exception):
    pass

min_required_pca = 5            # Minimum number of samples for PCA
min_required_xgb = 10           # Minimum number of samples for XGBoost
min_required_train_test = 20    # Minimum for using Train-Test Split method in XGBoost
test_size = 0.4                 # Proportion of data to use for testing when using Train-Test Split method in XGBoost
variance_threshold = 0.01       # Threshold for near-zero variance feature removal
correlation_threshold = 0.95    # Pearson correlation threshold. Features with correlation above this threshold will be grouped together.
random_seed = 42                # Random seed for reproducibility

def safe_ml_operation(operation_func, error_step=None, *args, **kwargs):
    try:
        return operation_func(*args, **kwargs)
    except XGBAbortException:
        raise
    except Exception:
        if error_step:
            complete_progress_step(error_step)

def remove_near_zero_variance(df, threshold=None):
    if threshold is None:
        threshold = variance_threshold
    selector = VarianceThreshold(threshold=threshold)
    feature_mask = selector.fit(df).get_support()
    removed_features = df.columns[~feature_mask].tolist()
    df_filtered = df.loc[:, feature_mask]

    return df_filtered, removed_features

def apply_category_filter(df, cat_filter):
    if cat_filter is None:
        return df
    if pd.api.types.is_numeric_dtype(df['Category']):
        try:
            filter_vals = [int(x) for x in cat_filter]
        except ValueError:
            filter_vals = cat_filter
        df.loc[:, 'Category'] = df['Category'].astype(int)
    else:
        filter_vals = [str(x) for x in cat_filter]
    filtered_df = df[df['Category'].isin(filter_vals)]
    if filtered_df.empty:
        with thread_lock:
            messages.append('No data available for the selected categories.')

    return filtered_df

def detect_correlated_features(df, threshold):
    corr_matrix = df.corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)
    pairs = [
        (i, j, corr_matrix.loc[i, j])
        for i in corr_matrix.columns
        for j in corr_matrix.columns
        if i < j and corr_matrix.loc[i, j] >= threshold
    ]
    pairs.sort(key=lambda x: -x[2])

    assigned = set()
    groups = []
    for i, j, corr in pairs:
        if i not in assigned and j not in assigned:
            groups.append({i, j})
            assigned.update([i, j])
        elif i not in assigned:
            for group in groups:
                if j in group:
                    group.add(i)
                    assigned.add(i)
                    break
        elif j not in assigned:
            for group in groups:
                if i in group:
                    group.add(j)
                    assigned.add(j)
                    break

    for col in df.columns:
        if col not in assigned:
            groups.append({col})

    groups = [sorted(list(g)) for g in groups]
    return groups

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

def create_correlation_sheet(df, feature_groups, threshold, writer):
    corr_matrix = df.corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)

    feature_to_group = {}
    for i, group in enumerate(feature_groups):
        for feature in group:
            feature_to_group[feature] = i if len(group) > 1 else None

    correlation_decisions = []
    for i in corr_matrix.columns:
        for j in corr_matrix.columns:
            if i < j:
                corr_value = corr_matrix.loc[i, j]
                same_group = feature_to_group[i] == feature_to_group[j]
                meets_threshold = corr_value >= threshold
                correlation_decisions.append({
                    'Feature_1': i,
                    'Feature_2': j,
                    'Correlation': corr_value,
                    'Meets_Threshold': meets_threshold,
                    'Actually_Grouped': same_group,
                    'Aggregated_Feature_1': feature_to_group[i] if feature_to_group[i] is not None else '',
                    'Aggregated_Feature_2': feature_to_group[j] if feature_to_group[j] is not None else ''
                })

    correlations_df = pd.DataFrame(correlation_decisions)
    correlations_df = correlations_df.sort_values('Correlation', ascending=False)
    correlations_df.to_excel(writer, sheet_name='Feature Correlations', index=False)

def create_aggregation_sheet(df, feature_groups, aggregated_df, writer):
    for i, group in enumerate(feature_groups):
        if len(group) > 1:
            original_data = df[group]
            group_name = ", ".join(group)
            aggregated_data = aggregated_df[[group_name]]
            aggregated_data.columns = ['Aggregated_Mean']
            comparison_df = pd.concat([original_data, aggregated_data], axis=1)
            sheet_name = f'Aggregated Feature {i}'
            comparison_df.to_excel(writer, sheet_name=sheet_name, index=False)

def preprocess_features_with_variance_filter(df, writer=None, analysis_type=""):
    df_features = df.drop(['Object ID', 'Category'], axis=1)
    categories = df['Category']
    object_ids = df['Object ID']

    if writer is not None:
        df.to_excel(writer, sheet_name='1. Filter Categories', index=False)

    zero_check_columns = ['Velocity Mean', 'Velocity Median']
    available_zero_check_columns = [col for col in zero_check_columns if col in df_features.columns]

    if available_zero_check_columns:
        zero_mask = (df_features[available_zero_check_columns] == 0).any(axis=1)
        df_features = df_features[~zero_mask]
        categories = categories[~zero_mask]
        object_ids = object_ids[~zero_mask]

    if writer is not None:
        df_after_zero_filter = df_features.copy()
        df_after_zero_filter.insert(0, 'Object ID', object_ids)
        df_after_zero_filter.insert(1, 'Category', categories)
        df_after_zero_filter.to_excel(writer, sheet_name='2. Remove Non-Moving Objs.', index=False)

    log_data = np.sign(df_features) * np.log(np.abs(df_features) + 1)
    log_data = log_data.dropna()
    categories = categories.loc[log_data.index]
    object_ids = object_ids.loc[log_data.index]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(log_data)
    scaled_df = pd.DataFrame(scaled_data, columns=log_data.columns, index=log_data.index)

    if writer is not None:
        scaled_data_with_ids = scaled_df.copy()
        scaled_data_with_ids.insert(0, 'Object ID', object_ids)
        scaled_data_with_ids.insert(1, 'Category', categories)
        scaled_data_with_ids.to_excel(writer, sheet_name='3. Transform & Scale', index=False)

    scaled_df_filtered, removed_features = remove_near_zero_variance(scaled_df, variance_threshold)

    def format_feature_list(features, max_display=5):
        if len(features) <= max_display:
            return ', '.join(features)
        return f"{', '.join(features[:max_display])}... ({len(features)} total)"

    if removed_features:
        with thread_lock:
            messages.append(
                f"Removed {len(removed_features)} near-zero variance feature(s) from {analysis_type}: {format_feature_list(removed_features)}")

    feature_groups = detect_correlated_features(scaled_df_filtered, threshold=correlation_threshold)
    aggregated_features, feature_mapping = aggregate_correlated_features(scaled_df_filtered, feature_groups)

    if writer is not None:
        processed_data = aggregated_features.copy()
        processed_data.insert(0, 'Object ID', object_ids)
        processed_data.insert(1, 'Category', categories)
        processed_data.to_excel(writer, sheet_name='4. Aggregate Corr. Features', index=False)

        create_correlation_sheet(scaled_df_filtered, feature_groups, correlation_threshold, writer)
        create_aggregation_sheet(scaled_df_filtered, feature_groups, aggregated_features, writer)

    return aggregated_features, categories, feature_mapping, removed_features

def prepare_data_for_training(features, categories, use_kfold=False):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(categories)

    if use_kfold:
        return features, None, y_encoded, None, label_encoder, True
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            features, y_encoded, test_size=test_size, random_state=random_seed, stratify=y_encoded
        )
        return x_train, x_test, y_train, y_test, label_encoder, False

def prepare_data_for_analysis(df_selected, min_samples, analysis_type=""):
    class_counts = df_selected['Category'].value_counts()
    to_drop = class_counts[class_counts < min_samples].index.tolist()
    df_filtered = df_selected[~df_selected['Category'].isin(to_drop)]

    if df_filtered.empty:
        return None, None, None

    categories = df_filtered['Category']
    features = df_filtered.drop(['Object ID', 'Category'], axis=1)
    return df_filtered, categories, features

def adaptive_hyperparameter_grid(total_samples):
    if total_samples < 50:
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [2, 3, 4],
            'gamma': [0.0, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'min_child_weight': [1, 3],
            'reg_alpha': [0.0, 0.1],
            'reg_lambda': [0.0, 0.1]
        }
    else:
        return {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'max_depth': [3, 5, 7, 9],
            'gamma': [0.0, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [1, 3, 5],
            'reg_alpha': [0.0, 0.1, 0.5],
            'reg_lambda': [0.0, 0.1, 0.5]
        }

def optimize_hyperparameters_adaptive(x_train, y_train, use_kfold=False):
    total_samples = len(x_train)
    param_grid = adaptive_hyperparameter_grid(total_samples)

    model = xgb.XGBClassifier(
        objective='multi:softmax',
        eval_metric='mlogloss',
        num_class=len(np.unique(y_train)),
        n_jobs=-1
    )

    try:
        if use_kfold:
            min_class_size = np.bincount(y_train).min()
            cv_folds = min(5, min_class_size, total_samples // len(np.unique(y_train)))
            cv_folds = max(2, cv_folds)

            cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
        else:
            cv_strategy = 3

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=50 if total_samples < 50 else 100,
            cv=cv_strategy,
            scoring='accuracy',
            n_jobs=-1,
            random_state=random_seed,
        )

        random_search.fit(x_train, y_train)
        return random_search.best_params_

    except Exception:
        if total_samples < 50:
            return {
                'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3,
                'gamma': 0.0, 'subsample': 1.0, 'colsample_bytree': 1.0,
                'min_child_weight': 1, 'reg_alpha': 0.0, 'reg_lambda': 0.0
            }
        else:
            return {
                'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5,
                'gamma': 0.0, 'subsample': 1.0, 'colsample_bytree': 1.0,
                'min_child_weight': 1, 'reg_alpha': 0.0, 'reg_lambda': 0.0
            }

def train_and_evaluate(x_train, y_train, x_test, y_test, params):
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        eval_metric='mlogloss',
        num_class=len(np.unique(y_train)),
        n_jobs=-1,
        **params
    )
    eval_set = [(x_train, y_train), (x_test, y_test)]
    model.fit(x_train, y_train, eval_set=eval_set, verbose=False)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, model

def decode_classification_report_labels(report_df, label_encoder):
    label_mapping = dict(enumerate(label_encoder.classes_))

    summary_rows = ['accuracy', 'macro avg', 'weighted avg']
    filtered_report = report_df[~report_df.index.isin(summary_rows)]

    new_index = []
    for idx_val in filtered_report.index:
        try:
            encoded_label = int(idx_val)
            new_index.append(str(label_mapping.get(encoded_label, idx_val)))
        except ValueError:
            new_index.append(str(idx_val))

    cleaned_report = filtered_report.copy()
    cleaned_report.index = new_index

    return cleaned_report

def xgb_output(writer, results_data):
    if 'Full Dataset' in results_data:
        full_data = results_data['Full Dataset']
        full_features = full_data['features']
        full_report = full_data['report']

        features_only = full_features[full_features['Feature'] != ''].copy()
        features_only = features_only[~features_only['Feature'].isin(['Categories:', 'Method:'])].copy()
        features_only.to_excel(writer, sheet_name='Full Dataset Features', index=False)

        report_only = full_report[full_report.index != ''].copy()
        classification_rows = report_only[~report_only.index.isin(['Accuracy:', 'Method:'])].copy()

        classification_rows.reset_index(inplace=True)
        classification_rows.columns = ['Category', 'Precision', 'Recall', 'F1 Score', 'Support']

        accuracy_value = full_data['accuracy']
        method_value = full_data['method']

        blank_row = pd.DataFrame([['', '', '', '', '']], columns=['Category', 'Precision', 'Recall', 'F1 Score', 'Support'])
        accuracy_info = pd.DataFrame([
            ['Accuracy:', f'{accuracy_value:.3f}', '', '', ''],
            ['Method:', method_value, '', '', '']
        ], columns=['Category', 'Precision', 'Recall', 'F1 Score', 'Support'])

        full_report_final = pd.concat([classification_rows, blank_row, accuracy_info], ignore_index=True)
        full_report_final.to_excel(writer, sheet_name='Full Dataset Report', index=False)

    pairwise_features = []
    pairwise_reports = []

    for comparison, data in results_data.items():
        if comparison != 'Full Dataset' and 'vs' in comparison:
            cat1, cat2 = comparison.split(' vs ')

            features_df = data['features'].copy()
            features_df = features_df[features_df['Feature'] != ''].copy()
            features_df = features_df[~features_df['Feature'].isin(['Categories:', 'Method:'])].copy()
            features_df.insert(0, 'Comparator 1', cat1)
            features_df.insert(1, 'Comparator 2', cat2)
            pairwise_features.append(features_df)

            report_df = data['report'].copy()
            report_df = report_df[report_df.index != ''].copy()
            classification_rows_pair = report_df[~report_df.index.isin(['Accuracy:', 'Method:'])].copy()

            classification_rows_pair.reset_index(inplace=True)
            classification_rows_pair.columns = ['Category', 'Precision', 'Recall', 'F1 Score', 'Support']

            accuracy_value_pair = data['accuracy']
            method_value_pair = data['method']

            blank_row_pair = pd.DataFrame([['', '', '', '', '']], columns=['Category', 'Precision', 'Recall', 'F1 Score', 'Support'])
            accuracy_info_pair = pd.DataFrame([
                ['Accuracy:', f'{accuracy_value_pair:.3f}', '', '', ''],
                ['Method:', method_value_pair, '', '', '']
            ], columns=['Category', 'Precision', 'Recall', 'F1 Score', 'Support'])

            report_df_final = pd.concat([classification_rows_pair, blank_row_pair, accuracy_info_pair], ignore_index=True)
            report_df_final.insert(0, 'Comparator 1', cat1)
            report_df_final.insert(1, 'Comparator 2', cat2)
            pairwise_reports.append(report_df_final)

    if pairwise_features:
        all_pairwise_features = pd.concat(pairwise_features, ignore_index=True)
        all_pairwise_reports = pd.concat(pairwise_reports, ignore_index=True)

        all_pairwise_features.to_excel(writer, sheet_name='Pairwise Features', index=False)
        all_pairwise_reports.to_excel(writer, sheet_name='Pairwise Reports', index=False)

        workbook = writer.book
        border_format = workbook.add_format({'top': 1})

        if 'Pairwise Features' in writer.sheets:
            worksheet_features = writer.sheets['Pairwise Features']
            worksheet_features.conditional_format(1, 0, 1, len(all_pairwise_features.columns)-1,
                                                {'type': 'no_errors', 'format': border_format})

            current_comparison = None
            for idx, row in all_pairwise_features.iterrows():
                comparison_key = f"{row['Comparator 1']} vs {row['Comparator 2']}"
                if current_comparison is not None and current_comparison != comparison_key:
                    worksheet_features.conditional_format(idx+1, 0, idx+1, len(all_pairwise_features.columns)-1,
                                                        {'type': 'no_errors', 'format': border_format})
                current_comparison = comparison_key

        if 'Pairwise Reports' in writer.sheets:
            worksheet_reports = writer.sheets['Pairwise Reports']
            worksheet_reports.conditional_format(1, 0, 1, len(all_pairwise_reports.columns)-1,
                                               {'type': 'no_errors', 'format': border_format})

            current_comparison = None
            for idx, row in all_pairwise_reports.iterrows():
                comparison_key = f"{row['Comparator 1']} vs {row['Comparator 2']}"
                if current_comparison is not None and current_comparison != comparison_key:
                    worksheet_reports.conditional_format(idx+1, 0, idx+1, len(all_pairwise_reports.columns)-1,
                                                       {'type': 'no_errors', 'format': border_format})
                current_comparison = comparison_key

def xgboost(df_selected, output_file, features, categories, suffix=""):
    save_xgb = output_file + f'_XGB{suffix}.xlsx'
    results_data = {}

    category_counts = pd.Series(categories).value_counts()
    use_kfold = any(count < min_required_xgb for count in category_counts)

    x_train, x_test, y_train, y_test, label_encoder, is_kfold = prepare_data_for_training(
        features, categories, use_kfold=use_kfold)

    best_params = optimize_hyperparameters_adaptive(x_train, y_train, is_kfold)

    if not is_kfold:
        _, model = train_and_evaluate(x_train, y_train, x_test, y_test, best_params)
        y_pred = model.predict(x_test)
    else:
        model = xgb.XGBClassifier(
            objective='multi:softmax',
            eval_metric='mlogloss',
            num_class=len(np.unique(y_train)),
            n_jobs=-1,
            **best_params
        )
        model.fit(x_train, y_train)

        y_pred_full = model.predict(x_train)
        y_test = y_train
        y_pred = y_pred_full

    feature_importance = pd.DataFrame({
        'Feature': features.columns.tolist(),
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    blank_row = pd.DataFrame({'Feature': [''], 'Importance': ['']})
    comparison_info = pd.DataFrame({
        'Feature': ['Categories:', 'Method:'],
        'Importance': [f'Full Dataset', 'K-fold CV' if is_kfold else 'Train-Test Split']
    })

    feature_importance_with_info = pd.concat([
        feature_importance,
        blank_row,
        comparison_info
    ], ignore_index=True)

    if y_test is not None and y_pred is not None:
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        accuracy_value = report_df.loc['accuracy', 'precision']
        report_df = decode_classification_report_labels(report_df, label_encoder)

        blank_row_report = pd.DataFrame({
            'precision': [''], 'recall': [''], 'f1-score': [''], 'support': ['']
        }, index=[''])

        accuracy_info = pd.DataFrame({
            'precision': ['Accuracy:', 'Method:'],
            'recall': [f'{accuracy_value:.3f}', 'K-fold CV' if is_kfold else 'Train-Test Split'],
            'f1-score': ['', ''],
            'support': ['', '']
        }, index=['', ''])

        report_with_info = pd.concat([report_df, blank_row_report, accuracy_info])
    else:
        report_with_info = pd.DataFrame()
        accuracy_value = 0

    results_data['Full Dataset'] = {
        'features': feature_importance_with_info,
        'report': report_with_info,
        'accuracy': accuracy_value,
        'method': 'K-fold CV' if is_kfold else 'Train-Test Split'
    }

    aligned_df_selected = df_selected.loc[features.index]
    aligned_df_selected = aligned_df_selected.copy()
    aligned_df_selected['Category'] = categories.loc[features.index]

    unique_categories = aligned_df_selected['Category'].nunique()

    if unique_categories > 2:
        labels = aligned_df_selected['Category']
        unique_cats = aligned_df_selected['Category'].unique()
        category_pairs = list(itertools.combinations(unique_cats, 2))

        for comparison_num, (cat1, cat2) in enumerate(category_pairs, 1):
            try:
                pair_mask = labels.isin([cat1, cat2])
                pair_indices = labels[pair_mask].index

                pair_data = features.loc[pair_indices]
                pair_labels = labels.loc[pair_indices]

                if pair_data.empty or pair_labels.empty:
                    continue

                pair_category_counts = pd.Series(pair_labels).value_counts()
                pair_use_kfold = any(count < min_required_xgb for count in pair_category_counts)

                if pair_use_kfold:
                    label_encoder_pair = LabelEncoder()
                    y_encoded = label_encoder_pair.fit_transform(pair_labels)

                    best_params_pair = optimize_hyperparameters_adaptive(pair_data, y_encoded, pair_use_kfold)

                    model_pair = XGBClassifier(
                        objective='multi:softmax',
                        eval_metric='mlogloss',
                        num_class=len(np.unique(y_encoded)),
                        n_jobs=-1,
                        **best_params_pair
                    )

                    cv_folds = max(2, min(3, len(pair_labels) // 4))
                    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

                    y_true_all = []
                    y_pred_all = []

                    for train_idx, test_idx in cv_strategy.split(pair_data, y_encoded):
                        try:
                            x_train_cv, x_test_cv = pair_data.iloc[train_idx], pair_data.iloc[test_idx]
                            y_train_cv, y_test_cv = y_encoded[train_idx], y_encoded[test_idx]

                            model_pair.fit(x_train_cv, y_train_cv)
                            y_pred_cv = model_pair.predict(x_test_cv)

                            if len(y_test_cv) == len(y_pred_cv):
                                y_true_all.extend(y_test_cv)
                                y_pred_all.extend(y_pred_cv)
                        except Exception:
                            continue

                    model_pair.fit(pair_data, y_encoded)

                else:
                    x_train_pair, x_test_pair, y_train_pair, y_test_pair, label_encoder_pair, _ = prepare_data_for_training(
                        pair_data, pair_labels, use_kfold=pair_use_kfold)

                    best_params_pair = optimize_hyperparameters_adaptive(x_train_pair, y_train_pair, False)

                    model_pair = XGBClassifier(
                        objective='multi:softmax',
                        eval_metric='mlogloss',
                        num_class=len(np.unique(y_train_pair)),
                        n_jobs=-1,
                        **best_params_pair
                    )

                    model_pair.fit(x_train_pair, y_train_pair)

                    y_true_all = y_test_pair
                    y_pred_all = model_pair.predict(x_test_pair)

                feature_importance_pair = pd.DataFrame({
                    'Feature': pair_data.columns.tolist(),
                    'Importance': model_pair.feature_importances_
                }).sort_values(by='Importance', ascending=False)

                blank_row_pair = pd.DataFrame({'Feature': [''], 'Importance': ['']})
                comparison_info_pair = pd.DataFrame({
                    'Feature': ['Categories:', 'Method:'],
                    'Importance': [f'{cat1} vs {cat2}', 'K-fold CV' if pair_use_kfold else 'Train-Test Split']
                })

                feature_importance_with_info_pair = pd.concat([
                    feature_importance_pair,
                    blank_row_pair,
                    comparison_info_pair
                ], ignore_index=True)

                report_pair = classification_report(y_true_all, y_pred_all, output_dict=True, zero_division=0)
                report_df_pair = pd.DataFrame(report_pair).transpose()
                accuracy_value_pair = report_df_pair.loc['accuracy', 'precision']
                report_df_pair = decode_classification_report_labels(report_df_pair, label_encoder_pair)

                blank_row_report_pair = pd.DataFrame({
                    'precision': [''], 'recall': [''], 'f1-score': [''], 'support': ['']
                }, index=[''])

                comparison_info_report_pair = pd.DataFrame({
                    'precision': ['Accuracy:', 'Method:'],
                    'recall': [f'{accuracy_value_pair:.3f}', 'K-fold CV' if pair_use_kfold else 'Train-Test Split'],
                    'f1-score': ['', ''],
                    'support': ['', '']
                }, index=['', ''])

                report_with_info_pair = pd.concat([report_df_pair, blank_row_report_pair, comparison_info_report_pair])

                comparison_key = f"{cat1} vs {cat2}"
                results_data[comparison_key] = {
                    'features': feature_importance_with_info_pair,
                    'report': report_with_info_pair,
                    'accuracy': accuracy_value_pair,
                    'method': 'K-fold CV' if pair_use_kfold else 'Train-Test Split'
                }

            except Exception:
                continue

    with pd.ExcelWriter(save_xgb, engine='xlsxwriter') as writer:
        xgb_output(writer, results_data)

def pca(df_selected, df_processed, categories, savefile):
    if df_processed.empty or len(df_processed) < 4 or df_processed.shape[1] < 4:
        with thread_lock:
            messages.append("Not enough data for PCA, bypassing...")
            messages.append('')
        complete_progress_step("PCA")
        return None

    with thread_lock:
        messages.append('Starting PCA...')

    valid_idx = ~df_processed.isnull().any(axis=1)
    df_processed = df_processed[valid_idx]
    categories = categories.loc[df_processed.index]
    df_selected = df_selected.loc[df_processed.index]

    object_ids = df_selected['Object ID'].values

    pca_full = PCA()
    pca_full.fit(df_processed)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    n_components_95 = max(2, n_components_95)
    pca_model = PCA(n_components=n_components_95)
    pcscores = pca_model.fit_transform(df_processed)

    df_expl_var = pd.DataFrame(pca_model.explained_variance_ratio_)
    df_expl_var.columns = ["Explained variance ratio"]
    df_expl_var.index = [f'PC{i+1}' for i in range(n_components_95)]

    pc_score_data = {
        'Object ID': object_ids,
        'Category': categories.values,
    }
    for i in range(n_components_95):
        pc_score_data[f'PC{i+1}'] = pcscores[:, i]

    df_pcscores = pd.DataFrame(pc_score_data)

    df_features = pd.DataFrame(pca_model.components_)
    df_features.columns = df_processed.columns
    df_features.index = [f'PC{i+1}' for i in range(n_components_95)]
    df_features = df_features.T

    kruskal_result_list = []
    def kw_test(pc_kw):
        kruskal = stats.kruskal(
            *[group[pc_kw].values for name, group in df_pcscores.groupby('Category')],
            nan_policy='omit'
        )
        df_result = pd.DataFrame({kruskal})
        return df_result

    for pc_kw_no in range(1, n_components_95 + 1):
        pc_current = f"PC{pc_kw_no}"
        kruskal_result_list.append(kw_test(pc_current))

    df_kruskal = pd.concat(kruskal_result_list)
    df_kruskal.index = [f'PC{i+1}' for i in range(n_components_95)]

    posthoc_tests = {}
    for i in range(1, n_components_95 + 1):
        pc_name = f'PC{i}'
        posthoc_tests[pc_name] = sp.posthoc_dunn(
            df_pcscores, val_col=pc_name, group_col='Category', p_adjust='holm'
        )

    save_pca = savefile + '_PCA.xlsx'
    writer = pd.ExcelWriter(save_pca, engine='xlsxwriter')
    df_pcscores.to_excel(writer, sheet_name='PC scores', index=False)
    df_expl_var.to_excel(writer, sheet_name='PC explained variance', index=True)
    df_features.to_excel(writer, sheet_name='PC features', index=True)
    df_kruskal.to_excel(writer, sheet_name='Kruskal-Wallis', index=True)

    for pc_name, test_df in posthoc_tests.items():
        pd.DataFrame(test_df).to_excel(writer, sheet_name=f'{pc_name} tests', index=True)

    workbook = writer.book
    format_white = workbook.add_format({'bg_color': 'white'})
    format_yellow = workbook.add_format({'bg_color': 'yellow'})

    def highlight_objs(x, sheet_name):
        x.conditional_format('A1:ZZ100', {'type': 'blanks', 'format': format_white})
        if sheet_name == 'Kruskal-Wallis':
            x.conditional_format('C2:C100', {
                'type': 'cell', 'criteria': '<=', 'value': 0.05, 'format': format_yellow
            })
        else:
            x.conditional_format('B2:L12', {
                'type': 'cell', 'criteria': '<=', 'value': 0.05, 'format': format_yellow
            })

    sheets_to_highlight = ['Kruskal-Wallis'] + [f'PC{i+1} tests' for i in range(n_components_95)]
    for sheet in sheets_to_highlight:
        if sheet in writer.sheets:
            worksheet = writer.sheets[sheet]
            highlight_objs(worksheet, sheet)

    writer.close()
    with thread_lock:
        msg = f" PCA done with {n_components_95} components explaining {cumulative_variance[n_components_95-1]:.1%} of variance."
        messages[-1] += msg
    complete_progress_step("PCA")

    return df_pcscores

def ml_analysis(df_sum, parameters, savefile):
    with thread_lock:
        messages.append("Starting machine learning analysis...")
    tic = time.time()

    exclude_features = []
    if parameters['arrest_limit'] == 0 and 'Arrest Coefficient' in df_sum.columns:
        exclude_features.append('Arrest Coefficient')
    if 'Convex Hull Volume' in df_sum.columns and (
            'z_col_name' not in parameters or parameters['z_col_name'] is None or
            (df_sum['Convex Hull Volume'] == 0).all()):
        exclude_features.append('Convex Hull Volume')
    df_sum = df_sum.drop(columns=exclude_features, errors='ignore')
    df_selected = apply_category_filter(df_sum, parameters.get('pca_filter'))

    df_pca, categories_pca, features_pca = prepare_data_for_analysis(df_selected, min_required_pca, "PCA")
    df_xgb, categories_xgb, features_xgb = prepare_data_for_analysis(df_selected, min_required_xgb, "XGBoost")

    df_pcscores = None

    if df_pca is not None:
        if parameters.get('verbose', False):
            verbose_file_pca = f'{savefile}_PCA_Verbose.xlsx'
            with pd.ExcelWriter(verbose_file_pca, engine='xlsxwriter') as writer:
                df_processed_pca, categories_filtered_pca, feature_mapping_pca, removed_features_pca = (
                    preprocess_features_with_variance_filter(df_pca, writer, "PCA"))
        else:
            df_processed_pca, categories_filtered_pca, feature_mapping_pca, removed_features_pca = (
                preprocess_features_with_variance_filter(df_pca, None, "PCA"))

        if df_processed_pca is not None and not df_processed_pca.empty:
            df_pcscores = safe_ml_operation(pca, "PCA", df_pca, df_processed_pca, categories_filtered_pca, savefile)
        else:
            with thread_lock:
                messages.append("PCA preprocessing resulted in empty dataset")
            complete_progress_step("PCA")

    else:
        with thread_lock:
            messages.append(f"Insufficient data for PCA (requires at least {min_required_pca} samples per category)")
        complete_progress_step("PCA")

    with thread_lock:
        messages.append('Starting XGBoost...')

    if df_xgb is not None:
        try:
            if parameters.get('verbose', False):
                verbose_file_xgb = f'{savefile}_XGB-Features_Verbose.xlsx'
                with pd.ExcelWriter(verbose_file_xgb, engine='xlsxwriter') as writer:
                    df_processed_xgb, categories_filtered_xgb, feature_mapping_xgb, removed_features_xgb = (
                        preprocess_features_with_variance_filter(df_xgb, writer, "XGBoost"))
            else:
                df_processed_xgb, categories_filtered_xgb, feature_mapping_xgb, removed_features_xgb = (
                    preprocess_features_with_variance_filter(df_xgb, None, "XGBoost"))

            if df_processed_xgb is not None and not df_processed_xgb.empty:
                safe_ml_operation(xgboost, "XGB", df_xgb, savefile, df_processed_xgb, categories_filtered_xgb, "-Features")
            else:
                with thread_lock:
                    messages.append("XGBoost preprocessing resulted in empty dataset")
                complete_progress_step("XGB")
        except XGBAbortException:
            pass
        except Exception as e:
            with thread_lock:
                messages.append(f"Error in XGBoost analysis: {str(e)}")
            complete_progress_step("XGB")
    else:
        with thread_lock:
            messages.append(f"Insufficient data for XGBoost (requires at least {min_required_xgb} samples per category)")
        complete_progress_step("XGB")

    if df_pcscores is not None:
        df_pcscores_xgb, categories_pcscores, features_pcscores = prepare_data_for_analysis(df_pcscores, min_required_xgb, "XGBoost-PCscores")

        if df_pcscores_xgb is not None:
            if parameters.get('verbose', False):
                verbose_file_pcscores = f'{savefile}_XGB-PCscores_Verbose.xlsx'
                with pd.ExcelWriter(verbose_file_pcscores, engine='xlsxwriter') as writer:
                    df_processed_pcscores, categories_filtered_pcscores, feature_mapping_pcscores, removed_features_pcscores = (
                        preprocess_features_with_variance_filter(df_pcscores_xgb, writer, "XGBoost-PCscores"))
            else:
                df_processed_pcscores, categories_filtered_pcscores, feature_mapping_pcscores, removed_features_pcscores = (
                    preprocess_features_with_variance_filter(df_pcscores_xgb, None, "XGBoost-PCscores"))

            safe_ml_operation(xgboost, "XGB", df_pcscores_xgb, savefile, df_processed_pcscores, categories_filtered_pcscores, "-PCscores")

    with thread_lock:
        msg = " XGBoost done."
        messages[-1] += msg
    complete_progress_step("XGB")

    toc = time.time()
    with thread_lock:
        messages.append(f"Machine learning analysis done in {int(toc - tic)} seconds.")
        messages.append('')
    return df_pcscores

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import scikit_posthocs as sp
from scipy import stats
import xgboost.sklearn as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
import itertools
import time

from shared_state import messages, thread_lock, complete_progress_step

class XGBAbortException(Exception):
    pass

min_required = 5    # Minimum number of samples required for a category to be included in PCA and XGBoost
correlation_threshold = 0.95  # Pearson correlation threshold. Features with correlation above this threshold will be grouped together.

def safe_ml_operation(operation_func, error_step=None, *args, **kwargs):
    try:
        return operation_func(*args, **kwargs)
    except XGBAbortException:
        raise
    except Exception:
        if error_step:
            complete_progress_step(error_step)
        error()

def prepare_data_for_training(features, categories, test_size=0.4):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(categories)

    x_train, x_test, y_train, y_test = train_test_split(
        features, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )

    return x_train, x_test, y_train, y_test, label_encoder

def error():
    with thread_lock:
        messages.append('Error in XGBoost, bypassing...')
        messages.append('Usually these errors are due to too low N within one of the object categories. Use the category filter to exclude these low-N categories.')
        messages.append('')
    complete_progress_step("XGB")
    raise XGBAbortException()

def apply_category_filter(df, cat_filter):
    if cat_filter is None:
        return df
    df = df.copy()
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

def group_highly_correlated_features(df, threshold):
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

def preprocess_features(df):
    try:
        df_features = df.drop(['Object ID', 'Category'], axis=1)
        categories = df['Category']
        log_data = np.sign(df_features) * np.log(np.abs(df_features) + 1)
        log_data = log_data.dropna()
        categories = categories.loc[log_data.index]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(log_data)
        scaled_df = pd.DataFrame(scaled_data, columns=log_data.columns)
        feature_groups = group_highly_correlated_features(scaled_df, threshold=correlation_threshold)
        aggregated_features, feature_mapping = aggregate_correlated_features(scaled_df, feature_groups)
        return aggregated_features, categories, feature_mapping
    except XGBAbortException:
        raise
    except Exception:
        complete_progress_step("PCA")
        error()

def optimize_hyperparameters(x_train, y_train):
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
            estimator=model,
            param_distributions=param_grid,
            n_iter=100,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )
        random_search.fit(x_train, y_train)
        return random_search.best_params_
    except Exception:
        return {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'gamma': 0.0,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'min_child_weight': 1,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0
        }

def train_and_evaluate(x_train, y_train, x_test, y_test, params):
    try:
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
    except XGBAbortException:
        raise
    except Exception:
        error()

def decode_classification_report_labels(report_df, label_encoder):
    encoded_labels = label_encoder.transform(label_encoder.classes_)
    label_mapping = dict(zip(encoded_labels, label_encoder.classes_))

    summary_rows = ['accuracy', 'macro avg', 'weighted avg']
    filtered_report = report_df[~report_df.index.isin(summary_rows)]

    new_index = []
    for idx_val in filtered_report.index:
        try:
            encoded_label = int(str(idx_val))
            if encoded_label in label_mapping:
                new_index.append(str(label_mapping[encoded_label]))
            else:
                new_index.append(str(idx_val))
        except ValueError:
            new_index.append(str(idx_val))

    cleaned_report = filtered_report.copy()
    cleaned_report.index = new_index

    return cleaned_report

def xgboost_pairwise(data, category_col, aggregated_features, writer):
    if category_col not in data.columns:
        raise KeyError(f"'{category_col}' column is missing from the DataFrame.")

    labels = data[category_col]
    unique_categories = data[category_col].unique()
    category_pairs = list(itertools.combinations(unique_categories, 2))

    for comparison_num, (cat1, cat2) in enumerate(category_pairs, 1):
        try:
            pair_indices = labels.isin([cat1, cat2])
            pair_data = aggregated_features[pair_indices]
            pair_labels = labels[pair_indices]

            if pair_data.empty or pair_labels.empty:
                continue

            x_train, x_test, y_train, y_test, label_encoder = prepare_data_for_training(
                pair_data, pair_labels, test_size=0.4
            )

            model = XGBClassifier(eval_metric='logloss')
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            feature_importance = pd.DataFrame({
                'Feature': pair_data.columns,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)

            blank_row = pd.DataFrame({'Feature': [''], 'Importance': ['']})
            comparison_info = pd.DataFrame({
                'Feature': ['Categories:'],
                'Importance': [f'{cat1} vs {cat2}']
            })

            feature_importance_with_info = pd.concat([
                feature_importance,
                blank_row,
                comparison_info
            ], ignore_index=True)

            sheet_name_features = f'Pair_{comparison_num}_Features'
            sheet_name_report = f'Pair_{comparison_num}_Report'

            feature_importance_with_info.to_excel(writer, sheet_name=sheet_name_features, index=False)

            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            accuracy_value = report_df.loc['accuracy', 'precision']
            report_df = decode_classification_report_labels(report_df, label_encoder)

            blank_row = pd.DataFrame({
                'precision': [''],
                'recall': [''],
                'f1-score': [''],
                'support': ['']
            }, index=[''])

            comparison_info = pd.DataFrame({
                'precision': ['Accuracy:'],
                'recall': [f'{accuracy_value:.3f}'],
                'f1-score': [''],
                'support': ['']
            }, index=[''])

            report_with_info = pd.concat([report_df, blank_row, comparison_info])
            report_with_info.to_excel(writer, sheet_name=sheet_name_report)

        except Exception:
            continue

def xgboost(df_selected, output_file, features, categories, feature_mapping):
    with thread_lock:
        messages.append('Starting XGBoost...')
    try:
        save_xgb = output_file + '_XGB.xlsx'
        category_col = 'Category'
        with pd.ExcelWriter(save_xgb, engine='xlsxwriter') as writer:
            x_train, x_test, y_train, y_test, label_encoder = prepare_data_for_training(features, categories)
            best_params = optimize_hyperparameters(x_train, y_train)
            _, model = train_and_evaluate(x_train, y_train, x_test, y_test, best_params)
            y_pred = model.predict(x_test)
            feature_importance = pd.DataFrame({
                'Features': [",".join(map(str, feature_mapping[col])) for col in features.columns],
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            feature_importance.to_excel(writer, sheet_name='Full Dataset Features', index=False)
            if y_test is not None and y_pred is not None:
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose()
                accuracy_value = report_df.loc['accuracy', 'precision']
                report_df = decode_classification_report_labels(report_df, label_encoder)

                blank_row = pd.DataFrame({
                    'precision': [''],
                    'recall': [''],
                    'f1-score': [''],
                    'support': ['']
                }, index=[''])

                accuracy_info = pd.DataFrame({
                    'precision': ['Accuracy:'],
                    'recall': [f'{accuracy_value:.3f}'],
                    'f1-score': [''],
                    'support': ['']
                }, index=[''])

                report_with_info = pd.concat([report_df, blank_row, accuracy_info])
                report_with_info.to_excel(writer, sheet_name='Full Dataset Report')
            xgboost_pairwise(
                data=df_selected,
                category_col=category_col,
                aggregated_features=features,
                writer=writer
            )
        with thread_lock:
            msg = " XGBoost done."
            messages[-1] += msg
        complete_progress_step("XGB")
    except XGBAbortException:
        raise
    except Exception:
        error()

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
    categories = categories[valid_idx]
    df_selected = df_selected[valid_idx]

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
            df_pcscores, val_col=pc_name, group_col='Category', p_adjust='bonferroni'
        )

    save_pca = savefile + '_PCA.xlsx'
    writer = pd.ExcelWriter(save_pca, engine='xlsxwriter')
    df_selected.to_excel(writer, sheet_name='Full dataset', index=False)
    df_processed.to_excel(writer, sheet_name='PCA dataset', index=False)
    df_expl_var.to_excel(writer, sheet_name='PC explained variance', index=True)
    df_pcscores.to_excel(writer, sheet_name='PC scores', index=False)
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

    class_counts = df_selected['Category'].value_counts()
    to_drop = class_counts[class_counts < min_required].index.tolist()
    if to_drop:
        for group in to_drop:
            with thread_lock:
                messages.append(f"Excluded category '{group}'. Only {class_counts[group]} objects (requires at least {min_required}).")
        df_selected = df_selected[~df_selected['Category'].isin(to_drop)]
        if df_selected.empty:
            complete_progress_step("PCA")
            error()

    df_selected = df_selected.dropna().drop(labels=['Duration', 'Path Length'], axis=1)
    df_processed, categories_filtered, feature_mapping = preprocess_features(df_selected)
    df_processed = df_processed.dropna()

    common_indices = categories_filtered.index.intersection(df_processed.index)
    df_selected = df_selected.loc[common_indices]
    df_processed = df_processed.loc[common_indices]
    categories_filtered = categories_filtered.loc[common_indices]

    remaining_categories = np.unique(categories_filtered)
    if len(remaining_categories) <= 1:
        with thread_lock:
            messages.append("No categories remaining after data cleaning, skipping PCA and XGBoost...")
            messages.append('')
        complete_progress_step("PCA")
        complete_progress_step("XGBoost")
        return None
    else:
        df_pcscores = pca(df_selected, df_processed, categories_filtered, savefile)
        xgboost(df_selected, savefile, df_processed, categories_filtered, feature_mapping)

        toc = time.time()
        with thread_lock:
            messages.append(f"Machine learning analysis done in {int(toc - tic)} seconds.")
            messages.append('')
        return df_pcscores
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scikit_posthocs as sp
from scipy import stats

from shared_state import messages, thread_lock, complete_progress_step


def apply_category_filter(df, filter):
    with thread_lock:
        messages.append(f"Unique Categories: {df['Category'].unique()}")
    if filter is None:
        return df
    if pd.api.types.is_numeric_dtype(df['Category']):
        try:
            filter_vals = [int(x) for x in filter]
        except ValueError:
            filter_vals = filter
            with thread_lock:
                messages.append("Error converting filter values to int. Using original filter values.")
        with thread_lock:
            messages.append(f"Filtering categories to {filter_vals}...")
        df['Category'] = df['Category'].astype(int)
    else:
        filter_vals = [str(x) for x in filter]
        with thread_lock:
            messages.append(f"Filtering categories to {filter_vals}...")
    filtered_df = df[df['Category'].isin(filter_vals)]
    if filtered_df.empty:
        with thread_lock:
            messages.append('No data available for the selected categories.')
    return filtered_df

def pca(df, parameters, savefile):
    filter = parameters.get('pca_filter')
    df = apply_category_filter(df, filter)

    with thread_lock:
        messages.append('Starting PCA...')

    df = df.dropna()
    df_pca = df.drop(
        labels=['Object ID', 'Duration', 'Path Length', 'Final Euclidean', 'Straightness', 'Velocity filtered Mean',
                'Velocity Mean', 'Velocity Median', 'Acceleration Filtered Mean', 'Acceleration Mean',
                'Absolute Acceleration Mean', 'Absolute Acceleration Median', 'Acceleration Filtered Mean',
                'Acceleration Filtered Median', 'Acceleration Filtered Standard Deviation',
                'Acceleration Median', 'Overall Euclidean Median', 'Convex Hull Volume', 'Category'], axis=1)
    df_pca.columns = df_pca.columns.str.strip()
    df_pca = df_pca.dropna()
    x = np.array(df_pca)
    x = StandardScaler().fit_transform(x)
    pca_model = PCA(n_components=4)
    PCscores = pca_model.fit_transform(x)
    df_expl_var = pd.DataFrame(pca_model.explained_variance_ratio_)
    df_expl_var.columns = ["Explained variance ratio"]
    df_expl_var.index = ['PC1', 'PC2', 'PC3', 'PC4']
    df_PCscores = pd.DataFrame(PCscores)
    df_PCscores.columns = ['PC1', 'PC2', 'PC3', 'PC4']
    df_PCscores["Object ID"] = df['Object ID'].values
    df_PCscores["Category"] = df['Category'].values
    df_features = pd.DataFrame(pca_model.components_)
    df_features.columns = df_pca.columns
    df_features.index = ['PC1', 'PC2', 'PC3', 'PC4']

    kruskal_result_list = []

    def kw_test(PC_kw):
        kruskal = stats.kruskal(
            *[group[PC_kw].values for name, group in df_PCscores.groupby('Category')],
            nan_policy='omit'
        )
        df_result = pd.DataFrame({kruskal})
        return df_result

    for PC_kw_no in range(1, 5):
        PC_current = f"PC{PC_kw_no}"
        kruskal_result_list.append(kw_test(PC_current))
    df_kruskal = pd.concat(kruskal_result_list)
    df_kruskal.index = ['PC1', 'PC2', 'PC3', 'PC4']

    PC1_test = sp.posthoc_dunn(df_PCscores, val_col='PC1', group_col='Category', p_adjust='bonferroni')
    PC2_test = sp.posthoc_dunn(df_PCscores, val_col='PC2', group_col='Category', p_adjust='bonferroni')
    PC3_test = sp.posthoc_dunn(df_PCscores, val_col='PC3', group_col='Category', p_adjust='bonferroni')
    PC4_test = sp.posthoc_dunn(df_PCscores, val_col='PC4', group_col='Category', p_adjust='bonferroni')

    df_PC1 = pd.DataFrame(PC1_test)
    df_PC2 = pd.DataFrame(PC2_test)
    df_PC3 = pd.DataFrame(PC3_test)
    df_PC4 = pd.DataFrame(PC4_test)

    savePCA = savefile + '_PCA.xlsx'
    with thread_lock:
        messages.append("Saving PCA output to " + savePCA + "...")
    writer = pd.ExcelWriter(savePCA, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Full dataset', index=False)
    df_pca.to_excel(writer, sheet_name='PCA dataset', index=False)
    df_expl_var.to_excel(writer, sheet_name='PC explained variance', index=True)
    df_PCscores.to_excel(writer, sheet_name='PC scores', index=False)
    df_features.to_excel(writer, sheet_name='PC features', index=True)
    df_kruskal.to_excel(writer, sheet_name='Kruskal-Wallis', index=True)
    df_PC1.to_excel(writer, sheet_name='PC1 tests', index=True)
    df_PC2.to_excel(writer, sheet_name='PC2 tests', index=True)
    df_PC3.to_excel(writer, sheet_name='PC3 tests', index=True)
    df_PC4.to_excel(writer, sheet_name='PC4 tests', index=True)

    workbook = writer.book
    format_white = workbook.add_format({'bg_color': 'white'})
    format_yellow = workbook.add_format({'bg_color': 'yellow'})

    def highlight_objs(worksheet):
        worksheet.conditional_format('A1:ZZ100', {'type': 'blanks', 'format': format_white})
        worksheet.conditional_format('B2:L12', {
            'type': 'cell',
            'criteria': '<=',
            'value': 0.05,
            'format': format_yellow
        })

    sheets = ['Kruskal-Wallis', 'PC1 tests', 'PC2 tests', 'PC3 tests', 'PC4 tests']
    for sheet in sheets:
        worksheet = writer.sheets[sheet]
        highlight_objs(worksheet)

    writer.close()
    with thread_lock:
        messages.append('...PCA done.')
        messages.append('')
    complete_progress_step("PCA")

    return df_PCscores
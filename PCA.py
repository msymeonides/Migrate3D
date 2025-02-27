import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scikit_posthocs as sp
from scipy import stats


def pca(df, parameters, savefile):
    # Filter PCA if specific categories are given
    filter_ = parameters['pca_filter']
    if filter_ is not None:
        filter_ = filter_.split(sep=',')
        filter_ = [int(x) for x in filter_]
        print(f'Filtering categories for PCA to {filter_}...')
        df = df[df['Category'].isin(filter_)]
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
    pca = PCA(n_components=4)
    PCscores = pca.fit_transform(x)
    df_expl_var = pd.DataFrame(pca.explained_variance_ratio_)
    df_expl_var.columns = ["Explained variance ratio"]
    df_expl_var.index = ['PC1', 'PC2', 'PC3', 'PC4']
    df_PCscores = pd.DataFrame(PCscores)
    df_PCscores.columns = ['PC1', 'PC2', 'PC3', 'PC4']
    df_PCscores['Object ID'] = df['Object ID'].values
    df_PCscores['Category'] = df['Category'].values
    df_features = pd.DataFrame(pca.components_)
    df_features.columns = df_pca.columns
    df_features.index = ['PC1', 'PC2', 'PC3', 'PC4']
    kruskal_result_list = []

    def kw_test(PC_kw):
        kruskal = stats.kruskal(*[group[PC_kw].values for name, group in df_PCscores.groupby('Category')],
                                nan_policy='omit')
        df_result = pd.DataFrame({kruskal})
        return df_result

    for PC_kw_no in range(1, 5):
        PC_current = "{}{}".format('PC', PC_kw_no)
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
    print('Saving PCA output to ' + savePCA + '...')
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
        worksheet.conditional_format('A1:ZZ100', {'type': 'blanks',
                                                  'format': format_white})
        worksheet.conditional_format('B2:L12', {'type': 'object',
                                                'criteria': '<=',
                                                'value': 0.05,
                                                'format': format_yellow})

    sheets = ['Kruskal-Wallis', 'PC1 tests', 'PC2 tests', 'PC3 tests', 'PC4 tests']

    for i in sheets:
        worksheet = writer.sheets[i]
        highlight_objs(worksheet)

    writer.close()
    print('...PCA done.')
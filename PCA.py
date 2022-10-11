import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scikit_posthocs as sp
from scipy import stats

"""parameters = {'Interval': 15, 'arrest_displacement': 3.0, 'contact_length': None, 'arrested': 0.95, 'moving': 4,
              'timelapse': 4, 'savefile': 'Migrate3D_Results.xlsx', 'parent_id': 'Parent ID', 'time_col': "Time",
              'x_for': 'X Coordinate', 'y_for': 'Y Coordinate', 'z_for': 'Z Coordinate', 'parent_id2': 'Id',
              'category_col': 'Code', 'Contact': False, 'Tau_val': 6, 'infile_tracks': False}

df = pd.DataFrame(pd.read_excel('Migrate3D_Results_with_PCA.xlsx', sheet_name='Summary Statistics'))"""


def pca(df, parameters):
    df_pca = df.drop(
        labels=['Cell ID', 'Duration', 'Path Length', 'Final Euclidean', 'Straightness', 'Velocity filtered Mean',
                'Velocity Mean',
                'Velocity Median', 'Acceleration Filtered Mean', 'Acceleration Mean', 'Acceleration Median',
                'Overall Angle Median',
                'Overall Euclidean Median', 'Convex Hull Volume'], axis=1)

    df_pca.columns = df_pca.columns.str.strip()

    print("PCA dataset: {}".format(df_pca.shape))
    df_pca = df_pca.dropna()
    print("PCA dataset: {}".format(df_pca.shape))
    x = np.array(df_pca)
    print(x)
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=4)
    PCscores = pca.fit_transform(x)
    df_expl_var = pd.DataFrame(pca.explained_variance_ratio_)
    df_expl_var.columns = ["Explained variance ratio"]
    df_expl_var.index = ['PC1', 'PC2', 'PC3', 'PC4']
    df_PCscores = pd.DataFrame(PCscores)
    df_PCscores.columns = ['PC1', 'PC2', 'PC3', 'PC4']
    df_PCscores['Cell ID'] = df['Cell ID'].values
    df_PCscores['Category'] = df['Cell Type'].values
    df_features = pd.DataFrame(pca.components_)
    df_features.columns = df_pca.columns
    df_features.index = ['PC1', 'PC2', 'PC3', 'PC4']

    # Kruskal-Wallis test
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

    writer = pd.ExcelWriter('PCA_' + str(parameters['savefile']), engine='xlsxwriter')
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

    def highlight_cells(worksheet):
        worksheet.conditional_format('A1:ZZ100', {'type': 'blanks',
                                                  'format': format_white})
        worksheet.conditional_format('B2:L12', {'type': 'cell',
                                                'criteria': '<=',
                                                # 'criteria': '=AND((NOT(ISBLANK(A1)),(A1<=0.05))',
                                                'value': 0.05,
                                                'format': format_yellow})

    sheets = ['Kruskal-Wallis', 'PC1 tests', 'PC2 tests', 'PC3 tests', 'PC4 tests']

    for i in sheets:
        worksheet = writer.sheets[i]
        highlight_cells(worksheet)

    writer.save()
    print('pca Done')


#pca(df, parameters)



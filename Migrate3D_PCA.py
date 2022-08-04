import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scikit_posthocs as sp
from scipy import stats

filename = 'Migrate3D_Results_2022-08-03.xlsx'   # Name of .xlsx Migrate3D results file (input).
#categories = [4, 5, 8]       # Change the numbers in square brackets to select Categories of interest,
categories = range(2, 9)      # or uncomment this to include all categories.
#outfile = 'Migrate3D_Results_2022-08-03_PCA-458-only.xlsx'        # Name of .xlsx PCA results file (output).
outfile = 'Migrate3D_Results_2022-08-03_PCA-2to9-only.xlsx'        # Name of .xlsx PCA results file (output).

# Prepare dataset
print("Loading dataset...")
def data(filename):
    global df
    infile = pd.read_excel(filename, sheet_name='Summary Statistics')
    df = pd.DataFrame(infile)
    df = df.dropna()
    return df

df_building = pd.DataFrame()

def subset(cats_of_interest):
    global df
    global df_building
    df_cats = df[df['Category'] == cats_of_interest]
    df_building = pd.concat([df_building, df_cats])
    df_building = df_building.sort_values(by=('Category'), ascending=True)
    return df_building

data(filename)

for i in categories:
    subset(i)

print("Full dataset: {}".format(df.shape))
df = df_building
print("Subsetted dataset: {}".format(df.shape))
df_pca = df.drop(labels=['Cell ID', 'Duration', 'Path Length', 'Final Euclidean', 'Straightness', 'Velocity filtered Mean', 'Velocity Mean', 'Velocity Median',
                     'Acceleration Filtered Mean', 'Acceleration Mean', 'Acceleration Median', 'Overall Euclidean Median', 'Category'], axis=1)
print("PCA dataset: {}".format(df_pca.shape))


# PCA
print("Running PCA...")
x = np.array(df_pca)
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=4)
PCscores = pca.fit_transform(x)
df_expl_var = pd.DataFrame(pca.explained_variance_ratio_)
df_expl_var.columns = ["Explained variance ratio"]
df_expl_var.index = ['PC1', 'PC2', 'PC3', 'PC4']
df_PCscores = pd.DataFrame(PCscores)
df_PCscores.columns = ['PC1', 'PC2', 'PC3', 'PC4']
df_PCscores['Cell ID'] = df['Cell ID'].values
df_PCscores['Category'] = df['Category'].values
df_features = pd.DataFrame(pca.components_)
df_features.columns = df_pca.columns
df_features.index = ['PC1', 'PC2', 'PC3', 'PC4']

# Kruskal-Wallis test
df_kruskal = pd.DataFrame()

def kw_test(PC_kw):
    global df_kruskal
    kruskal = stats.kruskal(*[group[PC_kw].values for name, group in df_PCscores.groupby('Category')], nan_policy='omit')
    df_result = pd.DataFrame({kruskal})
    df_kruskal = pd.concat([df_kruskal, df_result])

for PC_kw_no in range(1,5):
    PC_current = "{}{}".format('PC', PC_kw_no)
    kw_test(PC_current)

df_kruskal.index = ['PC1', 'PC2', 'PC3', 'PC4']


# Dunn post-hoc tests with Bonferroni p-value correction
print("Running posthoc tests...")
PC1_test = sp.posthoc_dunn(df_PCscores, val_col='PC1', group_col='Category', p_adjust='bonferroni')
PC2_test = sp.posthoc_dunn(df_PCscores, val_col='PC2', group_col='Category', p_adjust='bonferroni')
PC3_test = sp.posthoc_dunn(df_PCscores, val_col='PC3', group_col='Category', p_adjust='bonferroni')
PC4_test = sp.posthoc_dunn(df_PCscores, val_col='PC4', group_col='Category', p_adjust='bonferroni')
df_PC1 = pd.DataFrame(PC1_test)
df_PC2 = pd.DataFrame(PC2_test)
df_PC3 = pd.DataFrame(PC3_test)
df_PC4 = pd.DataFrame(PC4_test)


# Write all to Excel
print("Writing dataset and results to Excel file...")
writer = pd.ExcelWriter(outfile, engine='xlsxwriter')
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


# Highlight significant p-values
workbook = writer.book
format_white = workbook.add_format({'bg_color': 'white'})
format_yellow = workbook.add_format({'bg_color': 'yellow'})

def highlight_cells(worksheet):
    worksheet.conditional_format('A1:ZZ100', {'type': 'blanks',
                                    'format': format_white})
    worksheet.conditional_format('B2:L12', {'type': 'cell',
                                    'criteria': '<=',
                                        #'criteria': '=AND((NOT(ISBLANK(A1)),(A1<=0.05))',
                                    'value': 0.05,
                                    'format': format_yellow})

sheets = ['Kruskal-Wallis', 'PC1 tests', 'PC2 tests', 'PC3 tests', 'PC4 tests']

for i in sheets:
    worksheet = writer.sheets[i]
    highlight_cells(worksheet)

writer.save()


# Prepare data for plots
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }
df['Category'] = df['Category'].astype(str)

# PC scatter plots
print("Plotting PCxPC scatter plots...")
fig_matrix = px.scatter_matrix(
    PCscores,
    labels=labels,
    dimensions=range(4),
    opacity=0.7,
    color=df['Category'],
    color_discrete_sequence=['#00FFFF', '#00FF00', '#FF00FF'],
    #color_discrete_sequence=['#808080', '#808080', '#00FFFF', '#00FF00', '#808080', '#808080', '#FF00FF'],
    hover_name=df['Cell ID']
)
fig_matrix.update_traces(diagonal_visible=False, opacity=0.7, marker=dict(size=6, line=dict(width=0.5, color='black')))
fig_matrix.show()


# Violin plots
print("Plotting PCs by Category...")
df_PCscores['Category'] = df_PCscores["Category"].astype(str)

def violin(PC):
    plot = px.violin(df_PCscores, y=df_PCscores[PC], x=df['Category'],
                    box=True,
                    points="all",
                    violinmode='overlay',
                    color=df['Category'],
                    color_discrete_sequence=['#00FFFF', '#00FF00', '#FF00FF'],
                    #color_discrete_sequence=['#808080', '#808080', '#00FFFF', '#00FF00', '#808080', '#808080', '#FF00FF'],
                    title=PC,
                    hover_name=df_PCscores['Cell ID']
                    )
    return plot

for PC_vio_no in range(1,5):
    PC_current = "{}{}".format('PC', PC_vio_no)
    print(PC_current)
    plot_with_passed = violin(PC_current)
    plot_with_passed.update_traces(opacity=0.7, marker=dict(size=6, line=dict(width=0.5, color='black')))
    plot_with_passed.show()


# 3D scatter plot
print("Plotting 3D scatter plot for PC1-3...")
fig_3d = px.scatter_3d(df_PCscores, x=df_PCscores["PC1"], y=df_PCscores["PC2"], z=df_PCscores["PC3"],
                        hover_name=df_PCscores['Cell ID'],
                        opacity=0.7,
                        color=df['Category'],
                        color_discrete_sequence=['#00FFFF', '#00FF00', '#FF00FF'],
                        #color_discrete_sequence=['#808080', '#808080', '#00FFFF', '#00FF00', '#808080', '#808080', '#FF00FF']
                        #color_discrete_sequence = px.colors.qualitative.Plotly
                        )
fig_3d.update_traces(marker=dict(size=5, line=dict(width=0.5, color='black')))
fig_3d.show()

import pandas as pd
import numpy as np
from patsy import dmatrix
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import MultipleLocator

def remove_outliers(df):
    for col in df.columns[1:]:
        mean = df[col].mean()
        std = df[col].std()
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        df[col] = df[col].apply(lambda x: np.nan if x < lower_bound or x > upper_bound else x)
    return df

## read raw_data
df_raw = pd.read_excel('2.1.DataPreprocessed.xlsx')
with open('1.sel.id.txt', 'r') as file:
    ids = [line.strip().upper() for line in file]
df_raw = df_raw[['Metabolite.ID'] + ids]
df_raw.set_index('Metabolite.ID', inplace=True)
df_raw = df_raw.transpose()
df_raw.reset_index(drop=False, inplace=True)
df_raw.rename(columns={'index': 'ID'}, inplace=True)

## step 1. QC
numeric_columns = df_raw.columns.difference(['ID'])
df_raw[numeric_columns] = df_raw[numeric_columns].apply(np.log) # log-transformed
df_cleaned = remove_outliers(df_raw) # remove 3 standard deviations
df_cleaned_nor[numeric_columns] = df_cleaned[numeric_columns].apply(zscore, nan_policy='omit') # standardized
df_cleaned_nor.to_csv('2.1.trans.qc.csv', index=None)

## step 2. corrected batch, parity, and sampling time effects using the linear model
fix = {}
with open('One.dim.txt') as f:
    lines = f.readlines()
    for ln in lines:
        ln = ln.strip().split()
        fix[ln[0]] = ln[1], ln[2]
bat={}
with open('batch.txt') as f:
    lines = f.readlines()
    for ln in lines:
        ln = ln.strip().split()
        bat[ln[0]] = ln[1]

df_cleaned_nor['batch'] = df_cleaned_nor['ID'].apply(lambda x: bat[x])
df_cleaned_nor['dim'] = df_cleaned_nor['ID'].apply(lambda x: fix[x][1])
df_cleaned_nor['parity'] = df_cleaned_nor['ID'].apply(lambda x: fix[x][0])
df_lm = df_cleaned_nor.copy()
for mb in numeric_columns:
    df_tr = df_cleaned_nor[df_cleaned_nor[mb] > -99]
    dmat = dmatrix('batch+parity+dim', df_tr)
    x = np.array(dmat)[:, 1:]
    y = np.array(df_tr[mb])
    model = LinearRegression()
    model.fit(x, y)
    b = model.coef_
    u = model.intercept_
    resi = y - u - np.dot(x, b)
    corr = np.corrcoef(resi, y)
    val_lm = np.array(df_cleaned_nor[mb])
    val_lm[val_lm > -99] = resi
    df_lm[mb] = val_lm

df_lm.to_csv("2.1.trans.qc.resi.csv", index = None)

## step 3. visualization
# distribution of the metabolite values
df_info = pd.read_csv('Data.info.csv')
class_counts = df_info['Final.Class'].value_counts()
top_10_class_counts = class_counts.head(10)
df_info_top10 = df_info[df_info['Final.Class'].apply(lambda x: x in top_10_class_counts.index)]
df_info_top10_re = df_info_top10.drop_duplicates(subset=['Final.Class'], keep='last')

df_cleaned_nor = pd.read_csv('2.1.trans.qc.resi.csv')
df_cleaned_nor_pl = df_cleaned_nor[df_info_top10_re['Metabolite.ID'].values]
plt.figure(figsize=(5, 2.5), dpi=300)
plt.rc('font', family='Arial', size=8)
for col in df_cleaned_nor_pl.columns:
    sns.histplot(df_cleaned_nor_pl[col], kde=True, bins=50, label=df_info_top10_re[df_info_top10_re['Metabolite.ID'] == col].iloc[0,16])
plt.xlabel('Metabolomic value')
plt.ylabel('Frequency')
plt.xlim([-4.8, 4.8])
plt.legend(loc=1, frameon=False, fontsize=5)
ax = plt.gca()
y_major_locator = MultipleLocator(5)
#ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.savefig('qc.resi.destri.png', bbox_inches='tight', pad_inches=0.1)

# violinplot of the metabolite values
plt.clf()
plot_pla = {'0.783_197.08977':'N-acetyl-L-aspartate', '0.738_133.03764':'L-aspartate', '0.746_132.05353':'L-asparagine', 
                '0.756_155.01940':'D-aspartate','0.742_147.05322':'L-glutamate'}
plot_ser = {'B100282':'L-arginine'}
df_pla = pd.read_csv('farmA.plasma.raw_data.csv')
df_ser = pd.read_csv('farmB.serum.raw_data.csv')
df_pla = df_pla[list(plot_pla.keys())].melt(var_name='Feature', value_name='Value')
df_ser = df_ser[list(plot_ser.keys())].melt(var_name='Feature', value_name='Value')
df_pla['Feature'] = df_pla['Feature'].map(plot_pla)
df_ser['Feature'] = df_ser['Feature'].map(plot_ser)
df_pla['Value'] /= 10000
df_ser['Value'] /= 10000

fig = plt.figure(figsize=(11, 3), dpi=300)
plt.rc('font', family='Arial', size=8)

# Plot for plasma
ax1 = plt.subplot2grid((1, 10), (0, 0), colspan=7, rowspan=1)
sns.violinplot(x='Feature', y='Value', data=df_pla, ax=ax1)
ax1.set_xlabel('')
ax1.set_ylabel('Metabolomic value (×10,000)')
ax1.set_title('Plasma')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

# Plot for serum
ax2 = plt.subplot2grid((1, 10), (0, 8), colspan=2, rowspan=1)
sns.violinplot(x='Feature', y='Value', data=df_ser, ax=ax2)
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.set_title('Serum')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

fig.text(0.45, -0.19, 'Metabolite', ha='center', va='center', fontsize=8)
plt.savefig('violinplot.png', bbox_inches='tight', pad_inches=0.1)

# pie plot
plt.clf()
df_pie = pd.read_csv('class_counts.csv')
labels = df_pie['Class']
pie_colors = ["#1f78b4","#33a02c","#e31a1c","#ff7f00",
                          "#6a3d9a","#b15928","#a6cee3","#b2df8a",
                          "#fb9a99","#fdbf6f","#cab2d6"]
plt.figure(figsize=(5, 2.5), dpi=300)
plt.rc('font', family='Arial', size=8)
fig, ax = plt.subplots()
ax.pie(_pie['Count'], labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, wedgeprops=dict(width=0.4), colors=pie_colors)
plt.savefig('pie.png', bbox_inches='tight', pad_inches=0.1)
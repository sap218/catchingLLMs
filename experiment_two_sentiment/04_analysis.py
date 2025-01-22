# -*- coding: utf-8 -*-
"""
@date: 2024-10-24
@author: sap218
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import re
from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef
import scipy.stats as stats

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np

#####################

of_interest = "sentiment"

df = pd.read_csv("../spotify_user_reviews/exp2/DATASET_randomised_annotated_catch_llm.csv")
del df["review"]
df.rename(columns={'sentiment':'manual'}, inplace=True)

df["manual"] = df["manual"].replace({'0':np.nan, 'p':1, 'n':-1}, regex=True)
df = df.replace({-10:np.nan,}, regex=True)

df.to_csv("../results/exp2/annotations.tsv", index=False, header=True, sep="\t")

#####################

methods = ["catch","gemma2","llama","mixtral"]

#####################

def replace_values(x):
    if x > 0.05: return "positive"
    elif x < -0.05: return "negative"
    elif x > -0.05 and x < 0.05: return "neutral"
    else: return "na"
df_strings = df.applymap(replace_values)

def replace_values(x):
    if x == np.nan: return np.nan
    if x > 0.05: return 1
    elif x < -0.05: return -1
    elif x > -0.05 and x < 0.05: return 0
    else: return np.nan
df_ints = df.applymap(replace_values)

#####################

''' sum of 1s '''

sum_list = []
for col in list(df_strings):
    counts = df_strings[col].value_counts()
    for value, count in counts.items():
        sum_list.append({'method': col, 'sentiment': value, 'sum': count})
df_sum = pd.DataFrame(sum_list)

df_sum.to_csv("../results/exp2/sums.tsv", index=False, header=True, sep="\t")
del col, count, counts, df_sum, sum_list, value
del df_strings

#####################

''' plots '''

''' heatmap '''
#colors = ['#FF6961', '#D3D3D3', '#77DD77'] # corresponding to -1, 0, 1
colors = ['#9bdea9', '#a0b9b9', '#9bc6de'] # corresponding to -1, 0, 1
custom_cmap = ListedColormap(colors)

ax = sns.heatmap(df_ints, cmap=custom_cmap, cbar=False, vmin=-1, vmax=1, center=0) # cbar/legend?
cbar = ax.collections[0].colorbar

ax.set_title('Heatmap', fontsize=16)
ax.set_xlabel('Methods', fontsize=12)
ax.set_ylabel('Row', fontsize=12)
plt.savefig('../plots/exp2/heatmap.png', format='png', dpi=300, bbox_inches='tight')
plt.close()
del ax, cbar, colors

''' correlation '''
correlation = df.corr()
ax = sns.heatmap(correlation, annot=True, cmap='GnBu', vmin=0, vmax=1)#, center=0)
ax.set_title('Correlation', fontsize=16)
plt.savefig('../plots/exp2/corr.png', format='png', dpi=300, bbox_inches='tight')
plt.close()
del ax, correlation

''' box plot '''
sns.set_palette("Set2")
ax = df.boxplot() 
ax.set_title('Boxplot', fontsize=16)
ax.set_xlabel('Methods', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.5)
ax.grid(False)
plt.savefig('../plots/exp2/boxplot.png', format='png', dpi=300, bbox_inches='tight')
plt.close()
del ax

''' violin '''
df_floats_melted = df.melt(var_name='Method', value_name='Score')
sns.set_palette("crest")
ax = sns.violinplot(x='Method', y='Score', data=df_floats_melted)
ax.set_title('Violin plot', fontsize=16)
plt.savefig('../plots/exp2/violin.png', format='png', dpi=300, bbox_inches='tight')
plt.close()
del ax, df_floats_melted

#####################

df = df_ints.copy()
del df_ints
df = df.replace({np.nan:-10,}, regex=True)

#####################

''' statistics '''

results = pd.DataFrame(columns=[
    'Method', 
    'Accuracy', 'Kappa', 'Agreement', 'Correlation', 
    'T-statistic', 'P-value',
    ])

inner_methods = {
    "None":[],
    "Minimal":[],
    "Weak":[],
    "Moderate":[],
    "Strong":[],
    "Almost perfect":[],
    }

manual = df["manual"]
   
for col in methods:
    pred = df[col]
    
    t_stat, p_value = stats.ttest_rel(manual, pred)
    
    acc = round(accuracy_score(manual, pred), 2)    
    kappa = round(cohen_kappa_score(manual, pred), 2)
    
    if kappa <= 0.2: agreement = "None"
    elif kappa <= 0.39: agreement = "Minimal"
    elif kappa <= 0.59: agreement = "Weak"
    elif kappa <= 0.79: agreement = "Moderate"
    elif kappa <= 0.89: agreement = "Strong"
    elif kappa >= 0.9: agreement = "Almost perfect"
    
    mcc = round(matthews_corrcoef(manual, pred), 2)
    
    results.loc[len(results)] = [
        col,
        acc,kappa,agreement,mcc,
        round(t_stat, 2), p_value,
        ]
    
    
    ''' innner method agreement (pairwise kappa) '''
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            kappa = cohen_kappa_score(df[methods[i]], df[methods[j]])
            
            if kappa <= 0.2: inner_methods["None"].append(sorted([methods[i], methods[j]]))
            elif kappa <= 0.39: inner_methods["Minimal"].append(sorted([methods[i], methods[j]]))
            elif kappa <= 0.59: inner_methods["Weak"].append(sorted([methods[i], methods[j]]))
            elif kappa <= 0.79: inner_methods["Moderate"].append(sorted([methods[i], methods[j]]))
            elif kappa <= 0.89: inner_methods["Strong"].append(sorted([methods[i], methods[j]]))
            elif kappa >= 0.9: inner_methods["Almost perfect"].append(sorted([methods[i], methods[j]]))


del acc, agreement, col, kappa, manual, mcc, pred
del t_stat, p_value
del i, j

results.to_csv("../results/exp2/stats.tsv", index=False, header=True, sep="\t")

#####################

''' agreements for computational methods '''

inner_methods_cleaned = {}

for key,value in inner_methods.items():
    
    if value:
        new_value = []
        for v in value:
            new_value.append("%s+%s" % (v[0],v[1]) )
        new_value = list(set(new_value))
        
        new_value = [re.split(r'[_+]', x) for x in new_value]
        new_value = [[v[0], v[1]] for v in new_value]
        
        inner_methods_cleaned[key] = new_value
    else: pass
del key, new_value, v, value

# make into dataframe
rows = []
for key, value in inner_methods_cleaned.items():
    for v in value:
        pass
        rows.append([v[0], v[1], key])
inner_methods = pd.DataFrame(rows, columns=['method1', 'method2', 'agreement'])
del key, rows, v, value, inner_methods_cleaned

''' minor note here: ChatGPT helped debug me with this function! '''
def calculate_kappa(row):
    col1 = f"{row['method1']}"
    col2 = f"{row['method2']}"
    if col1 in df.columns and col2 in df.columns:
        list1 = df[col1]
        list2 = df[col2]
        return round(cohen_kappa_score(list1, list2), 2)
    else:
        return None
inner_methods['kappa'] = inner_methods.apply(calculate_kappa, axis=1)

inner_methods = inner_methods.sort_values(['kappa'], ascending=[False])

inner_methods.to_csv("../results/exp2/inner_methods.tsv", index=False, header=True, sep="\t")

#####################

df = df[df['manual'] != -10]

results_filter = pd.DataFrame(columns=[
    'Method', 
    'Accuracy', 'Kappa', 'Agreement', 'Correlation', 
    'T-statistic', 'P-value',
    ])

manual = df["manual"]
   
for col in methods:
    pred = df[col]
    
    t_stat, p_value = stats.ttest_rel(manual, pred)
    
    acc = round(accuracy_score(manual, pred), 2)    
    kappa = round(cohen_kappa_score(manual, pred), 2)
    
    if kappa <= 0.2: agreement = "None"
    elif kappa <= 0.39: agreement = "Minimal"
    elif kappa <= 0.59: agreement = "Weak"
    elif kappa <= 0.79: agreement = "Moderate"
    elif kappa <= 0.89: agreement = "Strong"
    elif kappa >= 0.9: agreement = "Almost perfect"
    
    mcc = round(matthews_corrcoef(manual, pred), 2)
    
    results_filter.loc[len(results_filter)] = [
        col,
        acc,kappa,agreement,mcc,
        round(t_stat, 2), p_value,
        ]
    
del acc, agreement, col, kappa, manual, mcc, pred
del t_stat, p_value

results_filter.to_csv("../results/exp2/stats_filter.tsv", index=False, header=True, sep="\t")

#####################

# end of script

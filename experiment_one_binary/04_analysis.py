# -*- coding: utf-8 -*-
"""
@date: 2024-10-17
@author: sap218
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef

import seaborn as sns
import matplotlib.pyplot as plt

#####################

df = pd.read_csv("../spotify_user_reviews/exp1/DATASET_randomised_annotated_catch_llm.csv")
del df["review"]
df.to_csv("../results/exp1/annotations.tsv", index=False, header=True, sep="\t")

categories = ["advertisement","audio","download","update"]

methods = ["catch","gemma2","llama","mixtral"]

#####################

''' sum of 1s '''

t = pd.DataFrame(columns=['index', 0])
for of_interest in categories:
    df_sum = df.loc[:, df.columns.str.contains(of_interest)]
    df_sum = df_sum[df_sum[of_interest] != 0]
    df_sum = df_sum.sum().to_frame().reset_index()#.T
    t = pd.concat([t, df_sum], ignore_index=True)
t.to_csv("../results/exp1/sums_manualfilter.tsv", index=False, header=True, sep="\t")

df_sum = df.sum().to_frame()#.T
df_sum = df_sum.reset_index().sort_values('index')

df_sum.to_csv("../results/exp1/sums_all.tsv", index=False, header=True, sep="\t")
del df_sum, t

#####################

''' plots '''

for of_interest in categories:
    cols = [of_interest]
    for method in methods: cols.append(of_interest + "_" + method)
    
    df_filter = df[cols]
    
    new_cols = ["manual"]
    for x in list(df_filter):
        if x == of_interest: pass
        else: new_cols.append(x.replace("%s_" % of_interest,""))
    df_filter.columns = new_cols
    
    ''' bar plot '''
    ax = df_filter.sum().plot(kind='bar', color=['#FFD1BA','#D4B9FF','#FFC2C7','#B0E0E6','#FFFACD'])
    ax.set_title('Bar plot of total of 1s for %s' % of_interest, fontsize=16)
    ax.set_xlabel('Methods', fontsize=12)
    ax.set_ylabel('Sum', fontsize=12)
    plt.xticks(rotation=0)
    plt.savefig('../plots/exp1/%s_bar.png' % of_interest, dpi=300, bbox_inches='tight')
    plt.close()
    
    #df_filter.cumsum().plot()

    ''' heatmap '''
    ax = sns.heatmap(df_filter, cmap='Blues', cbar=False)
    ax.set_title('Heatmap of 1s for %s' % of_interest, fontsize=16)
    ax.set_xlabel('Methods', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    plt.savefig('../plots/exp1/%s_heatmap.png' % of_interest, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    ''' correlation '''
    correlation = df_filter.corr()
    ax = sns.heatmap(correlation, annot=True, cmap='GnBu', vmin=0, vmax=1)#, center=0)
    ax.set_title('Correlation of %s' % of_interest, fontsize=16)
    plt.savefig('../plots/exp1/%s_corr.png' % of_interest, format='png', dpi=300, bbox_inches='tight')
    plt.close()

del ax, cols, correlation, df_filter, new_cols, x

#####################

''' statistics '''

results = pd.DataFrame(columns=[
    'Word','Method',
    'Accuracy', 'Kappa', 'Agreement', 'Correlation', 
    'Precision', 'Recall', 'F1-score',
    ])

inner_methods = {
    "None":[],
    "Minimal":[],
    "Weak":[],
    "Moderate":[],
    "Strong":[],
    "Almost perfect":[],
    }

#####################

for of_interest in categories:
    
    manual = df[of_interest]
    
    cols = []
    for method in methods: cols.append(of_interest + "_" + method)
     
        
    for col in cols:
    
        pred = df[col]
        
        method = col.split("_")[1]
        
        acc = round(accuracy_score(manual, pred), 2)
        prec = round(precision_score(manual, pred), 2)
        recall = round(recall_score(manual, pred), 2)
        f1 = round(f1_score(manual, pred), 2)
        
        kappa = round(cohen_kappa_score(manual, pred), 2)
        
        if kappa <= 0.2: agreement = "None"
        elif kappa <= 0.39: agreement = "Minimal"
        elif kappa <= 0.59: agreement = "Weak"
        elif kappa <= 0.79: agreement = "Moderate"
        elif kappa <= 0.89: agreement = "Strong"
        elif kappa >= 0.9: agreement = "Almost perfect"
        
        mcc = round(matthews_corrcoef(manual, pred), 2)
        
        results.loc[len(results)] = [
            of_interest,method,
            acc,kappa,agreement,mcc,
            prec,recall,f1,
            ]
        
        
        ''' innner method agreement (pairwise kappa) '''
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                kappa = cohen_kappa_score(df[cols[i]], df[cols[j]])
                
                if kappa <= 0.2: inner_methods["None"].append(sorted([cols[i], cols[j]]))
                elif kappa <= 0.39: inner_methods["Minimal"].append(sorted([cols[i], cols[j]]))
                elif kappa <= 0.59: inner_methods["Weak"].append(sorted([cols[i], cols[j]]))
                elif kappa <= 0.79: inner_methods["Moderate"].append(sorted([cols[i], cols[j]]))
                elif kappa <= 0.89: inner_methods["Strong"].append(sorted([cols[i], cols[j]]))
                elif kappa >= 0.9: inner_methods["Almost perfect"].append(sorted([cols[i], cols[j]]))


del acc, agreement, col, cols, f1, i, j, kappa, manual, mcc, prec, pred, recall

results.to_csv("../results/exp1/stats.tsv", index=False, header=True, sep="\t")

#####################

''' agreements for computational methods '''

inner_methods_cleaned = {}

for key,value in inner_methods.items():
    
    new_value = []
    for v in value:
        new_value.append("%s+%s" % (v[0],v[1]) )
    new_value = list(set(new_value))
            
    new_value = [re.split(r'[_+]', x) for x in new_value]
    new_value = [[v[0], v[1], v[3]] for v in new_value]
    
    inner_methods_cleaned[key] = new_value
del key, new_value, v, value

# make into dataframe
rows = []
for key, value in inner_methods_cleaned.items():
    for v in value:
        rows.append([v[0], v[1], v[2], key])
inner_methods = pd.DataFrame(rows, columns=['topic', 'method1', 'method2', 'agreement'])
del key, rows, v, value, inner_methods_cleaned

''' minor note here: ChatGPT helped debug me with this function! '''
def calculate_kappa(row):
    col1 = f"{row['topic']}_{row['method1']}"
    col2 = f"{row['topic']}_{row['method2']}"
    if col1 in df.columns and col2 in df.columns:
        list1 = df[col1]
        list2 = df[col2]
        return round(cohen_kappa_score(list1, list2), 2)
    else:
        return None
inner_methods['kappa'] = inner_methods.apply(calculate_kappa, axis=1)

inner_methods = inner_methods.sort_values(['topic','kappa'], ascending=[True,False])

inner_methods.to_csv("../results/exp1/inner_methods.tsv", index=False, header=True, sep="\t")

#####################

# end of script

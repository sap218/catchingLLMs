# -*- coding: utf-8 -*-
"""
@date: 2025-01-14
@author: sap218
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import friedmanchisquare
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef

import seaborn as sns
import matplotlib.pyplot as plt

#####################

df = pd.read_csv("../spotify_user_reviews/exp3/DATASET_randomised_annotated_catch_llm.csv")
del df["review"]

df.rename(columns={'annotate': 'manual'}, inplace=True)

new_order = ['manual',
             'catch_precise','catch_broad',
             'gemma2_precise','gemma2_broad',
             'llama_precise','llama_broad',
             'mixtral_precise','mixtral_broad',
             ]
df = df[new_order]
del new_order
manual = list(df["manual"])

df.to_csv("../results/exp3/annotations.tsv", index=False, header=True, sep="\t")

#####################

''' sum of 1s '''

df_sum = df.sum().to_frame()#.T
df_sum = df_sum.reset_index().sort_values('index')

df_sum.to_csv("../results/exp3/sums.tsv", index=False, header=True, sep="\t")
del df_sum

#####################
#####################

''' precise vs. broad in each method '''

results = pd.DataFrame(columns=['Method', 'McNemar', 'P-value',])

methods = ["catch","gemma2","llama","mixtral"]
#r = methods[0]

dfo = df.copy()

for r in methods:
    
    plottitlevar = "\nApp payments w/ %s method" % r
    
    df = dfo.filter(like=r)
    
    #####################
    
    new_cols = []
    for x in list(df):
        new_cols.append(x.replace("%s_" % r,""))
    df.columns = new_cols
    del new_cols, x
    
    ''' bar plot '''
    plt.figure(figsize=(5, 4))
    ax = df.sum().plot(kind='bar', color=['#ebbbb9','#b0e6be'])
    ax.set_title('Bar plot of total of 1s for %s' % plottitlevar, fontsize=16)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Sum', fontsize=12)
    plt.xticks(rotation=0)
    plt.savefig('../plots/exp3/bar_%s.png' % r, dpi=300, bbox_inches='tight')
    plt.close()
    
    #####################
    
    ''' statistics '''
    
    contingency_table = pd.crosstab(df['precise'], df['broad'])
    
    ''' heatmap '''
    plt.figure(figsize=(5, 4))
    sns.heatmap(contingency_table, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 16})
    plt.title('Contingency Heatmap of %s ' % plottitlevar, fontsize=16)
    plt.xlabel('Broad', fontsize=12)
    plt.ylabel('Precise', fontsize=12)
    plt.savefig('../plots/exp3/heatmap_%s.png' % r, dpi=300, bbox_inches='tight')
    plt.close()
    del ax, plottitlevar
    
    mcnemartest = mcnemar(contingency_table, exact=True)
    statistic = mcnemartest.statistic
    pvalue = mcnemartest.pvalue
    del mcnemartest
    
    results.loc[len(results)] = [
        r, statistic, pvalue]
    
del contingency_table, df, pvalue, statistic

results.to_csv("../results/exp3/stats_pvb.tsv", index=False, header=True, sep="\t")
del results

#####################
#####################

''' a version of methods (precise OR broad) '''

results = pd.DataFrame(columns=['Test', 'Friedman', 'P-value',])

methods = ["precise", "broad"]
#r = methods[1]

for r in methods:

    df = dfo.filter(like=r)
    
    df.insert(0, 'manual', manual) 
    
    new_cols = []
    for x in list(df):
        new_cols.append(x.replace("_%s" % r,""))
    df.columns = new_cols
    del new_cols, x
    
    #####################
    
    ''' plots '''
    
    ''' heatmap '''
    ax = sns.heatmap(df, cmap='Blues', cbar=False)
    ax.set_title('Heatmap of %s searches' % r, fontsize=16)
    ax.set_xlabel('Methods', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    plt.savefig('../plots/exp3/%s_heatmap.png' % r, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    del ax
    
    ''' correlation '''
    correlation = df.corr()
    ax = sns.heatmap(correlation, annot=True, cmap='GnBu', vmin=0, vmax=1)#, center=0)
    ax.set_title('Correlation of %s searches' % r, fontsize=16)
    plt.savefig('../plots/exp3/%s_corr.png' % r, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    del ax, correlation
    
    #####################
    
    ''' statistics '''
    
    m = ['manual', 'catch', 'gemma2', 'llama', 'mixtral']
    friedman, pvalue = friedmanchisquare(df[m[0]], df[m[1]], df[m[2]], df[m[3]], df[m[4]])
    results.loc[len(results)] = [str(r+"all"), friedman, pvalue]

    friedman, pvalue = friedmanchisquare(df[m[1]], df[m[2]], df[m[3]], df[m[4]])
    results.loc[len(results)] = [str(r+"comp"), friedman, pvalue]
    
del friedman, pvalue, m

results.to_csv("../results/exp3/stats_mvm.tsv", index=False, header=True, sep="\t")
del results

#####################
#####################

''' broads vs. manual '''

results = pd.DataFrame(columns=[
    'Task','Method',
    'Accuracy', 'Kappa', 'Agreement', 'Correlation', 
    'Precision', 'Recall', 'F1-score',
    ])

#####################

methods = ["precise", "broad"]
#r = methods[1]

#####################

for r in methods:

    df = dfo.filter(like=r)

    new_cols = []
    for x in list(df):
        new_cols.append(x.replace("_%s" % r,""))
    df.columns = new_cols
    del new_cols, x

    #del df["manual"]
    cols = list(df)

    for col in cols:
        pred = df[col]
        
        #method = col.split("_")[1]
        
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
            r, col,
            acc,kappa,agreement,mcc,
            prec,recall,f1,
            ]

del acc, agreement, col, cols, f1, kappa, mcc, prec, pred, recall

results.to_csv("../results/exp3/stats_manual.tsv", index=False, header=True, sep="\t")
del results

#####################

# end of script

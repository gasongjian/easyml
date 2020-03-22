# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 14:09:46 2018

@author: JSong
"""
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
#%maplotlib inline
import seaborn as sns
#from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
#import pysnooper



__all__=['score']



# def check_array(x):
#     return x


#@pysnooper.snoop()
def score(y_true,y_pred,y_score=None,groupid=None,data=None,objective='auto',output='auto',**kwargs):
    '''分类问题的指标计算
    param::
        y_true : 真实值
        y_pred : 预测值
        y_score : 模型输出值
        groupid : 分组标识符
        data : 数据集，如果前面四个参数是字符串，则使用该数据
        objective : reg、binary、multi、rank、quantile
        ordered : 多分类任务中，label是否有序
        categories: 多分类任务中，假设label有序，则使用该序
        top_k : 适合排序任务，输出precision@k 等
        output : 'auto'
    return::
        result:pd.DataFrame，存储相关结果
        crrosstab:列联表
    '''
    from sklearn.utils.multiclass import type_of_target
    output = {
        'reg':['rmse','rmsle','mae','mape','r2','corrcoef']
        ,'binary':['precision','recall','auc','f1','acc']
        ,'multi':['precision','recall','auc','f1','f1_micro','f1_macro','f1_weighted','acc','acc_lr']
        ,'rank':['precision@k','recall@k','ndcg@k','support']
        ,'quantile':['alpha_quantile','rmse','mae_quantile','mape','r2']
    }
    
    # 数据源预处理
    if isinstance(y_true,str) and isinstance(data,pd.DataFrame):
        y_true = np.array(data[y_true])
        y_pred = np.array(data[y_pred])
        if isinstance(groupid,str):
            groupid = data[groupid]
    else:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
    # 模型任务判别
    if objective == 'auto':
        labels = set(np.unique(y_true))
        if labels == set([0,1]) and groupid is None:
            objective = 'binary'
        elif labels == set([0,1]) and groupid is not None:
            objective = 'rank'
        elif type_of_target(y_true) == 'continuous':
            objective = 'reg'
        elif type_of_target(y_true) == 'multiclass':
            objective = 'multi'
        else:
            objective = 'unknown'
    if objective not in output:
        return None,None
    if objective in ['binary','multi']:
        '''
        对于多分类:
        f1_micro = 2*p*r/(p+r),p = \sum TP/(\sum TP+\sum FP),r = \sum TP/(\sum TP+\sum FN)
        f1_macro = (\sum f1) /n
        f1_weight = \sum f1*weight
        注意：f1_micro == acc         
        '''
        labels=sorted(np.unique(y_true))
        crosstab = pd.crosstab(y_true,y_pred).loc[labels,labels]
        crosstab = crosstab.rename_axis(index='true',columns='pred')
        p=metrics.precision_score(y_true,y_pred,labels=labels,average=None)
        r=metrics.recall_score(y_true,y_pred,labels=labels,average=None)
        acc=metrics.accuracy_score(y_true,y_pred)
        f1=2*p*r/(p+r)
        result=pd.DataFrame({'precision':p,'recall':r,'f1':f1,'acc':acc},index=labels)
        if objective == 'multi':
            f1_micro = metrics.f1_score(y_true,y_pred,labels=labels,average='micro')
            f1_macro = metrics.f1_score(y_true,y_pred,labels=labels,average='macro')
            f1_weighted = metrics.f1_score(y_true,y_pred,labels=labels,average='weighted')
            result['f1_micro'] = f1_micro
            result['f1_macro'] = f1_macro
            result['f1_weighted'] = f1_weighted
        if y_score is not None:
            auc=metrics.roc_auc_score(pd.get_dummies(y_true,columns=labels),y_score, average=None)
            result['auc'] = auc
        if kwargs.get('ordered',False):
            if kwargs.get('categories',None) is not None:
                labels = list(kwargs['categories'])
            crosstab = crosstab.loc[labels,labels]
            acc_lr=(np.diag(crosstab).sum()+np.diag(crosstab,k=1).sum()+np.diag(crosstab,k=-1).sum())/crosstab.sum().sum()
            #acc_l=(np.diag(crosstab).sum()+np.diag(crosstab,k=-1).sum())/crosstab.sum().sum()
            #acc_r=(np.diag(crosstab).sum()+np.diag(crosstab,k=1).sum())/crosstab.sum().sum()
            result['acc_lr'] = acc_lr
            
    elif objective in ['rank']:
        '''
        一些说明
        1. 此时y_pred 是rnk
        2. 参考 https://hivemall.incubator.apache.org/userguide/eval/rank.html
        计算公式
        
        '''
        k = kwargs.get('top_k',10)
        n_group = len(np.unique(groupid))
        index = np.arange(1,k+1)
        columns = ['precision@k','recall@k','ndcg@k']
        data = pd.DataFrame({'groupid':groupid,'y_true':y_true,'y_pred':y_pred})
        # 混淆矩阵
        crosstab = data.groupby(['y_pred','y_true'])['groupid'].unique().map(lambda x:len(x)).unstack().T
        # TOPK命中率： n次查询中, 用户购买的在前K个的占比
        # precision@k： 
        result = pd.DataFrame(index=np.arange(1,k+1),columns = ['precision@k','recall@k','ndcg@k','support'])
        # 第k个推荐命中的用户数
        result['support'] = crosstab.loc[1,:k]
        for kk in np.arange(1,k+1):
            tmp = data.loc[data.y_pred<=kk,:]
            # 前k个推荐命中的用户数
            tp = tmp.loc[tmp.y_true==1,'groupid'].unique().shape[0]
            result.loc[kk,'precision@k'] = tp/tmp.shape[0]
            result.loc[kk,'recall@k'] = tp/n_group
            tmp['rnk_true'] = tmp.groupby('groupid')['y_true'].apply(lambda x:x.rank(method='min',ascending=False))
            tmp['s1'] = (np.power(2,tmp['y_true'])-1)/np.log2(1+tmp['y_pred'])
            tmp['s2'] = (np.power(2,tmp['y_true'])-1)/np.log2(1+tmp['rnk_true'])
            ndcg = (tmp.groupby('groupid')['s1'].sum()/tmp.groupby('groupid')['s2'].sum()).fillna(0).mean()
            result.loc[kk, 'ndcg@k'] = ndcg
            
    elif objective in ['reg']:
        r2 = metrics.r2_score(y_true,y_pred)
        rmse = np.sqrt(np.mean((y_pred-y_true)**2))
        rmsle = np.sqrt(np.nanmean((np.log(1+y_pred)-np.log(1+y_true))**2))
        mae = np.mean(np.abs(y_pred-y_true))
        mape = np.mean(np.abs(y_pred-y_true)/y_true)
        corrcoef = np.corrcoef(y_true,y_pred)[0,1]
        result = pd.DataFrame({'score':[rmse,rmsle,mae,mape,r2,corrcoef]},index = ['rmse','rmsle','mae','mape','r2','corrcoef']).T
        crosstab = None
    elif objective in ['quantile'] and kwargs.get('alpha',-1)>=0:
        alpha = kwargs['alpha']
        r2 = metrics.r2_score(y_true,y_pred)
        rmse = np.sqrt(np.mean((y_pred-y_true)**2))
        mae = np.mean((alpha-1)*(y_true-y_pred)*(y_true-y_pred<0)+alpha*(y_true-y_pred)*(y_true-y_pred>=0))
        mape = np.mean(np.abs(y_pred-y_true)/y_true)
        alpha_quantile = np.sum(y_pred>=y_true)/len(y_true)
        result = pd.DataFrame({'score':[alpha_quantile,rmse,mae,mape,r2]},index = ['alpha_quantile','rmse','mae_quantile','mape','r2']).T
        crosstab = None
    else:
        return None,None
      
    columns = [c for c in output[objective] if c in result.columns]
    result = result[columns]
            
    return result,crosstab

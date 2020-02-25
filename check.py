import statsmodels.api as sm
from statsmodels.sandbox.nonparametric import kernels
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


'''特征空间解析

我们将特征类型分为如下四种
- numeric：连续的特征，表现为可以定义序关系且唯一值可以有无数个
- category：类别型特征
- Multi-category：多类别
- object：无结构数据，暂不提供任何解析
'''



def describe_numeric_1d(series,quantiles=None,missing_value = None):
    """Describe a numeric series.

    Args:
        series: The Series to describe.
        quantiles: list,like [0.25,0.75].

    Returns:
        A dict containing calculated series description values.

    """
    if quantiles is None:
        quantiles = [0.005,0.01,0.05,0.25,0.5,0.75,0.95,0.99,0.995]
        
    n = len(series)
    if missing_value:
        series = series.replace({missing_value:np.NaN})
    stats = {
        "mean": series.mean(),
        "std": series.std(),
        "variance": series.var(),
        "min": series.min(),
        "max": series.max(),
        "kurtosis": series.kurt(),
        "skewness": series.skew(),
        "zeros": (n - np.count_nonzero(series))/n,
        "missing":np.sum(series.isna())/n
    }
    stats.update({
        "{:.1%}".format(percentile).replace('.0',''): value
        for percentile, value in series.quantile(quantiles).to_dict().items()
        })
    stats["iqr"] = stats["75%"] - stats["25%"]
    stats["cv"] = stats["std"] / stats["mean"] if stats["mean"] else np.NaN
    return stats


def _get_distrbution(x,x_lim=None,gridsize=None,bw=None,bw_method='scott',eta=1):
    '''
    ## brandwidth select
       A = min(std(x, ddof=1), IQR/1.349)
       Scott's rule: 1.059 * A * n ** (-1/5.) 
       Silverman's Rule: .9 * A * n ** (-1/5.) 
    
    '''
    x = pd.Series(x)
    stats = describe_numeric_1d(x)
    x_mean_fix = x[(x>=stats['5%'])&(x<=stats['95%'])].mean()

    # 截断数据用于分析合理的密度函数
    if x_lim is None:
        cv_lower,cv_upper = x[x<=stats['5%']].std()/(abs(x_mean_fix)+1e-14), x[x>=stats['95%']].std()/(abs(x_mean_fix)+1e-14)
        x_lim = [stats['5%'] if cv_lower>=eta else stats['min'],stats['95%'] if cv_upper>=eta else stats['max']]

    domain = [stats['min'],stats['max']]
    if cv_lower>=eta:
        domain[0] = -np.inf
    if cv_upper>=eta:
        domain[1] = np.inf

    # 选择绘图和计算需要的网格大小
    try:
        bw = float(bw)
    except:
        bw = sm.nonparametric.bandwidths.select_bandwidth(x,bw=bw_method,kernel = None)
    # 特征的样本数一般较大，这里规定gridsize最小为 128
    n_fix = len(x[(x>=x_lim[0])&(x<=x_lim[1])])
    if gridsize is None:
        gridsize=max(128,min(int(np.round((x_lim[1] - x_lim[0])/bw)),n_fix)) if bw>=1e-7 else None
    
    dens = sm.nonparametric.KDEUnivariate(x.dropna().astype(np.double).values)
    dens.fit(gridsize=gridsize,bw=bw,clip=x_lim)

    # 获取最新的 brandwidth 等数据
    bw = dens.bw
    # bw_method = bw_method if dens.bw_method = 'user-given' else dens.bw_method
    gridsize =  len(dens.support)
    
    result = stats
    result.update({'key_dist':['bw','bw_method','support','density','x_lim','cdf','icdf','domain','gridsize','evaluate']
        ,'bw':bw
        ,'bw_method':bw_method
        ,'support':dens.support
        ,'density':dens.density
        ,'x_lim':x_lim
        ,'cdf':dens.cdf
        ,'icdf':dens.icdf
        ,'domain':domain
        ,'gridsize':gridsize
        ,'evaluate':dens.evaluate
    })
    return result


class feature_numeric(object):
    def __init__(self,name=None):
        self.name = name
        self.dtype = 'numeric'
        self.stats = None
        self.dist = None
        self.cross_proba=None
        self.cross_stats=None

    # 随机取样
    def sample(self,n):
        return self.dist['icdf'][np.random.randint(low=0,high = self.dist['gridsize'] -1,size=n)]
    
    def pdf(self,x):
        return self.dist['evaluate'](x)
    
    def describe(self):
        return self.stats
    
    def get_values(self,key):
        if key in self.dist:
            return self.dist[key]
        elif key in self.stats:
            return self.stats[key]
        elif key in self.cross_proba:
            return self.cross_proba[key]
        elif key in self.cross_stats:
            return self.cross_stats[key]
        else:
            return None
        
    def fit(self,x,y=None,**arg):
        result = _get_distrbution(x,**arg)
        self.stats = {key:value for key,value in result.items() if key not in result['key_dist']+['key_dist']}
        self.dist = {key:value for key,value in result.items() if key in result['key_dist']}
        if y is not None and len(x) == len(y):
            cross_proba,cross_stats = self.crosstab_bin(x,y)
            self.cross_proba = cross_proba
            self.cross_stats = cross_stats
    
    def crosstab_bin(self,x,y):
        
        x = pd.Series(x)
        y = pd.Series(y)
        n = len(y)
        dist_y = y.value_counts()/n

        bw = self.dist['bw']
        support = self.dist['support']
        domain = self.dist['domain']
        q995 = self.stats['99.5%']
        gridsize = self.dist['gridsize']
        seq = np.mean(support[1:] - support[0:-1])

        # 添加额外的支撑集，便于分析合理的泛化方法
        if domain[1] == np.inf:
            n_add = np.ceil((q995 - support[-1])/seq)
            support_add = [support[-1] + seq*(i+1) for i in range(int(n_add))]
            support_new = np.concatenate((support,support_add))
        else:
            support_new = support.copy()

        p_y1_x = np.zeros_like(support_new)
        cumulative = np.zeros_like(support_new)
        for i,xi in enumerate(support_new):
            ind =(x<=xi+bw)&(x>=xi-bw)
            tmp = y[ind].value_counts().to_dict()
            cnt = {0:dist_y[0],1:dist_y[1]}
            cnt[0] += tmp.get(0,0)
            cnt[1] += tmp.get(1,0)
            p_y1_x[i] = cnt[1]/(cnt[0]+cnt[1])
            cumulative[i] = np.sum(x<=xi)/n

        # 根据贝叶斯法则可以求出    
        p_x_y1 = self.dist['density']*p_y1_x[:gridsize]/dist_y[1]
        p_x_y0 = self.dist['density']*(1-p_y1_x[:gridsize])/dist_y[0]
        iv =np.sum((p_x_y1 - p_x_y0)*np.log2((1e-14+p_x_y1)/(p_x_y0+1e-14)))*seq

        cross_proba = {
            "p(y=1|x)":p_y1_x
            ,"p(x|y=1)":p_x_y1
            ,"p(x|y=0)":p_x_y0
            ,"woe(x)":np.log2(p_x_y0/p_x_y1)
            ,"cumulative":cumulative
            ,"support_x":support_new
            ,"support_y":support
        }

        cross_stats = {
            "iv":iv
            ,"p(y=1)":dist_y[1]
            ,"p(y=0)":dist_y[0]
        }

        return cross_proba,cross_stats
    
    
    def plot_pdf(self):
        x_min,x_max = self.stats['min'],self.stats['max']
        bw = self.dist['bw']
        if self.name:
            title = 'density curve of {}'.format(self.name)
        else:
            title = 'density curve'
        fig,ax=plt.subplots(figsize=[10,6.6])
        support = self.dist['support']
        #seq = np.mean(support[1:]-support[0:-1])
        #ind = (support>=x_min-3*seq)&(support<=x_max+3*seq)
        ax.plot(support,self.dist['density']);
        ax.set_title(title);
        ax.set_xlabel('range = [{},{}]'.format(x_min,x_max));
        fig.show()
        return None
    
    def summary(self):
        
        # 区分两个版本，一个有y 一个没 y
        tmp = pd.DataFrame(index=range(0,10),columns=['name1','value1','name2','value2','name3','value3'])
        tmp.name1 = ['missing','zeros','min','max','mean','std','skewness','kurtosis','cv','iqr']
        tmp.value1 = [self.stats[k] for k in tmp['name1'].values]
        tmp.name2 = ['0.5%','1%','5%','25%','50%','75%','95%','99%','99.5%','domain']
        tmp.value2 = [self.stats[k] for k in tmp['name2'][:-1].values]+[str(self.dist['domain'])]
        tmp.loc[0,'name3'] = 'iv'
        tmp.loc[0,'value3'] = self.cross_stats['iv']
        tmp.loc[1,'name3'] = 'p(y=1)'
        tmp.loc[1,'value3'] = self.cross_stats['p(y=1)']
        display(tmp)

        support_new = self.cross_proba['support_x']
        ind1 = (self.cross_proba['support_x']>=self.stats['min'])&(self.cross_proba['support_x']<=self.stats['max'])
        p_y1 = self.cross_stats['p(y=1)']
        
        fig,[ax1,ax2]=plt.subplots(2,1,figsize=[10,13])
        ax1.plot(support_new[ind1],self.cross_proba['p(y=1|x)'][ind1] ,'.');
        ax1.plot([support_new[0],support_new[-1]] ,[p_y1,p_y1],label = 'baseline')
        ax1_ = ax1.twinx()
        ax1_.plot(support_new[ind1],self.cross_proba['cumulative'][ind1],label = 'cumulative',color='red')
        ax1_.legend(loc = 'center left')
        ax1.set_title(r'$p(y=1|x)$');
        ax1.legend()

        ind2 = (self.cross_proba['support_y']>=self.stats['min'])&(self.cross_proba['support_y']<=self.stats['max'])
        ax2.plot(self.cross_proba['support_y'],self.cross_proba['p(x|y=1)'],label=r'$p(x|y=1)$')
        ax2.plot(self.cross_proba['support_y'],self.cross_proba['p(x|y=0)'],label=r'$p(x|y=0)$')
        ax2.plot(self.cross_proba['support_y'],self.dist['density'],label=r'$p(x)$',color = '0.5',linestyle='--')
        ax2_ = ax2.twinx()
        ax2_.plot(self.cross_proba['support_y'][ind2],self.cross_proba['woe(x)'][ind2],label = 'woe(x)',color='red')
        ax2_.legend(loc = 'center right')
        ax2.legend()
        ax2.set_title(r'$p(x|y=1)$ vs $p(x|y=0)$')
        ax2.set_xlabel('iv = {:.2f}'.format(self.cross_stats['iv']))

        fig.show()
        
        
def sample_size_cal(p,alpha=0.05,e=0.05):
    import scipy.stats as stats
    z=stats.norm.ppf(1-alpha/2)
    return int(np.ceil(z**2*p*(1-p)/e**2))

def describe_categorical(x
                         ,missing_value = None
                         ,pct_pos = 0.5
                         ,backoff_p = 0.05
                         ,backoff_rnk = 30
                         ,backoff_n = None
                         ,alpha=0.05
                         ,e=0.05):
    x = pd.Series(x)
    if missing_value:
        x = x.replace({str(missing_value):np.nan})
    n = len(x)
    missing = np.sum(x.isnull())
    p_x = x.value_counts().sort_values(ascending=False)/n
    itemlist = p_x.index.tolist()
    # 识别稀有类
    if backoff_n is None:
        backoff_n = sample_size_cal(pct_pos,alpha=alpha,e=e)
    x_base = pd.DataFrame(x.value_counts().sort_values(ascending=False),index=itemlist)
    x_base.columns = ['cnt']
    x_base['proba'] = x_base['cnt']/n
    x_base['type'] = 'normal'
    x_base['rnk'] = range(1,len(x_base)+1)
    x_base.loc[((x_base.proba<backoff_p)&(x_base.cnt<backoff_n))|(x_base.rnk>=backoff_rnk),'type'] = 'rare'
    stats = {
        "missing": missing/n,
        "distinct_count":len(itemlist),
        "n":n,
        "entropy":-1*np.sum(p_x*np.log2(p_x))
    }
    dist = {
        "itemlist":itemlist,
        "p(x)":p_x,
        "type":x_base['type'].to_dict(),
        "itemlist_rare":x_base[x_base.type=='rare'].index.tolist(),
        "data":x_base
    }
    return stats,dist



class feature_categorical(object):
    def __init__(self,name=None):
        self.name = name
        self.dtype = 'categorical'
        self.stats = None
        self.dist = None
        self.cross_proba=None
        self.cross_stats=None
        
    def crosstab_bin(self,x,y):
        x = pd.Series(x)
        y = pd.Series(y)
        n = x.shape[0]
      
        p_x = x.value_counts().sort_values(ascending=False)/n
        h_x = -1*np.sum(p_x*np.log2(p_x))
        p_y = y.value_counts()/n
        
        # woe 等需要知道 y=1 中 missing的缺失率
        n_y_missing = {1:0,0:0}
        n_missing = np.sum(x.isnull())
        if n_missing>=1:
            n_y_missing.update(y[x.isnull()].value_counts().to_dict())
            cross_missing = {
                "p(missing|y=1)":n_y_missing[1]/(p_y[1]*n)
                ,"p(missing|y=0)":n_y_missing[0]/(p_y[0]*n)
                ,"p(y=1|missing)":n_y_missing[1]/(n_y_missing[0]+n_y_missing[1])
                ,"p(y=0|missing)":n_y_missing[0]/(n_y_missing[0]+n_y_missing[1])
            }
        else:
            cross_missing = {
                "p(missing|y=1)":0
                ,"p(missing|y=0)":0
                ,"p(y=1|missing)":np.nan
                ,"p(y=0|missing)":np.nan
            }
       
        # 为避免部分类别项不同时存在正负样本，统计给每一个类别都加一个样本
        p_xy = (pd.crosstab(x,y)+[p_y[0],p_y[1]])/n
        p_x_y = p_xy.div(p_xy.sum(axis=0),axis=1)
        p_y_x = p_xy.div(p_xy.sum(axis=1),axis=0)
        p_xy_expected = pd.DataFrame(np.dot(pd.DataFrame(p_x),pd.DataFrame(p_y).T),index=p_x.index,columns=p_y.index)    
        info_gain = (p_xy*np.log2(p_xy/p_xy_expected)).sum().sum()
        info_gain_ratio = info_gain/h_x
        
        cross_proba = {
            "p(y)":p_y
            ,"p(x,y)":p_xy
            ,"p(x)p(y)":pd.DataFrame(np.dot(pd.DataFrame(p_x),pd.DataFrame(p_y).T),index=p_x.index,columns=p_y.index)
            ,"p(y=1|x)":p_y_x[1]
            ,"p(x|y=1)":p_x_y[1]
            ,"p(x|y=0)":p_x_y[0]
            ,"woe(x)":np.log2(p_x_y[0]/p_x_y[1])
            ,"cross_missing":cross_missing
        }

        cross_stats = {
            "iv":np.sum((p_x_y[0] - p_x_y[1])*np.log2(p_x_y[0]/p_x_y[1]))
            ,"p(y=1)":p_y[1]
            ,"p(y=0)":p_y[0]
            ,"info_gain":info_gain
            ,"info_gain_ratio":info_gain_ratio
        }

        return cross_proba,cross_stats
        
        
    def fit(self,x,y=None,missing_value=None,pct_pos=0.5,backoff_p=0.05,backoff_rnk=30
                         ,backoff_n=None,alpha=0.05,e=0.05):
        param = {'missing_value':missing_value,'pct_pos':pct_pos,'backoff_p':backoff_p,'backoff_rnk':backoff_rnk
                ,'backoff_n':backoff_n,'alpha':alpha,'e':e}
        stats,dist = describe_categorical(x,**param)
        
        self.stats = stats
        self.dist = dist
        if y is not None and len(x) == len(y):
            cross_proba,cross_stats = self.crosstab_bin(x,y)
            self.cross_proba = cross_proba
            self.cross_stats = cross_stats
            
    # 随机取样
    def sample(self,n,drop_na=True):
        itemlist = self.dist['itemlist']
        p=self.dist['p(x)'][itemlist]
        if drop_na and self.stats['missing']>0:
            itemlist+=[np.nan]
            p+=[self.stats['missing']]
        return np.random.choice(itemlist, n, p=p)
    
    def pdf(self,x):
        return self.dist['p(x)'][x]
    
    def describe(self):
        return self.stats,self.dist
    
    def get_values(self,key):
        if key in self.dist:
            return self.dist[key]
        elif key in self.stats:
            return self.stats[key]
        elif key in self.cross_proba:
            return self.cross_proba[key]
        elif key in self.cross_stats:
            return self.cross_stats[key]
        else:
            return None

    def plot_pdf(self):
        
        if self.name:
            title = 'frequency histogram of {}'.format(self.name)
        else:
            title = 'frequency histogram'
            
        x_base = self.dist['data']
        other = pd.Series({'Other values ({})'.format(len(x_base[x_base['type'] == 'rare']))
                           :x_base.loc[x_base['type'] == 'rare','proba'].sum()})
        tmp = x_base.loc[x_base.type == 'normal','proba']
        tmp = pd.concat([pd.Series({'(Missing)':self.stats['missing']}),tmp,other])

        fig,ax=plt.subplots(figsize=[10,6.6])
        sns.barplot(tmp.values*100,tmp.index,orient = 'h',ax=ax)
        ax.set_title(title)
        ax.set_xlabel('pct %')
        fig.show()

        
        
        
    def summary(self):
       
        if self.name:
            title = 'frequency histogram and woe(x) of {}'.format(self.name)
        else:
            title = 'frequency histogram and woe(x)'

        tmp = pd.DataFrame(index=range(0,6),columns=['name1','value1','name2','value2'])
        tmp.name1 = ['n','missing','distinct_count','distinct_count_normal','items_top3','entropy']
        tmp.value1 = [self.stats['n'],self.stats['missing'],self.stats['distinct_count'],self.stats['distinct_count']-len(self.dist['itemlist_rare'])
                      ,str(self.dist['itemlist'][:3]),self.stats['entropy']]

        tmp.name2 = ['p(y=1)','p(y=0)','iv','info_gain','info_gain_ratio',np.nan]
        tmp.value2 = [self.cross_stats[k] for k in tmp['name2'][:-1].values]+[np.nan]
        display(tmp)

        x_base = self.dist['data']
        other = pd.Series({'Other values ({})'.format(len(x_base[x_base['type'] == 'rare']))
                           :x_base.loc[x_base['type'] == 'rare','proba'].sum()})
        tmp = x_base.loc[x_base.type == 'normal','proba']
        tmp = pd.concat([pd.Series({'(Missing)':self.stats['missing']}),tmp,other])

        fig,ax=plt.subplots(figsize=[10,6.6])
        sns.barplot(tmp.values*100,tmp.index,orient = 'h',ax=ax)
        ax.set_title(title)
        ax.set_xlabel('pct %')


        # 绘制 woe
        item_rare = self.dist['itemlist_rare']
        if item_rare:
            woe_rare = np.log2(self.cross_proba['p(x|y=0)'][item_rare].sum()/self.cross_proba['p(x|y=1)'][item_rare].sum())
        else:
            woe_rare = np.nan 
        woe_rare = pd.Series({'Other values ({})'.format(len(x_base[x_base['type'] == 'rare']))
                           :woe_rare})

        if self.stats['missing']>0 and self.cross_proba['cross_missing']['p(missing|y=1)']>0:
            woe_missing = np.log2(self.cross_proba['cross_missing']['p(missing|y=0)']/self.cross_proba['cross_missing']['p(missing|y=1)'])
        else:
            woe_missing = np.nan
        itemlist_normal = [item for item in self.dist['itemlist'] if item not in item_rare]
        tmp2 = self.cross_proba['woe(x)'][itemlist_normal]
        tmp2 = pd.concat([pd.Series({'(Missing)':woe_missing}),tmp2,woe_rare])

        ax1 = ax.twiny()
        ax1.plot(tmp2.values,tmp2.index,'.:',color='red',label='woe(x)')
        ax1.legend()

        fig.show()
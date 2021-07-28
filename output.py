# -*- coding: utf-8 -*-
from v2x.config.config import LOGS_DIR
from v2x.utils.graphic_utils import Graphics
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from scipy.stats import t as tDistr
import re,sys,os
import warnings
warnings.filterwarnings("ignore")

def _changeColumnName(data,oldName,newName):
    cols = list(data.columns)
    if oldName in cols:
        cols[cols.index(oldName)] = newName
        data.columns = cols
    return data

def outputBackOffCharts(subdir=['25-25-60-rebid=1'],minStaging=1,
                        outputGraph=True):
    name='finishedBids'
    target = 'bidPrice'
    folder = subdir[0]
    capa = int(folder[0:re.search('-',folder).span()[0]]) * 2
    rebid = folder[-7:]
    data = pd.DataFrame()
    for filepath in subdir:
        path = os.path.join(LOGS_DIR,filepath)
        graph = Graphics(path)
        data0 = graph._collectData(name)
        if len(data)==0:
            data = data0
        else:
            data = pd.concat([data,data0],axis=0)
     
    if len(data)==0:
        return None
    
    col = ['step','totalActiveBids','sortIdx','batchCreate','createTime']
    for c in col:
        try:
            data[c] = data[c].astype(int)
        except:
            continue
    col = [target,'bidPayment','carBudget','bidValue']
    for c in col:
        try:
            data[c] = data[c].astype(float)
        except: 
            continue
    
    nrSites = '2sites'
    df = data.loc[data['nrSites']==nrSites]
    df['vehicle'] = df['bidId'].apply(lambda x: 
                                    x[0:re.search('_',x).span()[0]])
    maxStep = max(df['step'])
    df1 = df[df['step']>maxStep-5000]
        
    # average staging time
    df1['backoff time'] = df1.apply(lambda row: 
                        row['createTime']-row['batchCreate'],axis=1)
    df1 = df1[df1['backoff time']> minStaging] 
    
    try:
        df1['admitted'] = np.where(df1['decision']=='admitted',
           'admitted','failed')
    except: 
        df1['admitted'] = np.where(df1['success']==1,'admitted','failed')
    ttl = df1.groupby(['vehicle','bidQoS','carBudget'],as_index=False)\
        .agg({'bidId':'size'}).rename(columns={'bidId':'total_byQos'})
    df1 = pd.merge(df1,ttl,on=['vehicle','bidQoS','carBudget'],copy=False)
    
    df2 = df1[['vehicle','trained','bidQoS','bidId','bidPrice',
              'backoff time','total_byQos','carBudget']].groupby(
              ['vehicle','trained','bidQoS','carBudget'],
              as_index=False).agg({'bidId':'size','bidPrice':'mean',
              'backoff time':'mean','total_byQos':'mean'})
    df2 = _changeColumnName(df2,'bidQoS','deadline')
    df2['deadline'] = np.where(df2['deadline']=='low quality',
                                   'long deadline','short deadline')
    df2 = _changeColumnName(df2,'carBudget','budget')
    df2['budget'] = np.where(df2['budget']==min(df2['budget']),
                                   'low budget','high budget')
    df2['resource capacity'] = capa
    df2.sort_values(by=['deadline','budget','trained','bidPrice'],
                    ascending=[True,True,True,True],inplace=True)

    df2_1 = df2[df2['trained']==graph.learned]
    if outputGraph:
        graph._drawBoxplot(df=df2_1,x='deadline',y='backoff time',
            title='backoff_'+'capa='+str(capa)+'_'+rebid+'_budget',
            hue='budget',legends=1,ylabel=None,
            legendFontsize=8,figsize=None,
            myPalette = {'low budget':'C1','high budget':'C0'})

    df2_1['price range'] = np.where(
            df2_1['bidPrice']>np.mean(df2_1['bidPrice']),
            'high price','low price')
    if outputGraph:    
        graph._drawBoxplot(df=df2_1,x='deadline',y='backoff time',
            title='backoff_'+'capa='+str(capa)+'_'+rebid+'_priceRange',
            hue='price range',legends=None,ylabel=None,
            legendFontsize=12,figsize=None,
            myPalette = {'low price':'C1','high price':'C0'})
        
    return df2_1

def getServiceLevelData(filename='servicelevel_rebid_1.csv',
                        stepRange=(300,7999),decimals=2):
    filepath = os.path.join(LOGS_DIR,filename)
    oldname = 'max. rebid'
    newname = 'MP'
    try:
        servicedata = pd.read_csv(filepath,sep=';')
#        servicedata = _changeColumnName(servicedata,oldname,newname)
#        servicedata['type'] = servicedata.apply(lambda row:
#                row['algorithm type']+','+newname+'='+str(row['MP']),axis=1)
        return servicedata
    except:
        pass
    folders = []
    rebids = ['1','5']
    for i in np.arange(25,120,step=5):
        for rebid in rebids:
            folders.append(
                    '-'.join([str(int(i)),str(int(i)),'60','rebid='])+rebid)

    servicedata = pd.DataFrame()
    for folder in folders:
        path = os.path.join(LOGS_DIR,folder)
        graph = Graphics(path)
        idx = ['_'] + [str(int(x)) for x in list(range(1,10))]
        for j in idx:
            name = 'performance' + j
            df0 = graph._getPerformanceData(name=name,stepRange=stepRange,
                                target='totalSuccessRatio',sites=['2sites'])        
            if df0 is None:
                continue
            values = graph._outputFailureComparison(data=df0,textOutput=False,
                            graphicOutput=False,stepRange=stepRange,legends=2)
            df = pd.DataFrame(columns=['resource capacity','standard','cloud',
                                 'success rate','algorithm type',newname,
                                 'type','selected','version'])
            listOfValues = [np.round(x,decimals=decimals) 
                                for x in list(values.iloc[0])]
            df['success rate'] = listOfValues
            df['algorithm type'] = ['RIAL','DRACO']
    
            df[newname] = int(folder[-1])
            df['type'] = df['algorithm type'].apply(
                                    lambda x:x+','+newname+'='+folder[-1])
            capa = int(folder[0:re.search('-',folder).span()[0]])
            df['resource capacity'] = capa * 2
            df['standard'] = capa
            df['cloud'] = capa
            df['selected'] = 'Y'
            df['version'] = 'v2'
            df = df.loc[~df['success rate'].isnull()]
            
            if len(servicedata)==0:
                servicedata = df
            else:
                servicedata = pd.concat([servicedata,df],axis=0)
    
    servicedata.to_csv(filepath,sep=';',index=None)
    return servicedata
    

def outputServiceLevel(filename='servicelevel_rebid.csv',
                       separate=None,size=(8,4),ci=None,vertical=True,
                       legendFontsize=None,tickFontsize=None):
    graph = Graphics(LOGS_DIR)
    data = pd.read_csv(os.path.join(LOGS_DIR,filename),sep=';')
    data = data[data['selected']=='Y']
    oldname = 'max. rebid'
    newname = 'MP'    
    data = _changeColumnName(data,oldname,newname)
    
    if separate is None:
        hue = 'type'
        data = data[['resource capacity','success rate',hue]]
    else:
        hue = 'type'
        if separate==oldname:
            separate = newname
        data = data[['resource capacity','success rate',separate,hue]]
    
    if ci is None:
        data = data.groupby(['resource capacity',separate,hue],
                            as_index=False).mean().round(3)
    
    graph._drawLineplot(data,x='resource capacity',y='success rate',
          title='Success rate with different resource capacity',
          hue=hue,style=hue,order='sorted',legends=4,
          legendFontsize=legendFontsize,tickFontsize=tickFontsize,
          size=size,separate=separate,ci=ci,
          vertical=vertical)#,showTable=True)

def outputUtilizationOverTimeLowResCapa(path=None,nrSites='2sites',
                                site='site0',service='service2',
                                legends=None,stepRange=None):
    if path is None:
        path = os.path.join(LOGS_DIR,'logs_multipleService',nrSites)
    graph = Graphics(path)
    data = graph._collectData('performance')    
    data = data.loc[data['nrSites']==nrSites]
    if stepRange is None:
        stepRange = (min(data['step']),max(data['step']))
    data = data.loc[(data['step']<=stepRange[1]) 
                        & (data['step']>=stepRange[0])]

    data['trained'] = np.where(data['trained']==graph.random,graph.random,
                                                                graph.learned)
    data['modelType'] = np.where(data['modelType']=='ConvCritic_ConvActor',
                                 'CNN-HW','MLP')        
    data['modelType'] = np.where(data['trained']==graph.random,
                                 '',data['modelType'])
    data = data[data['modelType']!='MLP']

    pattern = re.compile('cloud|standard|slow')
    sitetype = pattern.findall(data.sitetype.iloc[0])
    sitetype = dict([('site'+str(i),x) 
                        for i,x in enumerate(sitetype)])
    
    utilized = graph._parseColumn(data,'utilization',sitetype)
    df = pd.concat([data,utilized],axis=1)
    analyzeCols = [x for x in utilized.columns if site in x and service in x]
    
    for t in analyzeCols:
        tmp = df[['step','trained',t]].groupby(
                        ['step','trained'],as_index=False).mean()
        tmp.columns = ['step','trained','utilization']
        graph._drawLineplot(df=tmp,x='step',y='utilization',
            title=t+'_'+nrSites, style='trained',hue='trained',
            hue_order=[graph.learned,graph.random],legends=legends,
            tickFontsize=12)

def outputNrRebidData(subfolder,resource,name):
    path = os.path.join(LOGS_DIR,subfolder)
    graph = Graphics(path)
    
    data = graph._collectData(name)
    if data.shape[0]<=1:
        return
    
    col = ['step','batchCreate','createTime','finishTime','nrRebid']
    for c in col:
        data[c] = data[c].astype(int)
    
    col = ['bidPrice','bidPayment','carBudget']
    for c in col:
        data[c] = data[c].astype(float)            
    
    df = data.copy()
    df['vehicle'] = df['bidId'].apply(lambda x: 
                                    x[0:re.search('_',x).span()[0]])
    
    maxStep = max(df['step'])
    df1 = df[df['step']>maxStep-5000]
    # average staging time
    df1['backoff time'] = df1.apply(lambda row: 
                        row['createTime']-row['batchCreate'],axis=1)
    
    # correlation admission rate vs. price vs. staging time, by qos
    df1['admitted'] = np.where(df1['status']=='finished',True,False)
    df1 = _changeColumnName(df1,'nrSites','max rebid')
    df1['max rebid'] = df1['max rebid'].apply(lambda x: int(x[0]))
    df1['trained'] = np.where(df1['trained']==graph.random,graph.random,
                               df1['trained'])
    
    ttl = df1.groupby(['max rebid','trained','vehicle','bidQoS','carBudget'],
                      as_index=False)\
        .agg({'bidId':'size'}).rename(columns={'bidId':'total_byQos'})
    df1 = pd.merge(df1,ttl,on=['max rebid','trained','vehicle','bidQoS',
                               'carBudget'],copy=False)
    df2 = df1[['max rebid','trained','vehicle','admitted','bidQoS',
               'bidId','bidPrice','backoff time','total_byQos',
               'carBudget','nrRebid']].groupby(['max rebid','trained',
               'vehicle','admitted','bidQoS','carBudget'],
               as_index=False).agg({'bidId':'size','bidPrice':'mean',
               'backoff time':'mean','total_byQos':'mean','nrRebid':'mean'})
    df2.sort_values('trained',ascending=False,inplace=True)
    df2['resource capacity'] = resource
    return df2

def outputNrRebidComparisonBoxplot(subfolder='25-25-60'):
    name = 'finishedBids'
    resource = int(subfolder[0:2]) * 2
    df2 = outputNrRebidData(subfolder,resource,name)
    title = (name+'_distr-nrRebid_all_byAlgorithm')
    graph._drawBoxplot(df=df2,x='max rebid',y='nrRebid',title=title,
            hue='trained',ylabel='rebidding overhead',legends=2,
            legendFontsize=8,figsize=(2,4))


def _drawHistLine(path,data_learned,data_random,note,
                  col=None,xlim=(0,1),loc=4):
    learned = 'DRACO'
    random = 'RIAL'
    if col is None:
        vehicleSuccess = data_learned.groupby('vehicle').agg(
                {'bidId':'size','success':'sum'})
        tmp = data_random.groupby('vehicle').agg(
                {'bidId':'size','success':'sum'})
    else:
        vehicleSuccess = data_learned.groupby(['vehicle',col],
                        as_index=False).agg({'bidId':'size','success':'sum'})
        tmp = data_random.groupby(['vehicle',col],as_index=False).agg(
                {'bidId':'size','success':'sum'})
    vehicleSuccess['success rate'] = (vehicleSuccess['success'] 
                                        / vehicleSuccess['bidId'])
    vehicleSuccess['trained'] = learned
    
    tmp['success rate'] = tmp['success'] / tmp['bidId']
    tmp['trained'] = random
    
    vehicleSuccess = pd.concat([vehicleSuccess,tmp])
    style = {learned:'-',random:'--'}
    graph = Graphics(path)
    graph._drawCdfFromKde(df=vehicleSuccess,hue='trained',
        target='success rate',style=style,
        title='allocation_'+note+'_vehicleSuccessRateCdf',
        xlim=xlim,col=col,loc=loc)

def outputIndividualSuccessRateWithHighResCapa(path=None):
    if path is None:
        path = os.path.join(LOGS_DIR,'35-35-60-rebid=1')
    graph = Graphics(path)
    name = 'finishedBids'
    data = graph._collectData(name)
    data.modelType = 'CNN-HW'
    data = data[data['step']>7000]
    col = ['step','success','batchCreate','createTime','finishTime','nrRebid']
    for c in col:
        try:
            data[c] = data[c].astype(int)
        except:
            pass
    target = 'bidPrice'
    col = [target,'bidPayment','carBudget']
    for c in col:
        data[c] = data[c].astype(float)
    data['vehicle'] = data['bidId'].apply(
            lambda x:x[0:re.search('_',x).span()[0]])
    data['carBudget'] = np.where(data['carBudget']<2000,'low','high')
    data = _changeColumnName(data,'carBudget','budget')
    
    data_learned = data[data.trained==graph.learned]
    data_random = data[data.trained==graph.random]
    _drawHistLine(path,data_learned,data_random,
                  'highResCapa_budgets',col='budget',xlim=(0.6,1),loc=2)

def _outputUtilizationMeanAndStdHighResCapa_perCapa(path=None,capa=None):
    if path is None:
        path = os.path.join(LOGS_DIR,'35-35-60-rebid=1')
        capa = 35
    graph = Graphics(path)
    name = 'performance'
    data = graph._collectData(name)
    if len(data)==0:
        return None,None
    data['trained'] = np.where(data['trained']==graph.random,
                                graph.random,data['trained'])
    pattern = re.compile('cloud|standard|slow')
    sitetype = pattern.findall(data.sitetype.iloc[0])
    sitetype = dict([('site'+str(i),x) 
                        for i,x in enumerate(sitetype)])
    
    utilized = graph._parseColumn(data,'utilization',sitetype)
    occupied = graph._parseColumn(data,'occupied',sitetype)
    maxAmount = graph._parseColumn(data,'maxAmount',sitetype)
    
    data = pd.concat([data,utilized,occupied,maxAmount],axis=1)
    df0 = data[data.step>5000]
    df0.fillna(0,inplace=True)
    
    targets = []
    for s in sitetype.keys():
        targets.append(s)
        occupiedColn = [x for x in df0.columns if 'occupied' in x 
                        and s in x and 'resource' not in x]
        maxAmountColn = [x for x in df0.columns if 'maxAmount' in x 
                         and s in x and 'resource' not in x]
        df0[s] = df0.apply(lambda row: 
            sum(row[occupiedColn])/sum(row[maxAmountColn]),axis=1)
        
    colname = 'allsites'
    targets.append(colname)
    occupiedColn = list(occupied.columns)
    maxAmountColn = [x for x in maxAmount.columns if 'resource' not in x]
    df0[colname] = df0.apply(lambda row: 
        sum(row[occupiedColn])/sum(row[maxAmountColn]),axis=1)
    
    def myFunc(data):
        results = []
        colnames = []
        for c in targets:
            results += [np.mean(data[c]),np.std(data[c])]
            colnames += [c+'_mean', c+'_std']
        return pd.Series(data=results,index=colnames)
    tmp = df0[['trained']+targets].groupby('trained').apply(myFunc)
    tmp.sort_index(inplace=True,ascending=False)
    tmp = tmp.transpose()
    tmp.reset_index(inplace=True)
    tmp.columns = ['site',graph.learned,graph.random]
    tmp['rebid'] = int(path[-1])
    tmp['resource'] = capa
    df0['rebid'] = int(path[-1])
    df0['resource'] = capa * 2
    return df0,tmp

def outputUtilizationMeanAndStdHighResCapa(boxchart=None,path=None,
                                           rebid='5',target='site0'):
    if boxchart is None:
        folders = []
        for i in np.arange(25,120,step=5):
            folders.append('-'.join([str(int(i)),str(int(i)),'60','rebid='])+rebid)
        boxchart = pd.DataFrame()
        for folder in folders:
            path = os.path.join(LOGS_DIR,folder)
            capa = int(folder[0:re.search('-',folder).span()[0]])
            df0,tmp = _outputUtilizationMeanAndStdHighResCapa_perCapa(path,capa)
            if df0 is None:
                continue
            if len(boxchart)==0:
                boxchart = df0
            else:
                boxchart = pd.concat([boxchart,df0],axis=0)
        boxchart = _changeColumnName(boxchart,'resource','resource capacity')

    graph = Graphics(path)
    graph._drawBoxplot(df=boxchart,x='resource capacity',
           y=target,ylabel='utilization',hue='trained',
           title='utilization_boxplot_'+target+'_rebid='+rebid,
           legendFontsize=12,figsize=(10,4),legends=1)
    return boxchart

def outputRebidBoxplotMeanAndStd(boxchart=None,path=None,rebid='5'): 
    if boxchart is None:
        name = 'finishedBids'
        folders = []
        for i in np.arange(25,120,step=5):
            folders.append('-'.join([str(int(i)),str(int(i)),'60','rebid='])+rebid)
        
        boxchart = pd.DataFrame()
        for folder in folders:
            path = os.path.join(LOGS_DIR,folder)
            resource = int(folder[0:re.search('-',folder).span()[0]]) * 2
            df0 = outputNrRebidData(folder,resource,name)
            if df0 is None:
                continue
            if len(boxchart)==0:
                boxchart = df0
            else:
                boxchart = pd.concat([boxchart,df0],axis=0)
    
    graph = Graphics(path)
    graph._drawBoxplot(df=boxchart,x='resource capacity',y='nrRebid',
                       ylabel='rebidding overhead',hue='trained',
                       title='rebid='+rebid+'_boxplot',
                       legendFontsize=8,figsize=(10,4))
    boxchart['max rebid'] = rebid
    return boxchart

def getInterval(tbl,targetCol,rebidCol,algorithmCol,
                capaCol='resource capacity'):
    interval = pd.DataFrame()
    tbl[rebidCol] = tbl[rebidCol].apply(str)
    for rebid in ['1','5']:
        for capa in set(tbl[capaCol]):
            x1 = list(tbl[(tbl[rebidCol]==rebid)
                        & (tbl[capaCol]==capa) 
                        & (tbl[algorithmCol]=='DRACO')][targetCol])
            x2 = list(tbl[(tbl[rebidCol]==rebid)
                        & (tbl[capaCol]==capa) 
                        & (tbl[algorithmCol]=='RIAL')][targetCol])
            if len(x1)==0:
                continue
            meanDiff,interval_ttest,interval_welch = welch_ttest(x1,x2)
            tmp = pd.DataFrame(
                [[rebid,capa,meanDiff,interval_ttest,interval_welch]],
                columns=['max rebid','resource capacity','mean difference',
                     'confidence interval ttest','confidence interval welch'])
            if len(interval)==0:
                interval = tmp
            else:
                interval = pd.concat([interval,tmp],axis=0)
    interval.sort_values(by=['max rebid','resource capacity'],inplace=True)
    return interval

def outputComparisonTbl(tbl,targetCol,rebidCol='MP',
                        capaCol='resource capacity',algorithmCol='trained'):
    tbl_1 = tbl[[algorithmCol,capaCol,targetCol]].groupby(
            [algorithmCol,capaCol],as_index=False).mean()
    tbl_2 = tbl_1.pivot_table(values=targetCol,index=capaCol,
                              columns=algorithmCol)
    tbl_2['difference'] = (tbl_2['RIAL'] - tbl_2['DRACO']) / tbl_2['RIAL']
    interval = getInterval(tbl,targetCol=targetCol,rebidCol=rebidCol,
                           algorithmCol=algorithmCol,capaCol=capaCol)
    return tbl_2, interval

def findOffloadRate(tbl,target,rebid=1,minRate=0.98,rebidCol='MP'):
    try:
        return min(tbl.loc[(tbl[rebidCol]==rebid) 
                           & (tbl[target]>=minRate),'resource capacity'])
    except:
        return None

def outputServiceCmpTbl(servicedata):
    newname = 'MP'
    serviceComp = servicedata[['algorithm type',newname,
                    'resource capacity','success rate']].pivot_table(
                    index=[newname,'resource capacity'],
                    columns='algorithm type',values='success rate')
    serviceComp['difference'] = (serviceComp['RIAL']
                                 - serviceComp['DRACO']) / serviceComp['RIAL']
    for c in serviceComp.columns:
        serviceComp[c] = serviceComp[c].apply(lambda x: np.round(x,2))
    serviceComp.reset_index(inplace=True)
    offloadRate = {}
    for rebid in [1,5]:
        for minRate in [0.98,0.99]:
            for target in ['DRACO','RIAL']:
                offloadRate[(rebid,minRate,target)] = findOffloadRate(
                        serviceComp,target=target,rebid=rebid,minRate=minRate)
    
    interval = getInterval(servicedata,targetCol='success rate',
                           rebidCol=newname,algorithmCol='algorithm type')
    return serviceComp, offloadRate, interval

def outputBackoffChartsComparison(rebid='1',contention=None,figsize=(10,4)):
    if contention=='high':
        folders = []
        for i in np.arange(25,50,step=5):
            folders.append('-'.join([str(int(i)),str(int(i)),
                                     '60','rebid='])+rebid)
    elif contention=='low':
        folders = []
        for i in np.arange(55,120,step=5):
            folders.append('-'.join([str(int(i)),str(int(i)),
                                     '60','rebid='])+rebid)        
    else:
        contention = 'all'
        folders = []
        for i in np.arange(25,120,step=5):
            folders.append('-'.join([str(int(i)),str(int(i)),
                                     '60','rebid='])+rebid)

    backoffBudget = pd.DataFrame()
    for folder in folders:
        backoffData = outputBackOffCharts(subdir=[folder],outputGraph=False)
        if len(backoffBudget)==0:
            backoffBudget = backoffData
        else:
            backoffBudget = pd.concat([backoffBudget,backoffData],axis=0)
    
    for dl in ['long deadline','short deadline']:
        tmp = backoffBudget.loc[backoffBudget['deadline']==dl]
        graph = Graphics(LOGS_DIR)
        graph._drawBoxplot(df=tmp,x='resource capacity',y='backoff time',
           ylabel='backoff time',hue='price range',
           title='backoffBudget_'+contention+'Contention_'+dl +'_rebid='+rebid,
           legendFontsize=14,figsize=figsize,legends=1,
           myPalette={'low price':'C1','high price':'C0'}) 
        
    return backoffBudget

def welch_ttest(x1,x2,ci=0.95,tail='one'):
    if tail=='two':
        ci = 1 - (1-ci)/2    
    n1 = len(x1)
    n2 = len(x2)
    mu1 = np.mean(x1)
    mu2 = np.mean(x2)
    dof1 = n1-1
    dof2 = n2-1
    var1 = np.var(x1,ddof=1)
    var2 = np.var(x2,ddof=1)
    pooled_samplevar = (dof1 * var1 + dof2 * var2) / (dof1 + dof2)
    pooled_sd = np.sqrt(pooled_samplevar)
    t1 = tDistr.ppf(ci,dof1+dof2)
    interval_ttest = t1 * pooled_sd * np.sqrt(1/n1 + 1/n2)
    
    welch_dof = (var1/n1 + var2/n2)**2 / ( 
                            (var1/n1)**2 / dof1 + (var2/n2)**2 / dof2 )
    t2 = tDistr.ppf(ci,welch_dof)
    interval_welch = t2 * np.sqrt(var1/n1 + var2/n2)
    meanDiff = mu1 - mu2
    return meanDiff,interval_ttest,interval_welch

def _collectRewardData(folder,filename,columnName,dataRange):
    filepath = os.path.join(LOGS_DIR,folder,filename)
    try:
        data = pd.read_csv(filepath,sep=';')
    except:
        return
    data1 = data[['step','avgReward']].groupby(
            'step',as_index=False).mean()
    data1 = data1[(data1['step']<=dataRange[1]) 
                   & (data1['step']>dataRange[0])]
    data1.columns = ['step',columnName]
    return data1

def outputFspReward(filepath=LOGS_DIR,folder='30-30-60-rebid=1',
                    dataRange=(100,1500)):
    dataSL = pd.DataFrame()
    for i in np.arange(0,10):
        filename = 'supervisedLearningModel'+str(i)+'_2_True.txt'
        data_sl = _collectRewardData(folder=folder,filename=filename,
                            columnName='average reward_sl',
                            dataRange=dataRange)
        if data_sl is None:
            continue
        if len(dataSL)==0:
            dataSL = data_sl
        else:
            dataSL = pd.concat([dataSL,data_sl],axis=0)

    dataRL = pd.DataFrame()
    for i in np.arange(0,10):
        filename = 'priceLearningModel'+str(i)+'_2_True.txt'
        data_rl = _collectRewardData(folder=folder,filename=filename,
                            columnName='average reward_rl',
                            dataRange=dataRange)
        if data_rl is None:
            continue
        if len(dataRL)==0:
            dataRL = data_rl
        else:
            dataRL = pd.concat([dataRL,data_rl],axis=0)
    
    data = dataSL.merge(dataRL,on='step',how='outer')
    data.fillna(0,inplace=True)
    data['average reward'] = (
            (1-data['average reward_sl']) * (1-50/(data['step']+50)) 
            + 50/(data['step']+50) * data['average reward_rl'])
    avgStep = 10
    data['step'] = data['step'].apply(lambda x: int(x/avgStep)*avgStep)
    
    graph = Graphics(filepath)
    graph._drawLineplot(df=data,x='step',y='average reward',
                        title='avgReward_'+folder,legends=None,
                        legendFontsize=None,tickFontsize=12,
                        size=None,decimals=2,ci='sd',showTable=False)

def addRows(tbl,dataRange,valueCol,dataCol,filler):
    exist = set(tbl[dataCol])
    diff = list(set(dataRange)-exist)
    exist = list(exist)
    exist.sort()
    for i in diff:
        oneSmaller = exist[0]
        oneBigger = i
        for j in exist:
            if j < i and j > oneSmaller:
                oneSmaller = j
            if j > i:
                oneBigger = j
                break
        resBefore = tbl[tbl[dataCol]==oneSmaller][valueCol].tolist()[-1]
        resAfter = tbl[tbl[dataCol]==oneBigger][valueCol].tolist()[0]
        newRes = int(np.round((resAfter - resBefore) * (i - oneSmaller) 
                        / (oneBigger - oneSmaller) + resBefore,decimals=0))
        tmpPd = pd.DataFrame([filler + [newRes,i]],columns=tbl.columns)
        tbl = pd.concat([tbl,tmpPd],axis=0)
    return tbl

def outputServiceTable(serviceComp,size=None,
                       legendFontsize=None,tickFontsize=None):
    newname = 'MP'
    dra = serviceComp[[newname, 'resource capacity', 'DRACO']]
    ria = serviceComp[[newname, 'resource capacity', 'RIAL']]
    perfRange = np.arange(0.90,0.985,step=0.01)
    perfRange = [np.round(x,decimals=2) for x in perfRange]
    draco = pd.DataFrame()
    rial = pd.DataFrame()
    for rebid in [1,5]:
        tmp = dra[dra[newname]==rebid]
        tmp = addRows(tbl=tmp,dataRange=perfRange,
                      valueCol='resource capacity',dataCol='DRACO',
                      filler=[rebid])
        if len(draco)==0:
            draco = tmp
        else: 
            draco = pd.concat([draco,tmp],axis=0)
        
        tmp0 = ria[ria[newname]==rebid]
        tmp0 = addRows(tbl=tmp0,dataRange=perfRange,
                       valueCol='resource capacity',dataCol='RIAL',
                       filler=[rebid])
        if len(rial)==0:
            rial = tmp0
        else:
            rial = pd.concat([rial,tmp0],axis=0)
    
    col = [newname,'resource capacity','success rate','type'] 
    draco['type'] = 'DRACO'
    draco.columns = col
    rial['type'] = 'RIAL'
    rial.columns = col
    data = pd.concat([draco,rial],axis=0)
    data = data[(data['success rate']>=perfRange[0])]
    data1 = data.groupby([newname,'success rate','type'],
                                                as_index=False).agg('min')
    graph = Graphics(LOGS_DIR)
    graph.drawRegplot(df=data1,x='success rate',y='resource capacity',
        title='capacityUse',hue='type',order='fixed',size=size,
        legendFontsize=legendFontsize,tickFontsize=tickFontsize,
        separate=newname,x_decimals=2,y_decimals=0,
        dataRange=perfRange+[0.99])    
    
    data = data.pivot_table(values='resource capacity',
                index=[newname,'success rate'],
                columns='type',aggfunc='min')
    data.reset_index(inplace=True)
    data['saving'] = (data['RIAL'] - data['DRACO']) / data['RIAL']
    data['saving'] = data['saving'].apply(
            lambda x: np.round(x,decimals=2) if not np.isnan(x) else '')
    for col in ['DRACO','RIAL']:
        data[col] = data[col].apply(
                lambda x: int(x) if not np.isnan(x) else '')
    return data,data1


#%%
if __name__ == '__main__': 
    path = LOGS_DIR
    try:
        path = os.path.join(path,sys.argv[1])
    except:
        pass
    
    folder = '30-30-60-rebid=1'
    graph = Graphics(os.path.join(path,folder))
    graph.drawPerformance(drawPerformanceOnly=True,textOutput=False,
                          sites='2sites',target='totalSuccessRatio',
                          legends=4,stepRange=(300,7999),
                          decimals=2,ci='sd')
    outputUtilizationOverTimeLowResCapa(
                          path=os.path.join(path,folder),
                          nrSites='2sites',site='site0',service='service2',
                          stepRange=(7700,7999))
    outputFspReward(filepath=path,folder=folder,dataRange=(150,7999))
    
    decimals = 3
    filename = 'servicelevel_rebid_decimals='+str(decimals)+'.csv'
    servicedata = getServiceLevelData(filename=filename,decimals=decimals)
#    outputServiceLevel(filename=filename,separate='MP',ci='sd',
#                       size=(20,4),vertical=False)
    size = (9, 4)
    legendFontsize = 12
    tickFontsize = 12
    outputServiceLevel(filename=filename,separate='MP',ci='sd',
                       size=size,legendFontsize=legendFontsize,
                       tickFontsize=tickFontsize,vertical=True)
    serviceComp,offloadRate,serviceCi = outputServiceCmpTbl(servicedata)
    serviceTbl,tbl = outputServiceTable(serviceComp,size=size,
                    legendFontsize=legendFontsize,tickFontsize=tickFontsize)

    try:
        rebidTbl5 = pd.read_csv(os.path.join(path,'rebid=5_Tbl.csv'),sep=';')
        rebidTbl5 = outputRebidBoxplotMeanAndStd(
                                    boxchart=rebidTbl5,path=None,rebid='5')
    except:
        rebidTbl5 = outputRebidBoxplotMeanAndStd(path=None,rebid='5')
        rebidTbl5.to_csv(os.path.join(path,'rebid=5_Tbl.csv'),
                                                     sep=';',index=None)
    rebidComp, rebidCi = outputComparisonTbl(tbl=rebidTbl5,
                    targetCol='nrRebid',rebidCol='max rebid',
                    capaCol='resource capacity',algorithmCol='trained')
 
    try:
        utilTbl1 = pd.read_csv(os.path.join(path,'rebid=1_utilTbl.csv'),
                               sep=';')
        utilTbl1 = outputUtilizationMeanAndStdHighResCapa(
                                                boxchart=utilTbl1,rebid='1')
    except:
        utilTbl1 = outputUtilizationMeanAndStdHighResCapa(rebid='1')
        utilTbl1.to_csv(os.path.join(path,'rebid=1_utilTbl.csv'),
                        sep=';',index=None)
    
    utilComparison, utilCi = outputComparisonTbl(tbl=utilTbl1,
        targetCol='site0',rebidCol='rebid',capaCol='resource capacity',
        algorithmCol='trained')
    
    outputIndividualSuccessRateWithHighResCapa()

    _ = outputBackOffCharts(subdir=['25-25-60-rebid=5'])
    backoffBudget5 = outputBackoffChartsComparison(rebid='5',contention='high',
                                                   figsize=(5,4))




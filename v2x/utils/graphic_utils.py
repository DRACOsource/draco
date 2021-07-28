# -*- coding: utf-8 -*-
from ..config.config import (GRAPH_DIR, MDL_PARAMS as mp, 
                             RESOURCE_SITE_PARAMS as rsp)
import glob,os,re,ast
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from itertools import product
import matplotlib.pyplot as plt
plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15
plt.rc('pdf',fonttype=42)
plt.ioff()
import seaborn as sns
from scipy.stats import gaussian_kde

class Graphics:
    learned = 'DRACO'
    random = 'RIAL'
    
    def __init__(self,path):
        self.path = path
    
    def _collectData(self,name):
        name = name + '*'
        filename = os.path.join(self.path,name)
        perfiles = glob.glob(filename)
        data = pd.DataFrame()
        data_part = pd.DataFrame()
        for f in perfiles:
            try:
                data_part = pd.read_csv(f,sep=';')
            except:
                continue
            locations = [x.span() for x in re.finditer('_',f)]
            try:
                nrSites = str(f[locations[-2][1]:locations[-1][0]]) + 'sites'
                fileversion = str(f[locations[-2][0]-1:locations[-2][0]])
            except:
                continue
            trained = (self.learned 
                       if f[locations[-1][1]:-4]=='True' else self.random)
            data_part['nrSites'] = nrSites
            data_part['trained'] = trained
            
            if 'bidId' in data_part.columns:
                data_part['bidId'] = data_part['bidId'].apply(
                                                lambda x: fileversion+x)
            
            if data.shape[0]==0:
                data = data_part
            else:
                data = pd.concat([data,data_part],axis=0)
        
        if not 'modelType' in data.columns:
            data['modelType'] = 'MLP'
        return data
    
    def _drawLineplot(self,df,x,y,title,style=None,hue=None,order='flex',
                      hue_order=None,legends=2,legendFontsize=None,
                      tickFontsize=None,size=None,separate=None,
                      decimals=1,ci=None,showTable=False,vertical=True):
        defaultFontsize = 16
        if tickFontsize is None:
            tickFontsize = defaultFontsize
        if legendFontsize is None:
            legendFontsize = defaultFontsize
        if size is None:
            length = 5
            height = 4
        else:
            length = size[0]
            height = size[1]

        if separate is None:
            if hue is not None:
                if hue_order is None:
                    if order=='flex':
                        tmp = df.groupby(hue)
                        tmp = (tmp.tail(1).drop_duplicates()
                               .sort_values(y,ascending=False))
                    else:
                        tmp = df.groupby(
                                hue,as_index=False).max().sort_values(hue)
                    hue_order = tmp[hue].tolist()
            else:
                hue_order = None
            
            fig,ax = plt.subplots()
            fig.set_size_inches(length,height)
            ax = sns.lineplot(x=x,y=y,style=style,hue=hue,data=df,ci=ci,
                              hue_order=hue_order,style_order=hue_order)
            ax.set_xlabel(xlabel=x,fontsize=tickFontsize)
            ax.set_ylabel(ylabel=y,fontsize=tickFontsize)
            ax.set_xticklabels(np.int0(ax.get_xticks()),size=tickFontsize)
            ax.set_yticklabels(np.round(ax.get_yticks(),decimals=decimals),
                                   size=tickFontsize)
            if legends is not None:
                handles, labels = ax.get_legend_handles_labels() 
                l = ax.legend(handles[1:],labels[1:],loc=legends,
                              fontsize=legendFontsize)
                plt.savefig(os.path.join(GRAPH_DIR,title+'.pdf'),
                            bbox_extra_artists=(l,), bbox_inches='tight')
            else:
                l = ax.legend()
                l.remove()
                plt.savefig(os.path.join(GRAPH_DIR,title+'.pdf'),
                            bbox_inches='tight')
            plt.clf()
        else:
            sepCol = list(set(df[separate]))
            if vertical:
                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
            else:
                fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
            fig.set_size_inches(length,height)
            
            df1 = df.loc[df[separate]==sepCol[0]]
            df2 = df.loc[df[separate]==sepCol[1]]
            
            if hue is not None:
                if hue_order is None:
                    if order=='flex':
                        tmp1 = df1.groupby(hue)
                        tmp1 = (tmp1.tail(1).drop_duplicates()
                               .sort_values(y,ascending=False))
                        tmp2 = df2.groupby(hue)
                        tmp2 = (tmp2.tail(1).drop_duplicates()
                               .sort_values(y,ascending=False))
                    else:
                        tmp1 = df1.groupby(
                                hue,as_index=False).max().sort_values(hue)
                        tmp2 = df2.groupby(
                                hue,as_index=False).max().sort_values(hue)
                    hue_order1 = tmp1[hue].tolist()
                    hue_order2 = tmp2[hue].tolist()
            else:
                hue_order1 = None
                hue_order2 = None
            
            g1 = sns.lineplot(x=x,y=y,style=style,hue=hue,data=df1,ci=ci,
                        hue_order=hue_order1,style_order=hue_order1,ax=ax1)      
            g2 = sns.lineplot(x=x,y=y,style=style,hue=hue,data=df2,ci=ci,
                        hue_order=hue_order2,style_order=hue_order2,ax=ax2)
            
            ax1.set_xticklabels(np.int0(ax1.get_xticks()),
                                size=tickFontsize)
            ax2.set_xticklabels(np.int0(ax2.get_xticks()),
                                size=tickFontsize)
            ax1.set_yticklabels(np.round(ax1.get_yticks(),
                                decimals=decimals),size=tickFontsize)
            ax2.set_yticklabels(np.round(ax2.get_yticks(),
                                decimals=decimals),size=tickFontsize)
            ax1.xaxis.label.set_size(tickFontsize)
            ax2.xaxis.label.set_size(tickFontsize)
            ax1.yaxis.label.set_size(tickFontsize)
            ax2.yaxis.label.set_size(tickFontsize)
                
            
            if showTable:
                fig.subplots_adjust(hspace=0.5)
                ax1.xaxis.set_visible(False)
                ax2.set_xticklabels([])
                ax2.xaxis.labelpad = 30
                ax2.tick_params(bottom=False)
                
                values1,values5,alg,column = self._createTbl(servicedata=df,
                            hue=hue,alg=['DRACO','RIAL'],separate=separate)
                ax1.table(cellText=values1,rowLabels=alg,
                                          colLabels=column,loc='bottom')
                ax2.table(cellText=values5,rowLabels=alg,
                                          colLabels=column,loc='bottom')
            
            if legends is not None:
                handles1, labels1 = ax1.get_legend_handles_labels() 
                ax1.legend(handles1[1:],labels1[1:],loc=legends,
                           fontsize=legendFontsize)
                handles2, labels2 = ax2.get_legend_handles_labels() 
                ax2.legend(handles2[1:],labels2[1:],loc=legends,
                           fontsize=legendFontsize)            
            else:
                l = ax.legend()
                l.remove()
            plt.savefig(os.path.join(GRAPH_DIR,title+'.pdf'),
                        bbox_inches='tight')
            plt.clf()
          
    def _drawCdfFromKde(self,df,hue,target,style,title,
                        col=None,xlim=(0,1),loc=4):
        if col is None:
            plt.figure(figsize=(5,4))
            hue_order = list(set(df[hue]))
            hue_order.sort()
            for grp in hue_order:
                tmp = df.loc[df[hue]==grp,target]
                tmp = np.array(tmp)
                kde = gaussian_kde(tmp)
                cdf = np.vectorize(lambda x: kde.integrate_box_1d(-np.inf,x))
                x = np.linspace(xlim[0],xlim[1])
                plt.plot(x,cdf(x),linestyle=style[grp],label=grp)
            plt.legend(loc=loc,fontsize=15)
            plt.ylabel('CDF',fontsize=15)
            plt.xlabel('Success rate',fontsize=15)
            plt.savefig(os.path.join(GRAPH_DIR,title+'.pdf'),
                        bbox_inches='tight')
            plt.clf()
        else:
            x = np.linspace(xlim[0],xlim[1])
            newDf = pd.DataFrame()
            for c in set(df[col]):
                for grp in set(df[hue]):
                    tmp = df.loc[(df[hue]==grp) & (df[col]==c),target]
                    tmp = np.array(tmp)
                    kde = gaussian_kde(tmp)
                    cdf = np.vectorize(
                            lambda y:kde.integrate_box_1d(-np.inf,y))
                    tmp0 = pd.DataFrame(np.vstack([x,cdf(x)]).transpose(),
                                         columns=[target,'CDF'])
                    tmp0[hue] = grp
                    tmp0[col] = c
                    if len(newDf)==0:
                        newDf = tmp0                        
                    else:
                        newDf = pd.concat([newDf,tmp0],axis=0)
            fig,ax = plt.subplots()
            ax = sns.FacetGrid(data=newDf,col=col,)
            ax.fig.set_size_inches(10,4)
            ax.map_dataframe(sns.lineplot,target,'CDF',hue,
                    style=hue,hue_order=list(style.keys()),
                    style_order=list(style.keys()),ci=None)
            ax.set(xlim=xlim)
            for axes in ax.axes.flat:
                axes.set_ylabel('CDF', fontsize=15)
                axes.set_xlabel('Success rate', fontsize=15)
                axes.set_title(axes.get_title(),fontsize=15)
            handles, labels = ax.axes[0][-1].get_legend_handles_labels()
            l = ax.axes[0][-1].legend(handles[1:],labels[1:],
                                           loc=loc,fontsize=15)          
            plt.savefig(os.path.join(GRAPH_DIR,title+'.pdf'),
                        bbox_extra_artists=(l,),bbox_inches='tight')
            plt.clf()
    
    def _drawBoxplot(self,df,x,y,title,hue=None,legends=3,ylabel=None,
                     legendFontsize=None,figsize=None,
                     myPalette=None,hue_order=None):
        if figsize is None:
            figsize = (5,4)
        defaultFontsize = 16
        if legendFontsize is None:
            legendFontsize = defaultFontsize
        if ylabel is None:
            ylabel = y
        
        sns.set_style('white')
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize)
        if myPalette is None:
            myPalette = {self.random:'C1',self.learned:'C0'}
        sns.boxplot(data=df,x=x,y=y,ax=ax,hue=hue,
                    showfliers=False,palette=myPalette,
            showmeans=True,meanprops={'marker':'o','markerfacecolor':'white',
                            'markeredgecolor':'white'},hue_order=hue_order)
        ax.set_xlabel(xlabel=x,fontsize=defaultFontsize)
        ax.set_ylabel(ylabel=ylabel,fontsize=defaultFontsize)
        if legends is not None:
            handles, labels = ax.get_legend_handles_labels()
            l = ax.legend(handles,labels,loc=legends,fontsize=legendFontsize)
            plt.savefig(os.path.join(GRAPH_DIR,title+'.pdf'),
                    bbox_extra_artists=(l,), bbox_inches='tight')
        else:
            l = ax.legend()
            l.remove()
            plt.savefig(os.path.join(GRAPH_DIR,title+'.pdf'),
                        bbox_inches='tight')
        plt.clf()
    
    def _parse(self,value,sitetype):
        a = ast.literal_eval(value)
        values = {}
        for i,x in enumerate(a):
            site = 'site' + str(i)
            stype = sitetype[site]
            for key in x.keys():
                values[stype+'_'+site+'_'+str(key[1])] = x[key]
        return pd.Series(values)
    
    def _parseColumn(self,df,target,sitetype):
        result = df[target].apply(lambda x: self._parse(x,sitetype))
        col = result.columns
        col0 = [target + '_' + x for x in col]
        result.columns = col0
        return result


    def _outputFailureComparison(self,data=None,path=None,prefix='',
                textOutput=False,graphicOutput=True,legends=2,stepRange=None,
                nrSites='2sites'):
        name = 'performance'
        if data is None:
            data = self._collectData(name)
            try:
                data['success rate'] = data['success'] / data['finishedBid']
            except:
                data['success rate'] = data['success'] / data['totalBid']
            try:
                data['failed rate'] = data['rejectedBid'] / data['finishedBid']
            except:
                data['failed rate'] = data['rejectedBid'] / data['totalBid']
            data['trained'] = np.where(data['trained']==self.random,
                                        self.random,self.learned)
            data['category'] = data.apply(lambda row: 
                        row['trained'] + '_' + str(row['nrSites']) 
                        + '_' + str(row['modelType']),axis=1)
            cols = list(data.columns)
            cols[cols.index('totalSuccessRatio')] = 'success rate by time'
            data.columns = cols
        
        barChart = pd.DataFrame()
        coln = ['nrSites','success '+self.learned,
                'success '+self.random,
                'success rate','failed '+self.learned,
                'failed '+self.random,'failed rate']
        barGrp = ['failed '+self.random, 'failed '+self.learned]
        barGrp2 = ['success '+self.random,'success '+self.learned]
        note = 'learned_vs_random'

        learned = data[(data.trained==self.learned) & (data.nrSites==nrSites)]
        random = data[(data.trained==self.random) & (data.nrSites==nrSites)]
        try:
            min_learned = max(learned['step']) - mp.recent
        except:
            min_learned = 0
        try:
            min_random = max(random['step']) - mp.recent
        except:
            min_random = 0
        learned = learned.loc[learned['step']>=min_learned,:]
        random = random.loc[random['step']>=min_random,:]
        
        successLearned = np.mean(learned['success rate'])
        successRandom = np.mean(random['success rate'])
        successRate = (successLearned-successRandom)/successRandom
        failedLearned = np.mean(learned['failed rate'])
        failedRandom = np.mean(random['failed rate'])
        failedRate = -(failedLearned-failedRandom)/failedRandom
        row = pd.DataFrame([[nrSites,successLearned,successRandom,successRate,
                failedLearned,failedRandom,failedRate]],columns=coln)
        if len(barChart)==0:
            barChart = row
        else:
            barChart = pd.concat([barChart,row],axis=0)
        
        if stepRange is None:
            if max(data.step)>4000:
                data = data[data.step>800]
        else:
            data = data.loc[(data['step']<=stepRange[1]) 
                                & (data['step']>=stepRange[0])]
        xGrp = 'nrSites'
        barChart.sort_values(by=xGrp,inplace=True)
        data = data[['step','modelType','trained','success rate']].groupby(
                    ['step','modelType','trained'],as_index=False).mean()
        if graphicOutput:
            self._drawLineplot(df=data,x='step',y='success rate',
                    title=name+'_line_'+prefix+'_success rate_'+note,
                    style='trained',hue='trained',order='flex',legends=legends)
            self._drawBarChart(df=barChart,xGrp=xGrp,barGrp=barGrp2,
                               yLabel='success rate',
                               title=name+'_'+prefix+'_success rate_'+note)
            self._drawBarChart(df=barChart,xGrp=xGrp,barGrp=barGrp,
                               yLabel='failed rate',
                               title=name+'_'+prefix+'_failed rate_'+note)
        if textOutput:
            print('capa:{},target:{},value:{}'.format(
                    rsp.serviceCapa,'success rate',barChart[barGrp2]))
        else:
            return(barChart[barGrp2])

    def _createTbl(self,servicedata,hue,alg=None,separate='max. rebid'):    
        servicedata[separate] = servicedata[separate].apply(str)
        if alg is None:
            alg = list(set(servicedata['algorithm type']))
        alg.sort(reverse=False)
        rebid = list(set(servicedata[separate]))
        rebid.sort()
        
        row1 = [x+',max.rebid='+rebid[0] for x in alg]
        row5 = [x+',max.rebid='+rebid[1] for x in alg]
        column = np.arange(
            min(servicedata[servicedata[separate]==rebid[0]][
                                                'resource capacity']),
            max(servicedata[servicedata[separate]==rebid[0]][
                                                'resource capacity'])+1,
            10,dtype=int).tolist()
        
        servicedata = servicedata[['resource capacity',hue,'success rate']]
        for capa,algorithm in product(column,row1+row5):
            tmp = pd.DataFrame([[int(capa),algorithm,np.nan]],
                                 columns=servicedata.columns)
            servicedata = pd.concat([servicedata,tmp],axis=0)
        
        data = servicedata[['resource capacity',hue,'success rate']].groupby(
                ['resource capacity',hue],as_index=False).mean().round(3)
        data.sort_values(by='resource capacity',ascending=True,inplace=True)
        
        values1 = []
        values5 = []
        for r in row1:
            values = data[data['type']==r]['success rate'].tolist()
            values0 = [np.round(x,decimals=2) if not np.isnan(x) 
                                                    else '' for x in values]
            values1.append(values0)
        for r in row5:
            values = data[data['type']==r]['success rate'].tolist()
            values0 = [np.round(x,decimals=2) if not np.isnan(x) 
                                                    else '' for x in values]
            values5.append(values0)
        
        return values1,values5,alg,column

    def _getPerformanceData(self,name,stepRange,target,sites):
        data = self._collectData(name)
        if len(data)==0:
            return
        
        if stepRange is None:
            stepRange = (min(data['step']),max(data['step']))
        data = data.loc[(data['step']<=stepRange[1]) 
                            & (data['step']>=stepRange[0])]
        
        try:
            data['success rate'] = data['success'] / data['finishedBid']
        except:
            data['success rate'] = data['success'] / data['totalBid']
        try:
            data['failed rate'] = data['rejectedBid'] / data['finishedBid']
        except:
            try:
                data['failed rate'] = data['rejectedBid'] / data['totalBid']
            except:
                data['failed rate'] = 1-data['success rate']
        
        data['modelType'] = np.where(data['modelType']=='ConvCritic_ConvActor',
                                     'CNN-HW','MLP')        
        data['modelType'] = np.where(data['trained']==self.random,
                                     '',data['modelType'])
        data['trained'] = np.where(data['trained']==self.random,
                                            self.random,self.learned)
        data['category'] = data.apply(lambda row: 
                    row['trained'] + '_' + str(row['nrSites']) 
                    + '_' + str(row['modelType']),axis=1)
        cols = list(data.columns)
        cols[cols.index('totalSuccessRatio')] = 'success rate by time'
        data.columns = cols
        if sites is None:
            sites = set(data['nrSites'])
        data = data.loc[data['nrSites'].isin(sites)]
        return data
    
    def _getRegData(self,data,x,y):
        model = lambda x,a1,a2,a3,a4,a5: a1+a2*x+a3*x**2+a4*x**3+a5*x**4      
        mdl = model
        a,b = curve_fit(mdl,data[x],data[y])
        lst = np.array(data[x])
        pts = mdl(lst,*a)
        return pts
    
    def drawRegplot(self,df,x,y,title,style=None,hue=None,order='flex',
                      hue_order=None,legends=2,legendFontsize=None,
                      tickFontsize=None,size=None,separate=None,
                      x_decimals=1,y_decimals=1,linestyle=None,
                      dataRange=None):
        defaultFontsize = 15
        if tickFontsize is None:
            tickFontsize = defaultFontsize
        if legendFontsize is None:
            legendFontsize = defaultFontsize
        if size is None:
            length = 5
            height = 4
        else:
            length = size[0]
            height = size[1]
        if linestyle is None:
            linestyle = ['-','--']
        if dataRange is not None:
            try:
                df = df[(df[x]>=dataRange[0]) & 
                        (df[x]<=dataRange[-1])]
            except:
                pass

        if separate is None:
            if hue is not None:
                if hue_order is None:
                    if order=='flex':
                        tmp = df.groupby(hue)
                        tmp = (tmp.tail(1).drop_duplicates()
                               .sort_values(y,ascending=False))
                    else:
                        tmp = df.groupby(
                                hue,as_index=False).max().sort_values(hue)
                    hue_order = tmp[hue].tolist()
            else:
                hue_order = None
            
            fig,ax = plt.subplots()
            fig.set_size_inches(length,height)
            
            for i,h in enumerate(hue_order):
                tmp = df[df[hue]==h]
                regData = self._getRegData(tmp,x,y)
                ax.scatter(x=tmp[x].values,y=tmp[y].values)
                ax.plot(tmp[x].values,regData,label=h,linestyle=linestyle[i]) 
        
            ax.set_xlabel(xlabel=x,fontsize=tickFontsize)
            ax.set_ylabel(ylabel=y,fontsize=tickFontsize)
            if dataRange is not None:
                ax.set_xticks(dataRange[0::2])
            if tickFontsize!=defaultFontsize:
                if x_decimals>0:
                    ax.set_xticklabels(np.round(ax.get_xticks(),
                                                decimals=x_decimals),
                                       size=tickFontsize)
                else:
                    ax.set_xticklabels(np.int0(ax.get_xticks()),
                                       size=tickFontsize)
                if y_decimals>0:
                    ax.set_yticklabels(np.round(ax.get_yticks(),
                                                decimals=x_decimals),
                                       size=tickFontsize)
                else:
                    ax.set_yticklabels(np.int0(ax.get_yticks()),
                                       size=tickFontsize)
            if legends is not None:
                handles, labels = ax.get_legend_handles_labels() 
                l = ax.legend(handles[0:],labels[0:],loc=legends,
                              fontsize=legendFontsize)
                plt.savefig(os.path.join(GRAPH_DIR,title+'.pdf'),
                            bbox_extra_artists=(l,), bbox_inches='tight')
            else:
                l = ax.legend()
                l.remove()
                plt.savefig(os.path.join(GRAPH_DIR,title+'.pdf'),
                            bbox_inches='tight')
            plt.clf()
        else:
            sepCol = list(set(df[separate]))
            sepCol.sort()
            
            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
            fig.set_size_inches(length,height)
            
            df1 = df.loc[df[separate]==sepCol[0]]
            df2 = df.loc[df[separate]==sepCol[1]]
            
            if hue is not None:
                if hue_order is None:
                    if order=='flex':
                        tmp1 = df1.groupby(hue)
                        tmp1 = (tmp1.tail(1).drop_duplicates()
                               .sort_values(y,ascending=False))
                        tmp2 = df2.groupby(hue)
                        tmp2 = (tmp2.tail(1).drop_duplicates()
                               .sort_values(y,ascending=False))
                    else:
                        tmp1 = df1.groupby(
                                hue,as_index=False).max().sort_values(hue)
                        tmp2 = df2.groupby(
                                hue,as_index=False).max().sort_values(hue)
                    hue_order1 = tmp1[hue].tolist()
                    hue_order2 = tmp2[hue].tolist()
            else:
                hue_order1 = None
                hue_order2 = None
            
            for i,sep in enumerate(sepCol):
                if i==0:
                    for j,h in enumerate(hue_order1):
                        tmp = df[(df[separate]==sep) & (df[hue]==h)]
                        regData = self._getRegData(tmp,x,y)
                        ax1.scatter(x=tmp[x].values,y=tmp[y].values,s=4)
                        ax1.plot(tmp[x].values,regData,
                                 label=h+', '+separate+'='+str(sep),
                                 linestyle=linestyle[j])
                else:
                    for j,h in enumerate(hue_order2):
                        tmp = df[(df[separate]==sep) & (df[hue]==h)]
                        regData = self._getRegData(tmp,x,y)
                        ax2.scatter(x=tmp[x].values,y=tmp[y].values,s=4)
                        ax2.plot(tmp[x].values,regData,
                                 label=h+', '+separate+'='+str(sep),
                                 linestyle=linestyle[j])

            ax1.set_ylabel(ylabel=y,fontsize=tickFontsize)
            ax2.set_ylabel(ylabel=y,fontsize=tickFontsize)
            
            ax2.set_xlabel(xlabel=x,fontsize=tickFontsize)
            if dataRange is not None:
                ax2.set_xticks(dataRange[0::2])
            
            if tickFontsize!=defaultFontsize:
                if x_decimals>0:
                    ax1.set_xticklabels(np.round(ax1.get_xticks(),
                                                 decimals=x_decimals),
                                        size=tickFontsize)
                    ax2.set_xticklabels(np.round(ax2.get_xticks(),
                                                 decimals=x_decimals),
                                        size=tickFontsize)
                else:
                    ax1.set_xticklabels(np.int0(ax1.get_xticks()),
                                        size=tickFontsize)
                    ax2.set_xticklabels(np.int0(ax2.get_xticks()),
                                        size=tickFontsize)
                if y_decimals>0:
                    ax1.set_yticklabels(np.round(ax1.get_yticks(),
                                                 decimals=x_decimals),
                                        size=tickFontsize)
                    ax2.set_yticklabels(np.round(ax2.get_yticks(),
                                                 decimals=x_decimals),
                                        size=tickFontsize)
                else:
                    ax1.set_yticklabels(np.int0(ax1.get_yticks()),
                                        size=tickFontsize)
                    ax2.set_yticklabels(np.int0(ax2.get_yticks()),
                                        size=tickFontsize)

                ax1.xaxis.label.set_size(tickFontsize)
                ax2.xaxis.label.set_size(tickFontsize)
                ax1.yaxis.label.set_size(tickFontsize)
                ax2.yaxis.label.set_size(tickFontsize)
            
            if legends is not None:
                handles1, labels1 = ax1.get_legend_handles_labels() 
                ax1.legend(handles1[0:],labels1[0:],loc=legends,
                           fontsize=legendFontsize)
                handles2, labels2 = ax2.get_legend_handles_labels() 
                ax2.legend(handles2[0:],labels2[0:],loc=legends,
                           fontsize=legendFontsize)            
            else:
                l = ax.legend()
                l.remove()
            plt.savefig(os.path.join(GRAPH_DIR,title+'.pdf'),
                        bbox_inches='tight')
            plt.clf()


    def drawPerformance(self,name='performance',drawPerformanceOnly=True,
                        target='recentNonRejectRatio',prefix='',
                        textOutput=False,sites=None,legends=None,stepRange=None,
                        decimals=1,ci=None):
        
        if sites is not None:
            sites = [sites]
        if target=='totalSuccessRatio':
            target = 'success rate by time'
        data = self._getPerformanceData(name,stepRange,target,sites)

        if not drawPerformanceOnly:
            pattern = re.compile('cloud|standard|slow')
            
            for nrSites in sites:
                df = data.loc[data['nrSites']==nrSites]
                sitetype = pattern.findall(df.sitetype.iloc[0])
                sitetype = dict([('site'+str(i),x) 
                                    for i,x in enumerate(sitetype)])
                
                utilized = self._parseColumn(df,'utilization',sitetype)
                df = pd.concat([df,utilized],axis=1)
                analyzeCols = utilized.columns
                
                for t in analyzeCols:
                    tmp = df[['step','trained',t]].groupby(
                                    ['step','trained'],as_index=False).mean()
                    self._drawLineplot(df=tmp,x='step',y=t,
                        title=t+'_'+nrSites, style='trained',hue='trained',
                        order='fix',legends=None)
        
        if len(sites)==1:
            hue = 'trained'
        else:
            hue = 'category'
        if ci is None:
            data = data[['step',hue,target]].groupby(
                            ['step',hue],as_index=False).mean()
        self._drawLineplot(df=data,x='step',y=target,title=name+'_'+target,
                       style=hue,hue=hue,order='flex',legends=legends,
                       legendFontsize=12,tickFontsize=12,
                       decimals=decimals,ci=ci)

    
    def drawPriceModelLoss(self,name='priceLearningModel'):
        data = self._collectData(name)
        data = data.loc[(~data['actorLoss'].isnull()) & (data['actorLoss']<3) 
                        & (data['actorLoss']>-3)]
            
        for target in ['avgReward','criticLoss','actorLoss']:
            title = name + '_' + target
            df = data[['step','nrSites',target]].groupby(
                            ['step','nrSites'],as_index=False).mean()
            self._drawLineplot(df=df,x='step',y=target,
                               title=title,style='nrSites',hue='nrSites',
                               order='fix')  
            
    def drawCompetitorModelLoss(self,name='competitorLearningModel'):
        target = 'competitorLoss'
        data = self._collectData(name)
        data = data.loc[~data[target].isnull()]
        df = data.loc[data['step']>=10]
        df = df[['step','nrSites',target]].groupby(
                            ['step','nrSites'],as_index=False).mean()
        outlier = np.percentile(df[target],80)
        df = df[df[target]<outlier]
        self._drawLineplot(df=df,x='step',y=target,title=name,
                           style='nrSites',hue='nrSites',order='fix')




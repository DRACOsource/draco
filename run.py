# -*- coding: utf-8 -*-
from v2x.config.config import (PSTATS_FILE,LOGS_DIR,
                               RESOURCE_SITE_PARAMS as rsp)
from v2x.utils.common_utils import openFiles,closeFiles
from v2x.solutions.v2x import V2XModel
import cProfile
import pstats
import os,sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")
    
def runProfiler():
    cProfile.run('runStandAloneModel()', PSTATS_FILE)
    p = pstats.Stats(PSTATS_FILE)
    p.strip_dirs().sort_stats('cumulative').print_stats(30)
        
def runStandAloneModel(nrSites=2,train=True,evaluation=False,
                       loadModel=False,steps=8000,rebid=True):
    nrRsu = 1
    
    filedict = openFiles([nrSites,str(train)])
    mdl = V2XModel(filedict,nrSites=nrSites,nrRsu=nrRsu,train=train,
                   evaluation=evaluation,loadModel=loadModel,rebid=rebid)
    
    iterations = len(rsp.resourceCapa)
    chpt = list(np.arange(start=steps/iterations,stop=steps,
                     step=steps/iterations,dtype=int))[0:iterations-1]
    for i in range(steps):
        mdl.step()
        if i in chpt:
            actor_learning_rate = []
            for v in mdl.vehicles.values():
                actor_learning_rate.append(v.priceMdl.actor_learning_rate)
            print('actor learning rate: {},{}.'.format(
                 min(actor_learning_rate),max(actor_learning_rate)))
    closeFiles(filedict)
    return mdl

#%%
if __name__ == '__main__':

    train = True
    nrSites = 2
    evaluation = False
    loadModel = True
    steps = 8000
    rebid = False
    path = LOGS_DIR

    if 'rial' in sys.argv[1]:
        train = False
    if 'eval' in sys.argv[1]:
        evaluation = True
    if 'rebid' in sys.argv[1]:
        rebid = True
        
    try:
        path = os.path.join(path,sys.argv[2])
    except:
        pass
    
    runStandAloneModel(nrSites=nrSites,train=train,evaluation=evaluation,
                       loadModel=loadModel,steps=steps,rebid=rebid)
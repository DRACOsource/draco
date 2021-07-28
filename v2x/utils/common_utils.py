# -*- coding: utf-8 -*-

import re
from ..config.config import FILES

class CommonUtils():
    
    def listClassVariables(clazz,text='__',include=False):
        pattern = re.compile(text)
        if include:
            elements = [clazz.__dict__[variable] for variable 
                         in clazz.__dict__.keys() if pattern.match(variable)]
        else:
            elements = [clazz.__dict__[variable] for variable 
                    in clazz.__dict__.keys() if not pattern.match(variable)]        
        return elements
    
    def listClassNames(clazz,text='__',include=False,func=False):
        pattern = re.compile(text)        
        if not func:
            names = [variable for variable in clazz.__dict__.keys() 
                                if not callable(clazz.__dict__[variable])]
        else:
            names = [variable for variable in clazz.__dict__.keys()]
        if include:
            names = [x for x in names if pattern.match(x)]
        else:
            names = [x for x in names if not pattern.match(x)]        
        return names
    
    def listClassItems(clazz,text='__',include=False,func=False):
        pattern = re.compile(text)        
        if not func:
            items = [(k,v) for k,v in clazz.__dict__.items() 
                                if not callable(v)]
        else:
            items = [(k,v) for k,v in clazz.__dict__.items()]
        if include:
            items = dict([x for x in items if pattern.match(x[0])])
        else:
            items = dict([x for x in items if not pattern.match(x[0])])        
        return items        
    
    def recreateDict(originalDict,idx):
        if isinstance(idx,int):
            return dict(zip(originalDict.keys(),[x[idx] 
                            for x in originalDict.values()]))
        elif isinstance(idx,str):
            return dict(zip(originalDict.keys(),[eval('x.'+idx) 
                            for x in originalDict.values()]))
        else: return originalDict
        
def openFiles(additional=None,files=None):
    if additional is None:
        additional = []
    if files is None:
        files = CommonUtils.listClassVariables(FILES,'__')
    filedict = dict()
    for name,addr in files:
        for i in additional:
            addr = addr + '_' + str(i)
        addr += '.txt'
        filedict[name] = open(addr,'a+')
    return filedict

def closeFiles(filedict):
    for addr in filedict.values():
        addr.close()
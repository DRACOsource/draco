# -*- coding: utf-8 -*-

import numpy as np
from itertools import combinations,product
import sys

from ..utils.common_utils import CommonUtils as utCom
from ..config.config import (RESOURCE_PARAMS as rp, VEHICLE_PARAMS as vp)

class Distr():
    GAUSSIAN = 'gaussian'
    POISSON  = 'poisson'

class ServiceType():
    TYPE1 = 'service1' # service capacity = max amount #/ 2
    TYPE2 = 'service2' # service capacity = max amount

class ServiceAmount():
    HIGH = 'high'
    LOW  = 'low'
    
class ServiceName():
    F1 = ('F1',ServiceType.TYPE1)
    F2 = ('F2',ServiceType.TYPE2)
    
class SiteType():
    TYPE1 = 'standard'
    TYPE2 = 'slow'
    TYPE3 = 'cloud'
    
class ResourceType():
    TYPE1 = 'resource1'
    TYPE2 = 'resource2'

class ResourceName():
    H1 = ('cpu', ResourceType.TYPE1)
    H2 = ('memory', ResourceType.TYPE2)

class BidStatus():
    PREP = 'in preparation'
    BID = 'new bid'
    ADMIT = 'admitted'
    TRANSMIT = 'in transmission'
    PROCESS = 'in process'
    REALLOCATE = 'can be reallocated'
    REJECT = 'rejected'
    FINISH = 'finished'
    ARCHIVE = 'archived'

class BidFailureReason():
    DUPLICATE = 'other bids in batch accepted'
    NO_AVAILABLE_SITE = 'no proposals from sites'
    COST = 'service cost too high'
    ESTIM_DUE_TIME = 'estimated time required too long'
    ACTUAL_DUE_TIME = 'already too late'
    NOPRIO = 'not prioritized'
    ONHOLD = 'on-hold for too long'
    NA = 'not applicable'

class QoSType():
    HIGH = 'high quality'
    LOW = 'low quality'

class BidState():
    lowest = vp.totalBudget[0] * 10
    services = utCom.listClassVariables(ServiceName,'__',include=False)
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.nrBidsInCategory = 0
        self.highestPrice = 0
        self.lowestPrice = self.lowest
        
        self.currentService = dict()
        self.futureService =dict()
        for s in self.services:
            self.currentService[s] = 0
            self.futureService[s] = 0
    
    def getRecord(self):
        if self.nrBidsInCategory==0:
            lowestPrice = 0
        else:
            lowestPrice = self.lowestPrice
        return ( [self.nrBidsInCategory,self.highestPrice,lowestPrice] 
                  + list(self.currentService.values()) 
                  + list(self.futureService.values()) )
        
class DefaultProfile():
    ''' creates duration based on amount from a normal distribution.
            Duration is a inverse function of amount.
        currently not scalable as only two service types are allowed.
    '''
    # setting all std deviations of resource profiles to zero, to simplify
    # simulation of service time.
    AVG_AMT = {ServiceType.TYPE1:(1, 0),
               ServiceType.TYPE2:(2, 0)}
    DURATION = dict()
    DURATION[(SiteType.TYPE1,ServiceType.TYPE1)] = lambda amount: 1
    DURATION[(SiteType.TYPE1,ServiceType.TYPE2)] = lambda amount: int(max(1,
                                    np.random.normal(10 / (amount + 1), 0)))

    DURATION[(SiteType.TYPE2,ServiceType.TYPE1)] = lambda amount: 3
    DURATION[(SiteType.TYPE2,ServiceType.TYPE2)] = lambda amount: int(max(1,
                                    np.random.normal(30 / (amount + 1), 0)))
    
    DURATION[(SiteType.TYPE3,ServiceType.TYPE1)] = lambda amount: 1
    DURATION[(SiteType.TYPE3,ServiceType.TYPE2)] = lambda amount: int(max(1,
                                    np.random.normal(10 / (amount + 1), 0)))
    
    @classmethod
    def randomGenerate(cls,siteType,serviceType):
        ''' given service type, create randomly amount required and duration 
                required of one (random) resource.
            
            @param serviceType: service type name
            @return: resource amount required, duration required
        '''
        amount = int(np.max([1,np.random.normal(*cls.AVG_AMT[serviceType])]))
        duration = cls.DURATION[(siteType,serviceType)](amount)
        return amount,duration
    
    @classmethod
    def randomPopulate(cls,siteType,serviceType):
        ''' given service type, randomly generate a resource profile. 
                Currently services require all resources available.
            
            @param serviceType: service type name
            @return: resource profile (amount, duration) including 
                all resources
        '''
        resources = utCom.listClassVariables(ResourceName,'__',include=False)
        resProfile = dict()
        for res in resources:
            resProfile[res] = cls.randomGenerate(siteType,serviceType)
        return resProfile

class ResourceSiteDistance():
    ''' calculate distance between sites and transmission time in case
            the data need to be transmitted.
    '''
    minDistance = 10 #km
    maxDistance = 50
    cloudDistance = 4000
    h = 127 # channel gain base
    sigmaSqrt = 2E-13 # gaussian noise
    p = 0.5 # transmission power of sender
    I = 1E-12 # inter-cell interference power
    W = 20E6 # channel width for wireless
    bitrate = 10E6 # 10Gbps, or 10 megabits per millisecond for wired
    # speed of light in fiber assumed to be 70% of vacuum
    speedOfLight = 299792458 / 1000 / 1000 * 0.7 # km/ms
    propRandomVar = 10 # random factor in calculating propagation delay
    
    def __init__(self,sitelist):
        self.sitelist = sitelist.copy()
        self.distanceMatrix = dict()
        for site in sitelist:
            self.distanceMatrix[(site,site)] = 0
        for send,recv in combinations(sitelist,2):
            if send==sitelist[0]:
                self.distanceMatrix[(send,recv)] = self.cloudDistance
            else:
                self.distanceMatrix[(send,recv)] = np.random.randint(
                                            self.minDistance,self.maxDistance)
            self.distanceMatrix[(recv,send)] = self.distanceMatrix[(send,recv)]
    
    def addSite(self,siteid):
        if siteid in self.sitelist:
            print('siteid {} already exsists.'.format(siteid))
            return 
        self.distanceMatrix[(siteid,siteid)] = 0
        for site in self.sitelist:
            self.distanceMatrix[(site,siteid)] = np.random.randint(
                                        self.minDistance,self.maxDistance)
            self.distanceMatrix[(siteid,site)] = self.distanceMatrix[
                                                            (site,siteid)]
        self.sitelist.append(siteid)
    
    def deleteSite(self,siteid):
        if siteid not in self.sitelist:
            print('siteid {} does not exsist.'.format(siteid))
            return
        self.sitelist.remove(siteid)
        del self.distanceMatrix[(siteid,siteid)]
        for site in self.sitelist:
            del self.distanceMatrix[(site,siteid)]
            del self.distanceMatrix[(siteid,site)]
    
    def getWirelessLatency(self,sendId,recvId,dataSize):
        D = self.distanceMatrix[(sendId,recvId)]
        h = self.h + 30 * np.log10(D)
        r = self.W * np.log2(1+(self.p * h)/(self.sigmaSqrt + self.I))
        return dataSize / r

    def getWiredLatency(self,sendId,recvId,dataSize):
        propagation = self._getPropagationDelay(sendId,recvId)
        transmission = dataSize / self.bitrate
        return propagation + transmission
        
    def _getPropagationDelay(self,sendId,recvId):
        avg = self.distanceMatrix[(sendId,recvId)] / self.speedOfLight # in ms
        avg = avg + np.abs(np.random.normal(avg,avg/self.propRandomVar))
        return avg


class Task():
    ''' tasks of a bid. each task is an instance of one service. '''
    def __init__(self,position,serviceName,serviceType,serviceAmount,
                 bidId,bidCreatetime,bidQos,resourceProfile):
        self.unique_id = 'task_' + str(bidId) + '_' + str(position)
        self.pos = position # position of the task in the service chain
        self.bidId = bidId # bid info
        self.bidQos = bidQos # bid info
        self.bidCreatetime = bidCreatetime # bid info
        self.serviceName = serviceName # corresponding service info
        self.serviceType = serviceType # corresponding service info
        self.serviceAmount = serviceAmount # number of service units required
        self.resProfileOriginal = resourceProfile # suggested resource profile
        self.resProfileSelected = None # final selected resource profile 
                                       # by the resource site
        self.resProfileActual = None   # final consumed resource profile in
                                       # resource site
        self.cost = 0                  # cost of the task
        self.serviceBinding = None     # service unique_id 
        self.queueStart = 0            # time unit when entering the queue
        self.queueEnd = 0              # time unit when exiting the queue
        self.estimWaitingTimeTotal = 0 # estimated time to wait in queue
        self.estimWaitingTimeToGo = 0  # count down, not used
        self.serviceStart = 0          # time unit when entering server
        self.serviceEnd = 0            # time unit when exiting server
        self.serviceTime = 0           # total time served
        self.serviceTimeToGo = self.serviceTime # count down. when down to 
                                   # zero, Service._unloadService is triggered
        self.isInQueue = False         # indicator if in queue
        self.isInService = False       # indicator if in service
        self.isFinished = False        # indicator if done
    
    def updateResourceTime(self):
        ''' called by Service._unloadService from Service.step. 
                deduct 1 in each time step from the time-to-go created 
                in Task.resProfileActual.
        '''
        release = list()
        for res in self.resProfileActual.keys():
            resAmount, resDuration, resTimeToGo = self.resProfileActual[res]
            if resTimeToGo<=0:
                continue
            resTimeToGo -= 1
            self.resProfileActual[res] = (resAmount,resDuration,resTimeToGo)
            if resTimeToGo==0:
                release.append(res)
        return release
    
    def updateServiceTime(self,servicetime,time,estim=False):
        ''' update service-related indicators and info '''
        if estim is False:
            self.isInService = True
            self.serviceStart = time
        self.serviceTime = servicetime
        self.serviceTimeToGo = self.serviceTime
        
    def updateQueueTime(self,queuetime,time,estim=False):
        if estim is False:
            self.isInQueue = True
            self.queueStart = time
        self.estimWaitingTimeTotal = queuetime
        self.estimWaitingTimeToGo = queuetime

class ActiveTaskInfo():
    def __init__(self,taskId,serviceName,serviceType,
                 amount,duration,startTime):
        ''' container for active task information specifically in resources '''
        self.taskId = taskId
        self.serviceName = serviceName
        self.serviceType = serviceType
        self.amount = amount
        self.duration = duration
        self.startTime = startTime
        self.endTime = None

class Resource():
    ''' Resource created by resource sites. keeps track of utilization, 
        loading and unloading, and allocation to different services sharing 
        the same resource.
    '''
    defaultMaxAmount = rp.defaultMaxAmount
    unitCost = {ResourceType.TYPE1:rp.unitCost[0], 
                ResourceType.TYPE2:rp.unitCost[1]}
    
    def __init__(self,resName,resType,uniqueId,maxAmount=None):
        self.resName = resName
        self.resType = resType
        self.unique_id = uniqueId
        self.maxAmount = (maxAmount if maxAmount is not None 
                          else self.defaultMaxAmount)
        self.cost = self._calculateCost()
        self.occupied = dict()
        self.activeTaskList = dict()
        self.utilization = 0 # occupied / maxAmount
        self.occupiedHistory = dict() 
        self.utilizationHistory = dict()
        self.taskHistory = list()
        self.allocated = dict() # allocated (max) resource per service
        self.allocatedHistory = dict()
    
    def _calculateCost(self):
        return int(np.max([1,np.random.normal(*self.unitCost[self.resType])]))

    def _updateInfo(self,serviceName,serviceType,resAmount,time):
        ''' helper function to update utilization '''
        self.occupied[(serviceName,serviceType)] += resAmount
        self.occupiedHistory[(serviceName,serviceType)].append((self.occupied[
                                            (serviceName,serviceType)],time))
        self.utilization = sum(self.occupied.values()) / self.maxAmount
        self.utilizationHistory[str(time)] = self.utilization

    def _fillin(self,servNameType):
        ''' helper function '''
        if servNameType not in self.occupied.keys():
            self.occupied[servNameType] = 0
            self.occupiedHistory[servNameType] = [(0,0)]
        
        if servNameType not in self.allocated.keys():
            self.allocated[servNameType] = 0
            self.allocatedHistory[servNameType] = [(0,0)]
    
    def startTask(self,taskId,serviceName,serviceType,
                                      amount,duration,startTime):
        ''' load task into server '''
        self.activeTaskList[taskId] = ActiveTaskInfo(taskId,
                         serviceName,serviceType,amount,duration,startTime)
        self._updateInfo(serviceName,serviceType,amount,startTime)
    
    def endTask(self,taskId,endTime):
        ''' unload task from server '''
        if taskId in self.activeTaskList.keys():
            taskInfo = self.activeTaskList.pop(taskId)
            taskInfo.endTime = endTime
            self.taskHistory.append(taskInfo)
            self._updateInfo(taskInfo.serviceName,taskInfo.serviceType,
                                                 -taskInfo.amount,endTime)
    
    def checkResourceFeasibility(self,servName,servType,resAmount):
        ''' 
        check if a required resource amount is feasible (within 
            the limit of allocation to service). The function does not
            change maximum resource allocation to service. 
        it's called by ResourceSite.checkResFeasibility and 
            ultimately by Service._loadService. If not enough 
            resource for the next-task-in-line then the queue waits.

        @param servName, servType: service identifiers
        @param resAmount: resource amount requested
        @return: feasibility, deviation from requested
        '''
        servNameType = (servName,servType)
        self._fillin(servNameType)        
        if resAmount > 0: # requests
            leftover = (self.allocated[servNameType] 
                        - self.occupied[servNameType])
            if leftover < 0:
                return False,0
            amount = min(resAmount,leftover)
        else:
            amount = resAmount
        return amount==resAmount, amount-resAmount
    
    def allocateResource(self,servName,servType,resAmount,time,estim=False):
        ''' 
        allocate resource to the given service. 
        called by ResourceSite.checkResFeasibility (for estimation only) 
            and _adjustResAllocation (for actual allocation),
            and ultimately by ResourceSite._adjustCapacity. 
        
        @param servName,servType: service identifier
        @param resAmount: requested resource amount
        @param time: time step for debugging
        @param estim: if True: estimate feasibility for capacity adjustment 
            (specifically for a request to increase allocation to the
            given service). if False: do the allocation.
        '''
        servNameType = (servName,servType)
        self._fillin(servNameType)
            
        if resAmount > 0: # requests
            leftover = self.maxAmount - sum(self.allocated.values())
            if leftover < 0:
                return False,0
            amount = min(resAmount,leftover)
        else:
            amount = resAmount
        
        if not estim: # actually allocate:
            self.allocated[servNameType] = max(self.occupied[servNameType], 
                                       self.allocated[servNameType] + amount)
            self.allocatedHistory[servNameType].append(
                                        (time,self.allocated[servNameType]))
        return amount==resAmount, amount-resAmount

class ServiceProposal():
    ''' container for proposals created by resource sites and sent 
            to the bids.
    '''
    def __init__(self,bidCost,destId,estimFinishTime,dataId,
                                       transmitTime,resProfileSelected):
        self.proposedBidCost = bidCost
        self.proposedDest = destId
        self.proposedEstimFinishTime = estimFinishTime
        self.proposedDataId = dataId
        self.proposedTransmitTime = transmitTime
        self.proposedResProfile = resProfileSelected

    def reset(self):
        self.proposedBidCost = sys.maxsize
        self.proposedDest = None
        self.proposedEstimFinishTime = self.dueTime
        self.proposedDataId = None
        self.proposedTransmitTime = 0
        self.proposedResProfile = dict()
    
    def __gt__(self,other): # "better than"
        return self.proposedBidCost < other.proposedBidCost
    
    
    def updateTaskSuccess(self,taskPos,taskSuccess=False):
        if self.taskPos==taskPos:
            self.taskSuccess = taskSuccess
    
    def updateTaskTransferred(self,taskPos,taskTransferred=False):
        if self.taskPos==taskPos:
            self.taskTransferred = taskTransferred
    
    def updateEnvInfo(self,nrVehicles,nrActiveBids):
        self.nrVehicles = nrVehicles
        self.nrActiveBids = nrActiveBids        
    
    def updateBidInfo(self,bidSuccess,qosActual,priceLeftover):
        self.qosActual = qosActual
        self.bidSuccess = bidSuccess
        self.finalPriceLeftover = priceLeftover
        

class SecondPriceData():
    ''' container for creating bid records either for estimating the
            second price in v2x.RSU._recordSecondPrice, or for calculating
            the payment in v2x.RSU._calculatePayment.
    '''
    services = utCom.listClassVariables(ServiceName,'__',include=False)
    TIME = 'time'
    QOS = 'qos'
    VEH = 'nrVehicles'
    BID = 'nrBids'
    ESTIM = 'estimSecondPrice'
    keys = services + [TIME,QOS,VEH,BID,ESTIM]
    def __init__(self,sites):
        self.record = dict()
        for k in sites + self.keys:
            self.record[k] = 0

def createKeysDeco(cls):
    # https://stackoverflow.com/questions/13900515/how-can-i-access-a-classmethod-from-inside-a-class-in-python
    cls.createKeys()
    return cls

@createKeysDeco
class CompetitorBaseData():
    ''' container for static list of all services and resources, and
            keys for CompetitorLearningModel
        to access a class method from inside a static class, a decorator 
            is used. This class method is executed together with creation of 
            the static class (done via the decorator which calls on the
            class method). No need to instantiate.
    '''
    services = utCom.listClassVariables(ServiceName,'__',include=False)
    serviceAmounts = utCom.listClassVariables(ServiceAmount,'__',include=False)
    qosTypes = utCom.listClassVariables(QoSType,'__',include=False)
    serviceLevels = list()
    keys = list()
    
    @classmethod
    def createKeys(cls):
        if len(cls.services)>1:
            levels = list()
            for s in cls.services:
                levels.append([(s,a) for a in cls.serviceAmounts])    
            serviceLevels = list(product(*levels))
        else:
            serviceLevels = [(cls.services[0],a) for a in cls.serviceAmounts]
        cls.serviceLevels = serviceLevels
        cls.keys = list(product(serviceLevels,cls.qosTypes))
    


            
        
        

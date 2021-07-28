# -*- coding: utf-8 -*-
from ..utils.common_utils import CommonUtils as utCom
from ..supports.data import (ServiceType,ServiceAmount,ServiceProposal,
                ResourceSiteDistance,SiteType,BidStatus,BidState,
                BidFailureReason as bfr,QoSType,Task,DefaultProfile,
                ResourceName,ResourceType,Resource,
                SecondPriceData as sec,CompetitorBaseData as cbd)
from ..config.config import (RESOURCE_SITE_PARAMS as rsp,SERVICE_PARAMS as sp,
                             VEHICLE_PARAMS as vp, BID_PARAMS as bp,
                             MDL_PARAMS as mp, RSU_PARAMS as rp,
                             TRANSITION_PARAMS as tp, \
                             PRICE_MODEL_PARAMS as pmp,
                             FILES,COUNTDOWN_FILE,BATCHCREATE_FILE,
                             MODEL_DIR)
from ..solutions.price_model_fsp import PriceLearningModel as plm, SupervisedModel as spv
from ..solutions.competitor_model import CompetitorLearningModel as clm

import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from collections import Counter
from scipy.stats import expon
import os,sys,re
from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import MultiGrid
import weakref
import pickle

class ResourceSite(Agent):
    '''
    step:.
    estimate: 1. release resource from finished tasks. 
        2. adjust services capacity based on updated resource profiles.
        3. collect resource and service information for the next time step 
        (after the release of resources).
        4. estimate time and cost of each bid. ResourceSite.estimTimeInSystem
        called from Bid.estimate.
    allocate:.
    advance: load newly assigned tasks to service queues. 
    '''
    _instances = set()
    # type3: cloud with high capacity
    serviceCapa = {(SiteType.TYPE1,ServiceType.TYPE1):rsp.serviceCapa[0],
                   (SiteType.TYPE1,ServiceType.TYPE2):rsp.serviceCapa[1],
                   (SiteType.TYPE2,ServiceType.TYPE1):rsp.serviceCapa[0],
                   (SiteType.TYPE2,ServiceType.TYPE2):rsp.serviceCapa[1],
                   (SiteType.TYPE3,ServiceType.TYPE1):rsp.serviceCapa[2],
                   (SiteType.TYPE3,ServiceType.TYPE2):rsp.serviceCapa[2]}

    transCost = rsp.transCost
    burnIn = rsp.burnIn
    lowerCapaLimit = rsp.lowerCapaLimit
    upperCapaLimit = rsp.upperCapaLimit
    resProfileSelectionThreshold = rsp.resProfileSelectionThreshold
    randomizeQueueTimeThres = rsp.randomizeQueueTimeThres
    
    def __init__(self,uniqueId,sitetype,resourceCapaIdx,model):
        super().__init__(uniqueId, model)
        self.unique_id = uniqueId
        self.sitetype = sitetype
#        if sitetype==SiteType.TYPE3:
#            self.lowerCapaLimit = 0 # no sizing down of cloud servers
        self.pos = (0,0) # coordinate of the resource site
        self.availBudget = 0 # dummy variable
        
        self.resourceCapa = None
        self._updateResCapa(resourceCapaIdx)
        
        self.servicelist = dict() # list of service ids
        self.resourcelist = dict() # list of resource ids
        self.bidlist = list() # list of bids allocated by the RSU
        self.currentTimeInSys = dict() # expected waiting time in queue
        self.maxAmounts = dict() # queue+server capacity of services
        self.currentCost = dict() # cost of one unit of service
        self.currentOccupied = dict() # occupied service units
        self.predictOccupied = dict() # predicted occupied service units 
                                # based on predicted service utilization
        self.currentUtilization = dict() # utilization of service capacities
        self.currentResourceOccupied = dict() # resource occupied by servers
        self.currentResourceUtilization = dict() # resource utilized by servers
        self.toBeReleased = dict() # resource to release from finished tasks
        self.siteInfo = dict() # information of other sites
        self.stateHistory = list() # history of utilization, occupied services,
                                   # cost and waiting time in queue
        self.serviceOfferHistory = list() # services queue length, 
                                    # servicelist length, maxAmount, 
                                    # service capacity; 
        self.serviceOfferHistoryCols = None
        self.resourceOfferHistory = list() # resources maxAmount, occupied
        self.resourceOfferHistoryCols = None
        self.utilHistory = list() # history of utilization
        self.prevBidCounter = 0 # record the previous time step's nr bids
        self.bidCounter = 0 # record current time step's nr bids
        self.serviceOccupiedHistory = dict() # history of service occupied
        self.serviceMaxAmountHistory = dict()
        self.randomizeQueueTimeHistory = dict() # history of waiting time
        self.randomizeQueueTimeParams = dict() # mean and std of queue length
        self.paramsHistory = dict() # history of randomization params
        
        self._populate()
        
        self._instances.add(weakref.ref(self))
    
    def _updateResCapa(self,idx):
        self.resourceCapa = {
                (SiteType.TYPE1,ResourceType.TYPE1):rsp.resourceCapa[idx][0],
                (SiteType.TYPE1,ResourceType.TYPE2):rsp.resourceCapa[idx][1],
                (SiteType.TYPE2,ResourceType.TYPE1):rsp.resourceCapa[idx][0],
                (SiteType.TYPE2,ResourceType.TYPE2):rsp.resourceCapa[idx][1],
                (SiteType.TYPE3,ResourceType.TYPE1):rsp.resourceCapa[idx][2],
                (SiteType.TYPE3,ResourceType.TYPE2):rsp.resourceCapa[idx][2]}
        try:
            for key in self.resourcelist.keys():
                self.maxAmounts[key] += np.random.randint(
                                *self.resourceCapa[(self.sitetype,key[1])])
                res = self.resourcelist[key]
                res.maxAmount = self.maxAmounts[key]
                for servKeys in res.occupied.keys():
                    res._updateInfo(servKeys[0],servKeys[1],0,
                                    self.model.schedule.steps)
        except:
            pass
    
    def _populate(self):    
        ''' populate ResourceSite with resources and services. Currently: 
                every ResourceSite has all available resources and service
                types in the system.
        '''
        resources = utCom.listClassVariables(ResourceName,'__',include=False)
        resCosts = dict()
        for resName,resType in resources:
            self.maxAmounts[(resName,resType)] = np.random.randint(
                                *self.resourceCapa[(self.sitetype,resType)])
            newRes = Resource(resName,resType,
                              str(self.unique_id)+'_'+str(resName),
                              self.maxAmounts[(resName,resType)])
            self.resourcelist[(resName,resType)] = newRes
            resCosts[(resName,resType)] = newRes.cost
            
            self.toBeReleased[(resName,resType)] = list()
            self._getResourceInfo(resName,resType)
        
        resCosts = utCom.recreateDict(self.resourcelist,'cost')
        
        services = cbd.services
        for servName,servType in services:
            self.maxAmounts[(servName,servType)] = np.random.randint(
                                *self.serviceCapa[(self.sitetype,servType)])
            newServ = Service(servName,servType,
                        str(self.unique_id)+'_'+str(servName), 
                        self.unique_id,self.sitetype,resCosts,self.model,
                        self.maxAmounts[(servName,servType)])
            self.servicelist[(servName,servType)] = newServ
            self.model.schedule.add(newServ)
            self.model.servicelist[newServ.unique_id] = newServ
            
            self._getServiceInfo(servName,servType)
            
            for res in newServ.resNeedsAvg.keys():
                resAmount = (newServ.resNeedsAvg[res][0] 
                                            * newServ.serviceCapacity)
                self.resourcelist[res].maxAmount += resAmount
                _ = self.resourcelist[res].allocateResource(servName,servType,
                                     resAmount,self.model.schedule.steps)
    
    def _estimTaskDuration(self,task,serv,qos=None):
        ''' estimates duration of given task. called by estimTimeInSystem,
                which is called by Bid.estimate.
            this function calls _selectResourceProfile by default. However
                if a specific qos requirement is given, the function 
                calls Service.estimateResourceAmount and
                Service.estimateServiceDuration directly.
            
            @param task: given task to be evaluated
            @param serv: service object required
            @param qos: required quality of service (time to finish)
            @return: selected resource profile, estimated task cost, 
                estimated task duration.
        '''
        if qos is None:
            profile = self._selectResourceProfile(task.resProfileOriginal,
                                    task.serviceName, task.serviceType)
        else:
            profile = serv.estimateResourceAmount(qos/task.serviceAmount)
            
        resAmounts = utCom.recreateDict(profile,0)
        feasible,_ = self.checkResFeasibility(resAmounts,serv.serviceName,
                                              serv.serviceType)
        if not feasible:
            return None
        
        cost = serv.calculateUnitCost(profile) * task.serviceAmount
        
        if qos is None:
            duration = serv.estimateServiceDuration(
                                            resAmounts) * task.serviceAmount
        else:
            duration = qos
            
        return profile,cost,duration
    
    def _getResourceInfo(self,resName,resType):
        ''' collect resource information in ResourceSite.estimate. 
            @param resName: name of resource
            @param resType: type of resource
        '''
        res = self.resourcelist[(resName,resType)]
        self.maxAmounts[(resName,resType)] = res.maxAmount
        self.currentResourceOccupied[(resName,resType)] = res.occupied
        self.currentResourceUtilization[(resName,resType)] = np.round(
                                                            res.utilization,1)
        # resources maxAmount, allocated per service, occupied per service
        step = self.model.schedule.steps
        self.resourceOfferHistory.append([step,(resName,resType),
                     res.maxAmount] + list(res.allocated.values())
                     + list(res.occupied.values()))
        if (self.resourceOfferHistoryCols is None or 
                        len(self.resourceOfferHistoryCols)
                        <len(self.resourceOfferHistory[-1])):
            self.resourceOfferHistoryCols = (['step','key','maxAmount'] 
                + ['allocated_'+x[1] for x in res.allocated.keys()] 
                + ['occupied_'+x[1] for x in res.occupied.keys()])
    
    def _getServiceInfo(self,servName,servType,prediction=True):
        ''' collect service information in ResourceSite.estimate. 
            @param servName: name of service
            @param servType: type of service
            @param prediction: if true: use predictive methods to forecast
                utilization and estimate cost of one unit of service. 
                if false: use the most recent utilization value.
        '''
        key = (servName,servType)
        serv = self.servicelist[key]
        self.currentCost[key] = serv.calculateUnitCost(serv.resNeedsAvg, 
                                                                prediction)

        self.currentOccupied[key] = serv.occupied
        self.predictOccupied[key] = serv.predictedUtil * serv.maxAmount
        if key not in self.serviceOccupiedHistory.keys():
            self.serviceOccupiedHistory[key] = list()
            self.serviceMaxAmountHistory[key] = list()
            self.randomizeQueueTimeHistory[key] = list()
            self.randomizeQueueTimeParams[key] = (1,0)
        self.serviceOccupiedHistory[key].append(serv.occupied)
        self.serviceMaxAmountHistory[key].append(serv.maxAmount)
        self.randomizeQueueTimeHistory[key].append(
                                    [serv.occupied,self.prevBidCounter,1])
        
        self.currentUtilization[key] = np.round(serv.utilization,1)
        self.maxAmounts[key] = serv.maxAmount
        
        serviceTime = int(max([x[1] for x in serv.resNeedsAvg.values()]))
        try:
            avgServiceAmountPerTask = int(np.round(serv.occupied / 
                    (len(serv.queuelist)+len(serv.servicelist)),decimals=0))
        except:
            avgServiceAmountPerTask = 1
        
        if len(serv.queuelist)>0:
            self.currentTimeInSys[key] = (serv.estimTimeInQueue 
                                 + serviceTime * avgServiceAmountPerTask)
        elif (len(serv.servicelist)*avgServiceAmountPerTask
                                          <serv.serviceCapacity):
            self.currentTimeInSys[key] = 0
        else:
            self.currentTimeInSys[key] = serviceTime * avgServiceAmountPerTask

        # services queue length, servicelist length, maxAmount, 
        # service capacity; resources maxAmount, occupied:
        step = self.model.schedule.steps
        self.serviceOfferHistory.append([step,key,
                len(serv.queuelist)*avgServiceAmountPerTask,
                len(serv.servicelist)*avgServiceAmountPerTask,
                serv.maxAmount,serv.serviceCapacity])
        if self.serviceOfferHistoryCols is None:
            self.serviceOfferHistoryCols = (['step','key','queueLength',
                'serviceLength','maxAmount','serviceCapacity'])
    
    def _adjustCapacity(self):
        ''' Adjusts queue and service capacity based on updated 
                resource profile. Trigger is Service.resNeedsRequested 
                information from Service._genActualServiceDuration called in
                Service.advance. The value is generated by 
                Service._estimateResNeedCurve() which runs a linear regression
                of duration vs. resource amount based on historical records
                of actual resource needs.
            To prevent jitters, adjustments are done with minimum interval 
                controlled by Service.recentCapaDecrease,  
                Service.recentCapaIncrease and Service.avgCapacityPeriod.
            For previously over-estimated resource profiles: simply release
                the excess. For previously under-estimated resource profiles:
                an increase in allocated resource to the service needs to 
                be 1. prioritized between services, based on the urgency 
                of the need; and 2. checked for feasibility among all 
                resources. 
        '''
        if self.model.schedule.steps <= self.burnIn:
            return
        
        required = dict()
        for servNameType in self.servicelist.keys():
            serv = self.servicelist[servNameType]
            # update resource needs after new estimation of resource
            # need becomes available. no change to capacity
            if serv.resNeedsRequested is not None:
                oldMax = serv.serviceCapacity
                newMax = self._adjustResAllocation(servNameType,
                                                   serv.resNeedsRequested)
                serv.updateUtilization(0,newMax-oldMax)
                serv.resNeedsRequested = None
            
            # update capacity
            if (len(serv.occupiedHistory)>=serv.avgCapacityPeriod):
                peak = max(list(
                    serv.occupiedHistory.values())[-serv.avgCapacityPeriod:])
            else: 
                peak = max(list(serv.occupiedHistory.values()))
            # in case of recent overflow
            if (len(serv.overCapacityHistory)>0 and 
              max([int(x) for x in serv.overCapacityHistory.keys()]) 
              >= self.model.schedule.steps - serv.avgCapacityPeriod):
                peak = serv.maxAmount
            
            req = peak + serv.capacityBuffer * serv.maxAmount
            reqPerc = req/serv.maxAmount
            
            if (reqPerc < self.lowerCapaLimit and serv.recentCapaDecrease 
                < self.model.schedule.steps - serv.avgCapacityPeriod):
                released = -int(serv.serviceCapacity / 2)
                # make sure serviceCapacity is at least 1
                if released == -serv.serviceCapacity:
                    released += 1
                for res in serv.resNeedsEstim.keys():
                    _ = self.resourcelist[res].allocateResource(
                            servNameType[0],servNameType[1],
                            released,self.model.schedule.steps)
                serv.updateUtilization(0,diffMax=released)               
            elif (reqPerc >= self.upperCapaLimit and serv.recentCapaIncrease 
                < self.model.schedule.steps - serv.avgCapacityPeriod):
                required[servNameType] = 2 * serv.serviceCapacity # total required
        
        # prioritize service by amount of resource increase required.
        for servNameType in sorted(required,key=required.get,reverse=True):
            serv = self.servicelist[servNameType]
            oldMax = serv.serviceCapacity
            if required[servNameType]==oldMax or required[servNameType]<=0:
                break
            resAmount = dict([(k,serv.resNeedsAvg[k][0]*required[servNameType])
                                for k in serv.resNeedsAvg.keys()])
            # check feasibility of the increase
            newMax = self._adjustResAllocation(servNameType,resAmount)
            serv.updateUtilization(0,newMax-oldMax)
        
    def _adjustResAllocation(self,servNameType,requestResAmount):
        ''' check feasibility of a resource allocation increase to a service.
            called by ResourceSite._adjustCapacity from ResrouceSite.estimate.
            
            @param servNameType: (service name, service type) key.
            @param requestResAmount: new maximum resource amounts requested (
                when service capacity is fully utilized).
            @return: new value for the service's maximum capacity, based on
                availability of resources.
        '''
        serv = self.servicelist[servNameType]
        originalMax = dict([(k,self.resourcelist[k].allocated[servNameType]) 
                            for k in requestResAmount.keys()])
        requested = dict([(k,requestResAmount[k]-originalMax[k]) 
                            for k in requestResAmount.keys()])
        
        key = list(requested.keys())[0]
        requestServAmount = int(requestResAmount[key] 
                                    / serv.resNeedsAvg[key][0]) 
        
        feasible,diffAmount = self.checkResFeasibility(
                                requested,servNameType[0],servNameType[1],
                                capaChange=True)
        
        newMaxAmountService = max(requestServAmount, serv.occupied)
        if not feasible:
            diffAmount = dict([(k,diffAmount[k]) for k in diffAmount.keys() 
                                                if serv.resNeedsAvg[k][0]>0])
            reduce = [int(diffAmount[k]/serv.resNeedsAvg[k][0])
                        for k in diffAmount.keys() if int(diffAmount[k]
                        /serv.resNeedsAvg[k][0])==int(min(diffAmount.values())
                        /serv.resNeedsAvg[k][0])][0]
            newMaxAmountService = max(newMaxAmountService + reduce, 1)
            
        actualDiff = dict()
        for res in requestResAmount.keys():
            requested[res] = (newMaxAmountService * serv.resNeedsAvg[res][0] 
                                                        - originalMax[res])
            resource = self.resourcelist[res]
            _,actualDiff[res] = resource.allocateResource(servNameType[0],
                    servNameType[1],requested[res],self.model.schedule.steps)
            
        if not np.alltrue([x==0 for x in actualDiff.values()]):
            reduce = [int(actualDiff[k]/serv.resNeedsAvg[k][0])
                        for k in actualDiff.keys() if int(actualDiff[k]
                        /serv.resNeedsAvg[k][0])==int(min(actualDiff.values())
                        /serv.resNeedsAvg[k][0])][0]
            newMaxAmountService = max(newMaxAmountService + reduce, 1)            
                
        return newMaxAmountService
    
    def _selectResourceProfile(self,userResProfile,servName,servType):
        ''' placeholder for more sophisticated resource profile selection. 
                currently: if ResourceSite has seen the service enough
                times, it uses its own resource profile learned from the past.
                Else it takes the suggested resource profile from 
                the vehicle.
            Currently: actual resource consumed is generated when task is 
                loaded to server for processing through 
                Service._genActualServiceDuration which is called in 
                Service._loadService by Service.advance. The actual resource
                consumed value is drawn from data.DefaultProfile. duration
                is a inverse function of amount, however in 
                Service.estimateServiceDuration and 
                Service.estimateResourceAmount, a linear approximation is
                used.
            this function is called by ResourceSite._estimTaskDuration which 
                is called by ResourceSite.estimTimeInSystem in Bid.estimate. 
                The selected profile is used if the estimated duration is
                within the bid's qos requirements. 
                
            @param userResProfile: resource profile suggested by the vehicle
            @param servName: required service name
            @param servType: required service type
            @return: resource amount required for the service as given by 
                the selected resource profile.
        '''
        serv = self.servicelist[(servName,servType)]
        if len(serv.servicedTaskHistory) < self.resProfileSelectionThreshold:
            return userResProfile
        else:
            qos = max([x[1] for x in userResProfile.values()])
            return serv.estimateResourceAmount(qos)    

    def _load(self):
        ''' load newly assigned tasks to services queues. called from 
                ResourceSite.advance.
            function through Service.loadQueue.
        '''
        for bidId in self.bidlist:
            bid = self.model.bidlist[bidId]
            
            # in transit bids
            if bid.transmissionTimeToGo > 0:
                bid.updateResult(BidStatus.TRANSMIT,self.model.schedule.steps)
                bid.transmissionTimeToGo -= 1
                continue
            
            bid.updateResult(BidStatus.PROCESS,self.model.schedule.steps)
            task = bid.chain[bid.currentTask]
            if (not task.isInQueue):
                try:
                    resAmounts = utCom.recreateDict(task.resProfileSelected,0)
                except:
                    resAmounts = utCom.recreateDict(task.resProfileOriginal,0)
                serv = self.servicelist[(task.serviceName,task.serviceType)]
                taskDuration = serv.estimateServiceDuration(
                                            resAmounts) * task.serviceAmount
                if (taskDuration + serv.estimTimeInQueue > bid.dueTime):
                    queuetime = None
                else: 
                    binding,queuetime = serv.loadQueue(task.unique_id,bidId,
                                task.serviceAmount,resAmounts)
                
                if queuetime is not None:
                    bid.priority = 0 # reset priority
                    task.serviceBinding = binding
                    task.updateQueueTime(queuetime,self.model.schedule.steps)
                else:
                    bid.priority += 1 # increase bid priority
                    bid.updateResult(BidStatus.REALLOCATE,
                                     self.model.schedule.steps)

    def checkResFeasibility(self,resAmounts,servName,servType,
                            capaChange=False):
        ''' check if the requested resource amount is feasible and can be 
                assigned. called by ResourceSite._adjustCapacity from 
                ResourceSite.estimate and Service._loadService 
                from Service.advance.
            
            @param resAmounts: requests resource amounts
            @param servName: service name
            @param servType: service type
            @param capaChange: if the resource request comes from 
                ResourceSite._adjustCapacity, then capaChange is true. 
                if it is to check feasibility at the time of 
                Service._loadService, then it is false.
            @return: if the resource requested can be assigned, and if
                there are discrepancies between requested and assigned, the
                difference is also returned.
        '''
        if not capaChange:
            difference = [self.resourcelist[k].checkResourceFeasibility(
                          servName,servType,resAmounts[k])
                          for k in resAmounts.keys()]
        else:
            difference = [self.resourcelist[k].allocateResource(servName,
                          servType,resAmounts[k],self.model.schedule.steps,
                          estim=True) for k in resAmounts.keys()]
        feasible = np.alltrue([x[0] for x in difference])
        diffAmount = [x[1] for x in difference]
        return feasible, dict(zip(resAmounts.keys(),diffAmount))   
    
    def randomizeQueueTime(self):
        ''' 
        randomize estimate of expected time in queue. The 
            ResourceSite.randomizeQueueTimeHistory is the record of service
            capacity occupied and actual number of bids in queue at the time, 
            collected in ResourceSite._getServiceInfo from 
            ResourceSite.estimate. This data will be the input to a linear
            regression model with the target of the actual service capacity
            occupied in the following time step. The result is a parameterized
            linear model to predict next time step's occupied service capacity.
            randomness is added from the distribution of the residuals. This
            affects the price offer from the resource site.
        Params are stored in ResourceSite.randomizeQueueTimeParams, 
            and used in ResourceSite.estimTimeInSystem called from 
            Bid.estimate. 
        this function is called in ResourceSite.step.
        '''
        for key in self.randomizeQueueTimeHistory.keys():
            if (len(self.randomizeQueueTimeHistory[key]) 
                >= self.randomizeQueueTimeThres):
                x = np.array(self.randomizeQueueTimeHistory[key][:-1])
                y = self.serviceOccupiedHistory[key][1:]
                try:
                    params = np.linalg.lstsq(x,y,rcond=None)[0]
                except:
                    params = np.zeros(x.shape[1])
                residuals = y - np.dot(x, params)
                self.randomizeQueueTimeParams[key] = (np.mean(residuals),
                                                 np.std(residuals))
                if key not in self.paramsHistory.keys():
                    self.paramsHistory[key] = list()
                self.paramsHistory[key].append((params,residuals))
    
    def estimTimeInSystem(self,bidId):
        ''' 
        estimates for the given bidId the total cost and duration, based on
            a snapshot of the current service utilization. if the default 
            resource profile cannot meet qos requirements, a reverse 
            calculation is done to get the required resource amount.
        the function is called from Bid.estimate.
        
        @param bidId: unique_id of the bid in question.
        @return: total estimated duration for the bid, 
            leftover time until requested qos,
            selected resource profile for each service in bid, 
            total estimated cost for the bid, and duration to transmit the 
            data from another site if applicable.
        '''
        bid = self.model.bidlist[bidId]
        pos = bid.currentTask
        waitingTime = dict()
        taskDuration = dict()
        transDuration = 0
        estimCost = dict()
        transCost = 0
        resProfile = dict() # list of dictionaries
        leftover = 0
        maxIter = 10
        for i in np.arange(pos,len(bid.chain)):
            task = bid.chain[i]
            servNameType = (task.serviceName,task.serviceType)
            serv = self.servicelist[servNameType]
            if ( (i==pos) and (task.serviceAmount
                 +self.currentOccupied[servNameType]
                 >=self.maxAmounts[servNameType]) ):
                # if queue is full
                return 0,0,resProfile,sys.maxsize,0
            # waiting time estimate in queue
            if self.currentOccupied[servNameType]==0:
                waitingTime[str(i)] = 0
            else:
                waitingTime[str(i)] = int(serv.estimTimeInQueue
                  + serv.estimTimeInQueue / serv.occupied
                  * np.random.normal(
                          *self.randomizeQueueTimeParams[servNameType]))
            # task duration estimate for the current task
            result = self._estimTaskDuration(task,serv,None)
            if result is None:
                # if resource is fully utilized
                return 0,0,resProfile,sys.maxsize,0
            resProfile[str(i)],estimCost[str(i)],taskDuration[str(i)] = result
                
        leftover = (bid.dueTime - self.model.schedule.steps 
                    - sum(waitingTime.values()) - sum(taskDuration.values()) )
        transDuration = self.siteInfo.getWiredLatency(bid.siteBinding,
                                                self.unique_id,bid.datasize)
        leftover -= transDuration
        transCost = self.transCost * self.siteInfo.distanceMatrix[
                                        (bid.siteBinding,self.unique_id)]
        count = 0
        while leftover<0 and count<maxIter:
            taskIdx = max(taskDuration,key=taskDuration.get)
            leftover -= taskDuration[taskIdx]

            task = bid.chain[int(taskIdx)]
            serv = self.servicelist[(task.serviceName,task.serviceType)]
            
            qos = int(max(1,taskDuration[taskIdx] / 2))
            result = self._estimTaskDuration(task,serv,qos)
            if result is None:
                return 0,0,resProfile,sys.maxsize,0
            resProfile[taskIdx],estimCost[taskIdx],taskDuration[taskIdx]=result
            leftover += taskDuration[taskIdx]
            count += 1
        
        if leftover < 0:
            return 0,0,resProfile,sys.maxsize,0
        
        totalDuration = int(sum(waitingTime.values()) 
                            + sum(taskDuration.values()) + transDuration)
        totalCost = int(sum(estimCost.values()) + transCost)
        
        self.bidCounter += 1
        return totalDuration,leftover,resProfile,totalCost,int(transDuration)

    def step(self):
        self.prevBidCounter = self.bidCounter
        self.bidCounter = 0
        self.randomizeQueueTime()
    
    def estimate(self):
        for res in self.toBeReleased.keys():
            for taskId in self.toBeReleased[res]:
                self.resourcelist[res].endTask(taskId,
                                               self.model.schedule.steps)
            self.toBeReleased[res] = list()
        self.utilHistory.append(self.currentUtilization.copy())
        self._adjustCapacity()
        
        for res in self.resourcelist.keys():
            self._getResourceInfo(*res)
        for serv in self.servicelist.keys():
            self._getServiceInfo(*serv)
    
    def allocate(self):
        util = [(k,(self.currentOccupied[k]/self.maxAmounts[k],
                    self.currentOccupied[k],self.currentCost[k],
                    self.currentTimeInSys[k])) 
                for k in self.currentOccupied.keys()]
        self.stateHistory.append(dict(util))
    
    def advance(self):
        self._load()
    
    @classmethod
    def getInstances(cls):
        nonexistent = set()
        for ref in cls._instances:
            obj = ref()
            if obj is not None:
                yield obj
            else:
                nonexistent.add(ref)
        cls._instances -= nonexistent


class Service(Agent):
    '''
    step: 1. Unload finished task from the service (Service._unloadService),
        free up resource for the current time unit.
        2.Predict service utilization (Service._predictUtilization) 
        based on previous utilization level. The result will be used 
        to calculate current cost of one unit of service, which in turn 
        will be used in ResourceSite for estimating offering price. 
    estimate: ResourceSite adjusts capacity (ResourceSite._adjustCapacity) 
        based on Service.resNeedsRequested, which is calculated while 
        loading service (Service._loadService) based on the actual service 
        duration (Service._genActualServiceDuration) in the "advance" step 
        from the previous time unit.
    allocate: Service checks queuelist for overdue tasks and kicks them out. 
        Time in queue estimation is updated (Service.estimTimeInQueue).
    advance: ResourceSite calls Service.loadQueue. Time in queue estimation 
        is updated (Service.estimTimeInQueue). At the same time the 
        service loads service from queue (Service._loadService). 
    '''
    _instances = set()
    avgCapacityPeriod = sp.avgCapacityPeriod
    avgCostPeriod = sp.avgCostPeriod
    utilPredictionWeights = sp.utilPredictionWeights
    capacityBuffer = sp.capacityBuffer
    resProfileUpdateThreshold = sp.resProfileUpdateThreshold
    null = sp.null
    
    def __init__(self,servName,servType,uniqueId,resourceSiteId,sitetype,
                 resCosts,model,maxAmount):
        super().__init__(uniqueId, model)
        self.serviceName = servName
        self.serviceType = servType
        self.unique_id = uniqueId
        self.model = model
        self.availBudget = 0 # dummy variable
        self.rs = resourceSiteId
        self.rsType = sitetype
        if sitetype==SiteType.TYPE2: # slow servers
            self.discount = sp.discount
        else:
            self.discount = 1
        self.maxAmount = maxAmount # server + queue capacity in unit of service
        self.serviceCapacity = maxAmount # simplified
        self.occupied = 0 # nr of occupied service units in queue and server
        self.utilization = 0 # percentage of occupied service capacity
        self.predictedUtil = 0 # predicted utilization of next time step
        self.queuelist = list() # list of bids in queue
        self.servicelist = list() # list of bids being served
        self.estimTimeInQueue = 0 # estim. time to wait in queue for a new task
        self.servicing = False # if any servers are occupied
        self.servicedTaskHistory = list() # history of serviced bids
        self.occupiedHistory = {'0':0} # history of nr. occupied service units
        self.utilHistory = dict() # history of utilization
        self.overCapacityHistory = dict() # history of overcapacity
        self.recentCapaIncrease = 0 # time unit of the most recent increase
        self.recentCapaDecrease = 0 # time unit of the most recent decrease
        self.resNeedsEstim = dict() # slope and intercept of the linear model
        self.resNeedsRandom = dict() # mean and stddev of randomness added 
                                     # to the resNeedsEstim curve
        self.resNeedsAvg = dict() # average amount and duration
        self.resNeedsAvgHistory = dict() # history of the estimated
        self.resNeedsRequested = dict() # requested amount from resource
        self.resNeedsHistory = dict() # history of the actual
        self.resCosts = resCosts # fixed unit costs of resources
        self.costHistory = [0] # service unit cost history
        
        self._createResourceNeeds()

        self._instances.add(weakref.ref(self))

    def _createResourceNeeds(self):
        ''' initiate resource needs '''
        resources = utCom.listClassVariables(ResourceName,'__',False)
        self.resNeedsHistory = dict([(res,
                [DefaultProfile.randomGenerate(self.rsType,self.serviceType)]) 
                for res in resources])
        for k in self.resNeedsHistory.keys():
            self.resNeedsHistory[k][0] = (self.resNeedsHistory[k][0][0],
                self.resNeedsHistory[k][0][1],self.resNeedsHistory[k][0][1])
        self.resNeedsRequested = self._estimateResNeedCurve()
        
    def _estimateResNeedCurve(self):
        ''' estimate resource needs (parameterized and average) of one unit
                of service based on historical (actual) records. Linear
                regression is used and the output is slope and intersect 
                params stored in Service.resNeedsEstim. 
            Around this curve, based on the residuals, randomness is stored
                in Service.resNeedsRandom as the parameters of a normal 
                distribution. The randomness will be added to the estimation
                of resource amounts in Service.estimateServiceDuration and
                Service.estimateResourceAmount, which in turn randomizes 
                the ResourceSite's price offers in 
                ResourceSite._estimTaskDuration and 
                ResourceSite.estimTimeInSystem, called by Bid.estimate.
            The average Service.resNeedsAvg is the median point on this curve.
                the average is used in ResourceSite._adjustCapacity (called 
                from ResourceSite.estimate) to update the resource site's
                resource profile for the service.
            This function is called in Service._genActualServiceDuration
                from Service.advance.
                
        '''
        requested = dict()
        for k in self.resNeedsHistory.keys():
            amountlist = [x[0] for x in self.resNeedsHistory[k]]
            durationlist = [x[1] for x in self.resNeedsHistory[k]]
            if len(amountlist) > 2:
                A = np.vstack([np.array(amountlist),
                               np.ones(len(amountlist))]).T
                try:
                    result = np.linalg.lstsq(A,durationlist,rcond=None)[0]
                except:
                    result = np.zeros(A.shape[1])
                residuals = durationlist - np.dot(A, result)
                self.resNeedsEstim[k] = (result[0],result[1])
                self.resNeedsRandom[k] = (np.mean(residuals),np.std(residuals))
            else:
                self.resNeedsEstim[k] = (np.mean(durationlist)
                                                /np.mean(amountlist),0)     
                self.resNeedsRandom[k] = (0,0)
            
            if len(amountlist) >= self.resProfileUpdateThreshold:
                self.resNeedsAvg[k] = (int(np.median(amountlist)),
                        int(self.resNeedsEstim[k][0] * np.median(amountlist) 
                        + self.resNeedsEstim[k][1]))
            else:
                self.resNeedsAvg[k] = (int(np.mean(amountlist)),
                                              int(np.mean(durationlist)))
            requested[k] = int(self.serviceCapacity * self.resNeedsAvg[k][0])
            if k not in self.resNeedsAvgHistory.keys():
                self.resNeedsAvgHistory[k] = list()
            self.resNeedsAvgHistory[k].append(self.resNeedsAvg[k])
        return requested

    def _genActualServiceDuration(self,resAmounts):
        ''' generate actual service duration given resource amount needed.
                called in Service._loadService, from Service.advance.
                DefaultProfile creates duration based on resource amounts 
                with a normal distribution.
        '''
        resNeedsActual = dict()
        for res in resAmounts.keys():
            duration = DefaultProfile.DURATION[
                        (self.rsType,self.serviceType)](resAmounts[res])
            # actual profile: amount, total duration, duration to go
            resNeedsActual[res] = (resAmounts[res],duration,duration)
            self.resNeedsHistory[res].append(resNeedsActual[res])
        self.resNeedsRequested = self._estimateResNeedCurve()
        return resNeedsActual
    
    def _checkQueuelist(self):
        ''' in case of long waiting time: check queue list for overdue
                tasks and discard them. called in Service.allocate.
        '''
        reduceTime = 0
        prevWaitingTime = 0
        toBeDeleted = list()
        for bidId in self.queuelist:
            bid = self.model.bidlist[bidId]
            if self.model.schedule.steps >= bid.dueTime:
                toBeDeleted.append(bidId)
                bid.chain[bid.currentTask].isInQueue = False
                self.model.sites[self.rs].bidlist.append(bidId)
                self.updateUtilization(
                        -bid.chain[bid.currentTask].serviceAmount)
                reduceTime += (bid.chain[bid.currentTask].estimWaitingTimeTotal 
                               - prevWaitingTime)
            else:
                bid.chain[bid.currentTask].estimWaitingTimeTotal -= reduceTime
            prevWaitingTime = bid.chain[bid.currentTask].estimWaitingTimeTotal
        self.queuelist = [x for x in self.queuelist if x not in toBeDeleted]

    def _predictUtilization(self):
        ''' placeholder for more sophisticated service capacity utilization 
                prediction methods. Currently: simply take the weighted 
                average over a given period of time, weights are specified
                in the config file.
            called in Service.step. result is saved in Service.predictedUtil
                which is used in Service.calculateUnitCost (either called 
                from ResourceSite.estimate or from Service.advance) 
        '''
        periods = range(self.model.schedule.steps - min(len(self.utilHistory),
                        self.avgCostPeriod), self.model.schedule.steps)
        weights = [self.utilPredictionWeights[self.model.schedule.steps
                        -1-int(k)] for k,_ in self.utilHistory.items() 
                        if int(k) in periods]
        if sum(weights)>0:
            weights = [x/sum(weights) for x in weights]
        value = [np.mean(v) for k,v in self.utilHistory.items() 
                if int(k) in periods]
        self.predictedUtil = sum([x*y for x,y in zip(value,weights)])
            
    def _loadService(self):
        ''' load service from queue to server for processing. 
                function is called in Service.advance. create real resource
                needs with randomness, and update Task.resProfileActual of
                the current task.
            Currently, if the next-service-in-line cannot be served by the
                server due to resource shortage, the entire service queue
                waits until resources can be freed. However this will be 
                recorded, and will be used in ResourceSite._adjustCapacity
                to request for a increase in resource allocation to the
                service.
        '''
        while (len(self.servicelist) < self.serviceCapacity 
                                        and len(self.queuelist) > 0):        
            # unload queue:
            bidId = self.queuelist[0]
            bid = self.model.bidlist[bidId]
            task = bid.chain[bid.currentTask]
            
            resAmounts = utCom.recreateDict(task.resProfileSelected,0)
            totalAmounts = dict([k,resAmounts[k] * task.serviceAmount]
                                    for k in resAmounts.keys())
            if task.resProfileActual is None:
                task.resProfileActual = self._genActualServiceDuration(
                                                                resAmounts)
            # resource shortage at time of loading
            feasible,diffAmount = self.model.sites[
                    self.rs].checkResFeasibility(totalAmounts,
                    self.serviceName,self.serviceType)
            if not feasible:
                self.overCapacityHistory[str(self.model.schedule.steps)] = (
                        self.occupied, 
                        self.maxAmount - self.occupied - task.serviceAmount)
                return
            
            # allocate resource
            for res in resAmounts.keys():
                self.model.sites[self.rs].resourcelist[res].startTask(
                            task.unique_id,self.serviceName,self.serviceType,
                            task.resProfileActual[res][0] * task.serviceAmount,
                            task.resProfileActual[res][1],
                            self.model.schedule.steps)
            
            # update queue list
            self.queuelist = self.queuelist[1:]
            self.servicelist.append(bidId)
            self.servicing = True
            # update current task
            task.queueEnd = self.model.schedule.steps
            actualServiceTime = max([task.resProfileActual[res][1] 
                                    for res in task.resProfileActual.keys()])
            task.updateServiceTime(actualServiceTime,self.model.schedule.steps)
            unitCost = self.calculateUnitCost(task.resProfileActual,
                                        prediction=False)
            task.cost = unitCost * task.serviceAmount
            self.costHistory.append(unitCost)
                                
    def _unloadService(self):
        ''' end of service for a task when any task being served reaches 
                end of its service time.
            this function is called from Service.step.
        '''
        for bidId in self.servicelist:
            bid = self.model.bidlist[bidId]
            task = bid.chain[bid.currentTask]
            task.serviceTimeToGo -= 1
            release = task.updateResourceTime()
            if len(release)>0:
                for res in release:
                    self.model.sites[self.rs].toBeReleased[res].append(
                                                                task.unique_id)
            if task.serviceTimeToGo <= 0:
                # update task
                task.serviceEnd = self.model.schedule.steps
                task.isFinished = True
                # update resource
                for res in task.resProfileActual.keys():
                    self.model.sites[self.rs].toBeReleased[res].append(
                                                                task.unique_id)
                # update service
                self.servicedTaskHistory.append(bidId)
                self.servicelist.remove(bidId)
                self.updateUtilization(-task.serviceAmount)
                if len(self.servicelist)==0:
                    self.servicing = False                
                # update bid
                bid.priceLeftover -= task.cost                              
                if (task.pos==len(bid.chain)-1) or (bid.priceLeftover<0):
                    bid.updateResult(BidStatus.FINISH,
                                self.model.schedule.steps,
                                self.model.schedule.steps-bid.batchCreateTime)
                else:
                    bid.currentTask += 1
                    bid.updateResult(BidStatus.REALLOCATE,
                                     self.model.schedule.steps)
                    
    def _estimateTimeInQueue(self):
        '''
        update estimated time in queue for a new task coming into the queue.
            called by Service.updateUtilization.
        '''
        resAmount = utCom.recreateDict(self.resNeedsAvg,0)
        self.estimTimeInQueue = (self.occupied
                              * self.estimateServiceDuration(resAmount) 
                              / self.serviceCapacity)
    
    def loadQueue(self,taskId,bidId,serviceAmount,resAmounts):
        ''' load newly arrived tasks into service queue. called by 
                ResourceSite._load from ResourceSite.advance.
        '''
        # service queue overflow at time of loading
        if self.occupied + serviceAmount > self.maxAmount:
            time = self.model.schedule.steps
            self.overCapacityHistory[str(time)] = (self.occupied, 
                        self.maxAmount - self.occupied - serviceAmount)
            return None,None
        queuetime = self.estimTimeInQueue
        self.updateUtilization(serviceAmount)
        self.queuelist.append(bidId)
        return self.unique_id,queuetime

    def estimateServiceDuration(self,resAmounts):
        ''' estimate duration for one time service, given resource profile '''
        result = np.max([1] + [self.resNeedsEstim[k][0] * resAmounts[k] 
                       + self.resNeedsEstim[k][1] 
                       + np.random.normal(*self.resNeedsRandom[k])
                       for k in self.resNeedsEstim.keys()])
        return result
    
    def estimateResourceAmount(self,duration):
        ''' estimate resource profile for quotes, given QoS '''
        duration = max(1,duration)
        result = dict()
        for k in self.resNeedsEstim.keys():
            if self.resNeedsEstim[k][0]==0: # if slope is 0
                result[k] = (1,self.resNeedsEstim[k][1])
            else:
                result[k] = (max( 1,int(np.round(
                        (duration - np.random.normal(*self.resNeedsRandom[k])
                         - self.resNeedsEstim[k][1]) 
                         / self.resNeedsEstim[k][0])) ), 
                         int(np.round(duration,0)))
        return result      
    
    def calculateUnitCost(self,resNeeds,prediction=True):
        ''' estimates cost of one unit of service based on the service 
                capacity utilization. 
            called by ResourceSite.getServiceInfo from ResourceSite.estimate 
                for estimation of the service cost, and by 
                Service._loadService from Service.advance with actual 
                resource requirements and no prediction method. 
            
            @param resNeeds: amount and duration of resources needed for the
                service. cost is based on estimated required resource 
                amount and duration of one unit of service. 
            @param prediction: if true, use predicted utilization based on 
                the timeseries. if false, use the most recent utilization 
                value. 
            @return: cost for one unit of service.
        '''
        if self.utilization == 1:
            return sys.maxsize
        else:
            if prediction:
                utilization = self.predictedUtil
            else:
                utilization = self.utilization
            return self.discount * utilization * sum([self.resCosts[res] 
             * resNeeds[res][0] * resNeeds[res][1] for res in resNeeds.keys()])

    def updateUtilization(self,serviceAmount,diffMax=0):
        '''
        update service capacity utilization after Service.loadQueue from
            ResourceSite._load (ResourceSite.advance), Service._unloadService 
            from Service.step, ResourceSite._adjustCapacity from 
            ResourceSite.estimate.
        '''
        key = str(self.model.schedule.steps)
        self.occupied = np.max([0,self.occupied + serviceAmount])
        self.occupiedHistory[key] = self.occupied
        self.maxAmount = max(self.occupied, self.maxAmount + diffMax)
        self.utilization = self.occupied / self.maxAmount
        
        if key not in self.utilHistory.keys():
            self.utilHistory[key] = list()
        self.utilHistory[key].append(self.utilization)
        self._estimateTimeInQueue()
        self.serviceCapacity = max(1, self.serviceCapacity + diffMax)
        if diffMax > 0:
            self.recentCapaIncrease = self.model.schedule.steps
        elif diffMax < 0:
            self.recentCapaDecrease = self.model.schedule.steps

    def step(self):
        self._unloadService()
        self._predictUtilization()

    def estimate(self):
        for bidId in self.queuelist:
            bid = self.model.bidlist[bidId]
            task = bid.chain[bid.currentTask]
            task.estimWaitingTimeToGo -= 1
        
    def allocate(self):
        self._checkQueuelist()
    
    def advance(self):
        self._loadService()

    @classmethod
    def getInstances(cls):
        nonexistent = set()
        for ref in cls._instances:
            obj = ref()
            if obj is not None:
                yield obj
            else:
                nonexistent.add(ref)
        cls._instances -= nonexistent
        

class RSU(Agent):
    '''
    step:.
    estimate: 1. update nrVehicles and nrBids. 
        2. update resource sites' free capacity
    allocate: 1. allocate bids to resource sites
        2. calculate payment and feed back to vehicles
        3. update competitor states in TransitionTbl for learning. The state
        information will be consolidated in TransitionTbl._addStateRecord in 
        TransitionTbl.advance as the actual state, which is the target output
        for previous input to CompetitorLearningModel, as well as the
        nextStateVec for PriceLearningModel's critic module.
    advance:.
    '''
    secondPriceThres = rp.secondPriceThres
    overload = rp.overload
    _instances = set()
    def __init__(self,uniqueId,siteMat,model):
        super().__init__(uniqueId, model)
        self.unique_id = uniqueId
        self.pos = (0,0) # coordinate of the RSU
        self.availBudget = 0 # dummy variable
        self.bids = list() # bid ids submitted by vehicles to the RSU
        self.sortedBids = dict() # sorted bids according to profitability
        self.sites = list() # connected resource site ids
        self.homesite = None # directly connected resource site id
        self.vehicles = list() # vehicles connected to the RSU
        self.siteMat = siteMat # information of all resource sites
        self.siteDist = dict() # distance to all resource sites
        self.siteFreeCapa = dict() # estimation of resource site free capa
        self.siteMaxAmounts = dict() # record site max amount
        self.nrVehicles = 0 # nr of connected vehicles
        self.nrActiveBids = 0 # nr of active bids on the RSU
        self.nrActiveBidsBeforeAllocateHistory = dict() # for debugging
        self.nrActiveBidsAfterAllocateHistory = dict() # for debugging
        self.secondPriceRecords = list() # records eligible for calculation 
                                         # of second price
        self.secondPrices = dict() # records second price by category
        self.secondPriceHistory = list()
        self.secondPriceParams = None # curve for 2.price
        self.bidRecordedForPrice = list() # for debugging
        
        self._instances.add(weakref.ref(self))
        
    def _findResourceSites(self):
        ''' placeholder for finding resource sites dynamically '''
        self.sites = self.model.siteId
        self.secondPriceParams = np.zeros(len(self.sites)+len(sec.keys)+1)
    
    def _getSiteDistance(self,siteMat):
        '''
        look up the distance table from resource site information. 
        @ param siteMat: matrix of distance between sites
        '''
        try:
            homesite = self.sites[1] # standard server
        except:
            homesite = self.sites[0]
        self.siteDist = siteMat.distanceMatrix
        for s in self.sites:
            self.siteDist[(self.unique_id,s)] = self.siteDist[(homesite,s)]
        self.homesite = homesite

    def _sortBids(self):
        ''' sort active bids by prio and profitability. called by
                RSU.allocate
        '''
        sortedBids = [(x.unique_id,(x.priority,x.priceLeftover 
                                    - x.bestProposal.proposedBidCost)) 
                    for x in self.model.bidlist.values() if x.isActive==1]
        sortedBids = dict(sortedBids)
        return {k: v for k,v in sorted(sortedBids.items(),
                                        key=lambda item:item[1],reverse=True)}
    
    def _populateSecondPriceRecord(self,bid,siteId,target=None):
        ''' helper function to create second price records. used in 
                RSU._recordSecondPrice, and RSU.allocate for inputs to 
                RSU._calculatePayment. Container is a SecondPriceData class.
        '''
        record = sec(self.sites).record
        record[sec.TIME] = self.model.schedule.steps
        record[sec.QOS] = bid.dueTime - self.model.schedule.steps
        record[sec.VEH] = self.nrVehicles
        record[sec.BID] = self.nrActiveBids
        if target is None:
            record[sec.ESTIM] = bid.priceLeftover
        else:
            record[sec.ESTIM] = target
        record[siteId] = 1
        for i in np.arange(start=bid.currentTask,stop=len(bid.chain)):
            t = bid.chain[i]
            servNameType = (t.serviceName,t.serviceType)
            record[servNameType] = t.serviceAmount
        return record
    
    def _recordSecondPrice(self,bid,siteId,target=None):
        '''
        based on all bid records eligible for second price calculation (i.e.
            the bids which did not get allocated by are next-in-line), 
            fit a curve from bids' qos requirements, nr active bids in env, 
            amount in unit of service, etc. to the estimated second price. 
            a linear regression model is used. params are stored in 
            RSU.secondPriceParams which is used in RSU._calculatePayment
            called from RSU.allocate.
        @param bid: bid object for the second price record. note that within
            the same class, objects can be passed on to functions. Across
            classes the objects need to be accessed directly from the mesa 
            model, or the copies of objects are not sync'd.
        '''
        self.bidRecordedForPrice.append(bid.unique_id)
        record = self._populateSecondPriceRecord(bid,siteId,target=target)
        if record not in self.secondPriceRecords:
            self.secondPriceRecords.append(record)
            if len(self.secondPriceRecords) > self.secondPriceThres:
                arr = np.array([list(x.values()) 
                                        for x in self.secondPriceRecords])
                xcol = np.hstack([arr[:,:-1],
                            np.ones(arr.shape[0]).reshape(arr.shape[0],-1)])
                ycol = arr[:,-1]
                try:
                    params = np.linalg.lstsq(xcol,ycol,rcond=None)[0]
                    self.secondPriceParams = params
                except:
                    pass
    
    def _calculatePayment(self,serviceRecord):
        ''' calculate second price payment based on linear regression of 
                service needs, env. variables, etc. 
            the function is called from RSU.allocate.
            @param serviceRecord: a dictionary containing bid info, for which
                the payment needs to be calculated.
            @return: second price for the given bid.
        '''
        if len(self.secondPriceRecords) <= self.secondPriceThres:
            denom = 0
            total = 0
            for key in serviceRecord:
                if key in sec.services:
                    denom += sum([x[key] for x in self.secondPriceRecords])
                    total += serviceRecord[key]
            if denom<=0:
                unit = 0
            else:
                unit = sum([x[sec.ESTIM] 
                                for x in self.secondPriceRecords]) / denom
            return max(0, unit * total)
        else:
            result = np.dot(np.array(list(serviceRecord.values()))[:-1],
                self.secondPriceParams[:-1]) + self.secondPriceParams[-1]
            return max(0, result)

    def _rejectBids(self,bid,rejectReason):
        ''' helper function to include steps when bids are rejected '''
        bid.updateResult(BidStatus.REJECT,self.model.schedule.steps,
                         reason=rejectReason)
        self.removeBids(bid.siteBinding,bid.unique_id)
        # calculate reward and record state info in case of rejection:
        self.model.vehicles[bid.user].deactivateBid(bid.unique_id)
            
    def step(self):
        if len(self.sites)==0:
            self._findResourceSites()
            self._getSiteDistance(self.siteMat)

    def estimate(self):
        self.nrVehicles = sum([self.model.vehicles[x].isActive 
                                               for x in self.vehicles])
        self.nrActiveBids = sum([self.model.bidlist[x].isActive 
                                                 for x in self.bids])
        self.nrActiveBidsBeforeAllocateHistory[
                        str(self.model.schedule.steps)] = self.nrActiveBids
        for siteId in self.model.sites.keys():
            site = self.model.sites[siteId]
            delay = max(0,self.model.schedule.steps - int(np.ceil(
                    self.siteMat.getWiredLatency(self.homesite,siteId,8))))
            for serviceNameType in site.servicelist.keys():
                # add the effect of data transmission delay
                try:
                    maxAmount = site.serviceMaxAmountHistory[
                                                    serviceNameType][delay]
                except:
                    maxAmount = site.maxAmounts[serviceNameType]
                try:
                    currentOccupied = site.serviceOccupiedHistory[
                                                    serviceNameType][delay]
                except:
                    currentOccupied = site.currentOccupied[serviceNameType]
                self.siteFreeCapa[(siteId,serviceNameType)] = (
                        maxAmount * (1 + self.overload) - currentOccupied)
                self.siteMaxAmounts[(siteId,serviceNameType)] = (
                                            maxAmount * (1 + self.overload))
    
    def allocate(self):
        '''
        if bids dont have feasible quotes from any resource sites, or if 
            one of the duplicate bids are already accepted, or if the bid is
            down-prioritized and there is no free capacity left in the
            resource sites.
        if a bid is accepted and allocated, it is appended to the resource
            site's bidlist. a payment is also calculated for the bid. all 
            results are fed back to the vehicle. payment as a indicator of
            competitor states is updated in corresponding TransitionTbl.
        '''
        logfile = self.model.filedict[FILES.ALLOC_FILE[0]] 
        self.sortedBids = self._sortBids()
        acceptedBids = list()
        recordedForSecondPrice = dict()
        secondPrices = dict()
        for s in self.sites:
            for k in cbd.keys:
                recordedForSecondPrice[(s,k)] = False
                secondPrices[(s,k)] = 0
        counter = -1
        
        for bidId in self.sortedBids.keys():            
            bid = self.model.bidlist[bidId]
            counter += 1
            
            def myLogFunc(decision):
                if bid.bestProposal.proposedDest is None:
                    sitetype = None
                else:
                    sitetype = self.model.sites[
                                    bid.bestProposal.proposedDest].sitetype
                if decision=='skipped':
                    return
                nrRebid = bp.nrRebid
                logfile.write(
                    '{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}\n'.format(
                    self.model.schedule.steps,len(self.sortedBids),counter,
                    bidId,bid.status,bid.learningKey,bid.qostype,
                    bid.price,bid.payment,bid.availBudget,bid.batchCreateTime,
                    bid.createTime,bid.bestProposal.proposedDest,sitetype,
                    bid.finishTime,nrRebid-bid.nrRebid,decision,bid.value))
                
            # ignore those in progress
            if bid.status in (BidStatus.PREP,BidStatus.PROCESS,
                              BidStatus.TRANSMIT):
                myLogFunc('skipped')
                continue
            # reject duplicate bids if another bid in the same batch 
            # is admitted. This is used when multiple bids are created under
            # the same batch number so that there are many choices for the RSU
            # but only one of them can be accepted and allocated.
            if bid.batch_id in acceptedBids: # the other qos is accepted
                self._rejectBids(bid,rejectReason=bfr.DUPLICATE)
                myLogFunc('same batch duplicate rejected')
                continue
            
            serviceNameType = (bid.chain[bid.currentTask].serviceName,
                               bid.chain[bid.currentTask].serviceType)
            serviceAmount = bid.chain[bid.currentTask].serviceAmount
            siteId = bid.bestProposal.proposedDest
            learningKey = (bid.learningKey,bid.qostype)
            
            if siteId is not None:
                # reject bids with low priority when the total capacity 
                # of the resource site is exceeded. Test all proposals before 
                # rejection:
                if (self.siteFreeCapa[(siteId,
                                       serviceNameType)]-serviceAmount<0):
                    for proposal in sorted(bid.proposals.values(),
                                           reverse=True):
                        sid = proposal.proposedDest
                        if (self.siteFreeCapa[(sid,serviceNameType)]
                            -serviceAmount>=0 
                            and proposal.proposedBidCost<=bid.priceLeftover):
                            bid.bestProposal = proposal
                            siteId = sid
                            break
                if self.siteFreeCapa[(siteId,serviceNameType)]-serviceAmount<0:
                    if (not recordedForSecondPrice[(siteId,learningKey)] 
                        and bid.priority==0):
                        secondPrices[(siteId,
                                      learningKey)] = bid.priceLeftover
                        self._recordSecondPrice(bid,siteId)
                        recordedForSecondPrice[(siteId,learningKey)] = True
                    self._rejectBids(bid,rejectReason=bfr.NOPRIO)
                    myLogFunc('down prioritized rejected')
                    continue
                
                bidStatus = dict(Counter([x[0] for x in bid.statusHistory]))                
                if (BidStatus.ADMIT not in bidStatus or 
                    bidStatus[BidStatus.ADMIT]<bidStatus[BidStatus.PREP]):
                    bid.updateResult(BidStatus.ADMIT,self.model.schedule.steps)
                if bid.siteBinding != siteId:
                    self.removeBids(bid.siteBinding,bidId)
                    self.appendBids(siteId,bidId)

                bid.acceptProposal(siteId=siteId,rsuId=self.unique_id,
                                   rsuHomesiteId=self.homesite)
                self.siteFreeCapa[(siteId,serviceNameType)] -= serviceAmount
                
                serviceRecord = self._populateSecondPriceRecord(bid,siteId)
                if bid.payment==-1: # first time allocated
                    bid.payment = min(bid.price-1,
                                  self._calculatePayment(serviceRecord))
                    self.model.vehicles[bid.user].rewardBidlist.append(bidId)                    
                    transTbl = self.model.vehicles[bid.user].transitionTbl
                    if transTbl is not None:
                        self.model.transitionTbl[
                                            transTbl].updateStateInfo(bidId)

                acceptedBids.append(bid.batch_id)
                myLogFunc('admitted')
                
            else:
                self._rejectBids(bid,rejectReason=bfr.NO_AVAILABLE_SITE)
                myLogFunc('no proposals rejected')
            
        for siteId in self.sites:
            recorded = False
            for key in cbd.keys:
                if (len(self.sortedBids)>0 
                    and not recordedForSecondPrice[(siteId,key)]):
                    secondPrices[(siteId,key)] = 0
                    if not recorded:
                        self._recordSecondPrice(bid,siteId,target=0)   
                        recorded = True
        self.secondPrices = secondPrices.copy()
        self.secondPriceHistory.append(self.secondPrices)
        
    def advance(self):
        self.nrActiveBidsAfterAllocateHistory[
                        str(self.model.schedule.steps)] = sum(
                        [self.model.bidlist[x].isActive for x in self.bids])
    
    def removeBids(self,siteId,bidId):
        ''' helper function to remove the given bidId from the given site '''
        if siteId != self.unique_id:
            self.model.sites[siteId].bidlist.remove(bidId)
            
    def appendBids(self,siteId,bidId):
        ''' helper function to append given bidId to the given site '''
        self.model.sites[siteId].bidlist.append(bidId)
        
    @classmethod
    def getInstances(cls):
        nonexistent = set()
        for ref in cls._instances:
            obj = ref()
            if obj is not None:
                yield obj
            else:
                nonexistent.add(ref)
        cls._instances -= nonexistent
        

class Vehicle(Agent):
    '''
    step: 1. activate bids in the pipeline (put into the pipeline in the
        previous time step in Vehicle.estimate). 
        2. periodically re-train price learning model (when nr of new inputs
        exceeds a given threshold).
    estimate: in Vehicle._fillPipeline, 
        1. create new bids which share the same budget.
        2. run competitor learning model's inference function to
        get competitor's next state.
        3. run price learning model's inference function to get budget 
        split between bids, assign bidding price to each bid, and put the 
        bids into pipeline for activation in next time step.
    allocate: RSU.allocate will make decisions to allocate or reject bids,
        calculate payment for each, and record state information.
    advance: 1. if TransitionTbl is not yet created, create and populate 
        the TransitionTbl. in TransitionTbl.advance, actual state information
        from RSU.allocate will be consolidated to create one record of 
        nextStateVec, which is used both as output target for the supervised 
        learning in TransitionTbl.competitorMdl, and as the next state value
        in Vehicle.priceMdl.critic. 
        2. calculate bids' rewards and add to PriceLearningModel.
    '''
    _instances = set()
    # randomness to each vehicle's maximum budget
    minNrBids = vp.minNrBids
    maxNrBids = vp.maxNrBids
    totalNrBids = vp.totalNrBids
    maxSensingDist = vp.maxSensingDist
    budgetRenewal = vp.budgetRenewal    
    qostypes = cbd.qosTypes
    servAmount = cbd.serviceAmounts
    competitorDataThres = vp.competitorDataThres
    plmTrainingThres = vp.plmTrainingThres
    lamMax = vp.lamMax
    lamMin = vp.lamMin
    lamChangeInterval = vp.lamChangeInterval
    lamCategory = vp.lamCategory
    ph = vp.ph
    pl = vp.pl
    stagingMaxsize = vp.stagingMaxsize
    stagingMaxtime = vp.stagingMaxtime
    stagingThreshold = np.random.choice(vp.stagingThreshold)
    stagingPeriod = vp.stagingPeriod

    def __init__(self,uniqueId,createTime,model,totalBudget=None,rebid=False):
        super().__init__(uniqueId, model)
        self.unique_id = uniqueId
        if totalBudget is None:
            self.totalBudget = np.random.choice([vp.totalBudget[0],
                                                 vp.totalBudget[1]])
        else:
            self.totalBudget = totalBudget
        self.maxLifespan = np.random.randint(vp.lifespan[0],vp.lifespan[1])
        self.pos = (0, 0)
        self.createTime = createTime
        self.availBudget = 0 # dummy variable
        self.availRsu = list()
        self.rsu = None # connected rsu
        self.bidCount = 0 # for generating unique ids for the bids
        self.bidPipeline = list() # bids which are to be activated
        self.bidStaging = list() # bids which are not to be activated
        self.lam = 0
        self._updateLambda()
        # for repeatability
        self.countdownHistory = list()
        self.batchCreateHistory = list()
        self.saveRecord = True
        if self.model.repeatable:
            _ = self._loadRepeatable()
        # for random creation of bid batches:
        self.countdown = self._generateCountdown()
        self.activeBid = list() # bids which are active
        # temporary list to record rewards in self._activateBids
        self.rewardBidlist = list() # bids to be collected rewards
        self.rebidlist = list() # bids to be re-bid
        self.freeCapa_assigned = dict() # free capacity in each site for each 
                                        # service as input state information
        self.freeCapa_all = dict() # free capacity from each rsu estimate
        self._initFreeCapaInfo()
        self.bidHistory = list() # history of finished bids
        self.bids_random = pd.DataFrame()
        self.tasks_random = pd.DataFrame()
        self.rebid = rebid
        self.stagingDistr = 'uniform'
        
        #time to stay in the system, not used:
        self.availLifespan = self.maxLifespan
        self.isActive = True # indicator, not used
        # object to hold all competitor learning model related information:
        self.transitionTbl = None 
        # for normalization of environment variable (total nr.bids)
        self.benchmarkNrBid = (vp.budgetRenewal+vp.minNrBids) * mp.initVehicles
        
        # output of the price learning model: prediction of best response
        # 1st digit is whether to 
        # activate bid in the current time unit. rest of output is 
        # proportion of allocated budget to each bid.
        self.priceMdl = plm(uniqueId=self.unique_id,
                            dimOutput=1+self.budgetRenewal,
                            evaluation=self.model.evaluation,
                            loadModel=self.model.loadModel)
        
        # output of the supervised learning: prediction of own behavior
        self.superviseMdl = spv(uniqueId=self.unique_id,
                                dimOutput=1+self.budgetRenewal,
                                evaluation=self.model.evaluation,
                                loadModel=self.model.loadModel)
        self.learningDataPos = str(0) # connection key to link input records
                                      # to corresponding rewards and outputs
        self.learningDataPosSL = str(0) # for supervised learning model
        self.prevRecordNr = 0 # used to determine if to re-train the priceMdl.
        
        self._instances.add(weakref.ref(self))
    
    def _loadRepeatable(self):                
        try:
            filename0 = COUNTDOWN_FILE + '_' + self.unique_id + '.pkl'
            self.countdownHistory = pickle.load(open(filename0,'rb'))
            filename1 = BATCHCREATE_FILE + '_' + self.unique_id + '.pkl'
            self.batchCreateHistory = pickle.load(open(filename1,'rb'))
            self.saveRecord = False
            return (filename0,filename1)
        except:
            pass        
        try:
            locations = [x.span() for x in re.finditer('_',self.unique_id)]
            uid = (self.unique_id[:locations[-1][1]] 
                            + str(not self.model.trainable))
            filename0 = COUNTDOWN_FILE + '_' + uid + '.pkl'
            self.countdownHistory = pickle.load(open(filename0,'rb'))
            filename1 = BATCHCREATE_FILE + '_' + uid + '.pkl'
            self.batchCreateHistory = pickle.load(open(filename1,'rb'))
            self.saveRecord = False
            return (filename0,filename1)
        except:
            return (None,None)
    
    def _generateCountdown(self):
        if self.model.repeatable and not self.saveRecord:
            try:
                record = self.countdownHistory.pop(0)
                return record
            except:
                pass
        # average interval of bid creation is 1/lambda time steps
        self.saveRecord = True
        record = int(np.random.exponential(scale=1/self.lam))
        self.countdownHistory.append(record)
        return record

    def _updateLambda(self):
        if self.lam==0:
            self.lam = self.lamMin
            return

        avg = (self.lamMax + self.lamMin) / 2
        band = (self.lamMax - self.lamMin) / self.lamCategory
        prob = np.random.rand()
        if self.lam >= avg:
            if prob<=self.ph:
                self.lam = np.random.uniform(low=self.lamMin,
                                             high=self.lamMin+band)
        else:
            if prob<=self.pl:
                self.lam = np.random.uniform(low=self.lamMax-band,
                                             high=self.lamMax)
    
    def _initFreeCapaInfo(self):
        for rsu in self.model.rsu.keys():
            for siteId in self.model.sites.keys():
                for servKey in self.model.sites[siteId].servicelist.keys():
                    self.freeCapa_assigned[(rsu,siteId,servKey)] = 1
            self.freeCapa_all[rsu] = 1
    
    def _createBid(self,bidId=None,qos=None,servAmountLevel=None,
                   savedRecord=None):
        ''' called by Vehicle._fillPipeline in Vehicle.estimate. randomly
                create a batch of bids in different 
                (service amount,qos requirement) categories, which share
                the same budget.
        '''
        if bidId is None:
            bidId = str(self.unique_id) + '_' + str(self.bidCount)
        if qos is None:
            qos = np.random.choice(self.qostypes)
        if servAmountLevel is None:
            servAmountLevel = np.random.choice(self.servAmount)

        bid = Bid(bidId=bidId,qostype=qos,servAmount=servAmountLevel,
                  user=self.unique_id,
                  budget=self.totalBudget/self.budgetRenewal,
                  createTime=self.model.schedule.steps,
                  coord=self.pos,model=self.model,savedRecord=savedRecord)
        self.bidCount += 1

        if (bid.dueTime < self.createTime + self.maxLifespan
            and bid.dueTime < self.model.totalSimTime):
            bid.rsu = self.rsu
            bid.environment = self.model.rsu[self.rsu].nrActiveBids
            return bid
        else:
            return None
    
    def _calDistance(self,otherCoord):
        ''' placeholder for finding RSUs. not used. '''
        return np.sqrt((otherCoord[0] - self.pos[0])^2 
                       + (otherCoord[1] - self.pos[1])^2)
    
    def _findRSU(self):
        ''' placeholder for finding RSUs. not used. '''
        self.availRsu = list()
        for obj in RSU.getInstances():
            self.availRsu.append(obj.unique_id)
    
    def _chooseRSU(self):
        ''' placeholder for finding RSUs. not used. '''
        if len(self.availRsu)>0:
            self.rsu = self.availRsu[0]
            self.model.rsu[self.rsu].vehicles.append(self.unique_id)
            
    def _createTransitionTbl(self):
        ''' called in Vehicle.advance to create object for 
                CompetitorLearningModel.
        '''
        transitionTbl = TransitionTbl(vehicleId=self.unique_id,
            budget=self.totalBudget/self.budgetRenewal,
            benchmarkNrBid=self.benchmarkNrBid,
            model=self.model,clmfile=self.model.filedict[FILES.CLM_FILE[0]])
        self.transitionTbl = transitionTbl.unique_id
        self.model.schedule.add(transitionTbl)
        self.model.transitionTbl[self.transitionTbl] = transitionTbl
    
    def _fsp(self,bestResponse,behavior,const=50):
        mixingParam = const / (self.model.schedule.steps + const)
        randomNr = np.random.rand()
        if randomNr <= mixingParam:
            return bestResponse
        else:
            return behavior
    
    def _fillPipeline(self):
        '''
        1. create new bids which share the same budget.
        2. run competitor learning model's inference function to
        get competitor's next state.
        3. run price learning model's inference function to get budget 
        split between bids, assign bidding price to each bid, and put the 
        bids into pipeline for activation in next time step.
        
        the function is called in Vehicle.estimate.
        '''
        if self.countdown<0: # create new batch with given avg. interval
            self.countdown = self._generateCountdown()
            if not self.saveRecord:
                try:
                    batchNew = self.batchCreateHistory.pop(0)
                except:
                    batchNew = list()
                    self.saveRecord = True
            else:
                batchNew = list()
            # bidVector is a onehot matrix indicating which bid belongs
            # to which category.
            bidVectorNew = list()
            batchRecord = list()
            for i in range(self.budgetRenewal):
                if self.saveRecord:
                    bid = self._createBid()
                    batchNew.append(bid)
                    if bid is None:
                        batchRecord.append(None)
                    else:
                        batchRecord.append(bid.savedRecord)
                else:
                    savedRecord = batchNew[i]
                    if savedRecord is None:
                        bid = None
                    else:
                        bidId = (savedRecord['user'] 
                                    + str(self.model.trainable) 
                                    + '_' + savedRecord['batch'])
                        bid = self._createBid(bidId=bidId,
                                    qos=savedRecord['qostype'],
                                    servAmountLevel=savedRecord['servAmount'],
                                    savedRecord=savedRecord)
                    batchNew[i] = bid
                    
                bidInfo = {k:0 for k in cbd.keys}
                taskSeq = [0] * len(cbd.services)
                if bid is not None:    
                    bidInfo[(bid.learningKey,bid.qostype)] += 1
                    taskSeq = bid.taskSeq.values()
                bidVectorNew += list(bidInfo.values())
                bidVectorNew += taskSeq
            if self.saveRecord:
                self.batchCreateHistory.append(batchRecord)
            if sum(bidVectorNew)>0:        
                for i in range(len(batchNew)):
                    bid = batchNew[i]
                    if bid is None:
                        batchNew[i] = (None,0)
                        continue
                    self.model.schedule.add(bid)
                    self.model.bidlist[bid.unique_id] = bid
                    batchNew[i] = (bid.unique_id,bid.qosRequired)
                self.bidStaging.append((self.model.schedule.steps,
                                    batchNew,bidVectorNew,
                                    self.model.schedule.steps))
        
        if len(self.bidStaging)==0:
            return
        toRemove = []
        toDeactivate = []
        normalizedStagingSize = len(self.bidStaging) / self.stagingMaxsize

        if (self.transitionTbl is not None and 
            len(self.model.transitionTbl[
                                self.transitionTbl].estimNextState)>0):
            competitorState = self.model.transitionTbl[
                                self.transitionTbl].estimNextState
        else:
            competitorState = list(np.random.rand(
                                len(self.model.sites) * len(cbd.keys)))

        for i in range(len(self.bidStaging)):
            createTime, batch, bidVector, nextChance = self.bidStaging[i]
            if nextChance > self.model.schedule.steps:
                continue
            dueTime = max([self.model.bidlist[bidId].dueTime 
                                           for (bidId,qos) in batch])
            if dueTime>0 and dueTime <= self.model.schedule.steps:
                toDeactivate.append(self.bidStaging[i])
                continue
            
            normalizedNrBids = (self.model.rsu[self.rsu].nrActiveBids 
                                / self.benchmarkNrBid)
            normalizedStagingTime = (self.model.schedule.steps 
                                     - createTime) / self.stagingMaxtime
            # for supervised learning
            inputVecSL = (bidVector + [normalizedNrBids,
                        normalizedStagingTime,normalizedStagingSize,self.lam]
                        + list(self.freeCapa_all.values()) 
                        + list(self.freeCapa_assigned.values()))
            inputVec = competitorState + inputVecSL
            
            self.learningDataPos = self.priceMdl.collectInput(inputVec)
            if self.model.evaluation:
                randomness = 0
            else:
                randomness = None
            # best response \beta_{j+1} from reinforcement learning
            bestResp = self.priceMdl.inference(inputVec,randomness).tolist()
            
            # average behavior record for supervised learning
            self.learningDataPosSL = self.superviseMdl.collectInput(inputVecSL)
            behavior = self.superviseMdl.inference(inputVecSL).tolist()
            result = self._fsp(bestResp,behavior)
            
            # best response behavior record for supervised learning
            self.superviseMdl.collectBehavior(result,self.learningDataPosSL)
            self.learningDataPosSL = self.superviseMdl.collectInput(inputVecSL)
            self.superviseMdl.collectBehavior(bestResp,self.learningDataPosSL)
            
            if self.model.trainable:
                act = result[0]
            else: 
                act = 0
            try:
                prices = [x * self.totalBudget / self.budgetRenewal 
                      for x in result[1:]]
            except:
                prices = [x * self.totalBudget / self.budgetRenewal 
                      for x in result[1:]]
            savings = [self.totalBudget / self.budgetRenewal - x 
                       for (x,y) in zip(prices,batch) if y[0] is not None]
            stagingRandom = self._getAction()
            if stagingRandom<act: # decision to hold bids
                self.priceMdl.collectOutput(result,self.learningDataPos)
                # nextStateVec is the same as current state
                self.priceMdl.collectNextState(competitorState,
                                               self.learningDataPos)
                stagingPeriod = (1-act) * self.stagingPeriod
                # loss per time unit delayed is porportional to the budget
                qos = [qos for bidId,qos in batch if bidId is not None]
                avg = self.totalBudget / len(qos)
                delay = (self.model.schedule.steps + 1 - createTime 
                         + stagingPeriod)
                reward = -sum([avg / x * delay for x in qos])
                
                self.priceMdl.collectReward(reward/10,self.learningDataPos)
                
                self.bidStaging[i] = (createTime, batch, bidVector, 
                                      nextChance + stagingPeriod)
                continue
            self.priceMdl.collectOutput(result,self.learningDataPos)
            self.priceMdl.collectNextState(competitorState,
                                           self.learningDataPos) #placeholder
            
            output = dict([(y[0],x) for (x,y) in zip(prices,batch) 
                                                        if y[0] is not None])
            
            for bidId,_ in batch:
                if bidId is None:
                    continue
                bid = self.model.bidlist[bidId]
                bid.setPrice(output[bid.unique_id],
                             self.totalBudget/self.budgetRenewal)
                bid.priceLearningDataPos = self.learningDataPos
                self.bidPipeline.append(bid.unique_id) 
            
            # move eligible batches to pipeline
            toRemove.append(i)
        
        self.bidStaging = [x for i,x in enumerate(self.bidStaging) 
                                                    if i not in toRemove]
        
        # clean the queue
        tobeDeleted = [(w,x,y,z) for (w,x,y,z) in self.bidStaging 
            if (self.model.schedule.steps - w >= self.stagingMaxtime) 
            or (z - w >= max([qos for bidId,qos in x if bidId is not None]))]
        tobeDeleted += toDeactivate
        if len(self.bidStaging)>self.stagingMaxsize:
            tobeDeleted.append(self.bidStaging[0])
        for i in range(len(tobeDeleted)):
            createTime,batch,bidVector,nextChance = tobeDeleted[i]
            try:
                self.bidStaging.remove(tobeDeleted[i])
            except: # if there are duplicates in the list
                continue
            for bidId,_ in batch:
                if bidId is None:
                    continue
                bid = self.model.bidlist[bidId]
                bid.updateResult(BidStatus.REJECT,self.model.schedule.steps,
                         reason=bfr.ONHOLD)
                self.deactivateBid(bidId)
    
    def _collectRewards(self):
        ''' calculate bid rewards and write to priceMdl. payment and 
                rewardBidlist is updated in RSU.allocate through 
                vehicle.deactivateBid, or directly in RSU.allocate 
                if the bid is admitted.
            this function is called in Vehicle.advance. 
        '''
        loadBalancingWeight = 0.1
        for rsu in self.model.rsu.keys():
            freeCapa = maxAmount = 0
            for siteId in self.model.sites.keys():
                for servKey in self.model.sites[siteId].servicelist.keys():
                    try:
                        freeCapa_assigned = self.model.rsu[rsu].siteFreeCapa[
                                                            (siteId,servKey)]
                    except:
                        freeCapa_assigned = 1000
                    try:
                        maxAmount_assigned = self.model.rsu[
                                rsu].siteMaxAmounts[(siteId,servKey)]
                    except:
                        maxAmount_assigned = 1000
                    self.freeCapa_assigned[(rsu,siteId,servKey)] = (
                                freeCapa_assigned / maxAmount_assigned)
                    freeCapa += freeCapa_assigned
                    maxAmount += maxAmount_assigned
            self.freeCapa_all[rsu] = freeCapa / maxAmount

        for bidId in self.rewardBidlist:
            bid = self.model.bidlist[bidId]
            rsu = bid.rsu

            addReward = loadBalancingWeight * (
                            self.totalBudget / self.budgetRenewal 
                            * self.freeCapa_all[rsu])
            bid.addReward = (freeCapa,maxAmount,addReward)
            
            bidStatus = dict(Counter([x[0] for x in bid.statusHistory]))
            if (BidStatus.ADMIT in bidStatus.keys() and 
                bidStatus[BidStatus.ADMIT]>=bidStatus[BidStatus.PREP]):
                reward = bid.value - bid.payment + addReward
                if bid.payment==0:
                    reward = reward - bid.value
            else: #if bid.failReason in [bfr.NO_AVAILABLE_SITE,bfr.NOPRIO 
                reward = -bid.value + addReward
                        
            pos = bid.priceLearningDataPos
            self.priceMdl.collectReward(reward,pos)
        self.rewardBidlist = list()
        
        for bidId in self.rebidlist:
            bid = self.model.bidlist[bidId]
            self._rebid(bid)
        self.rebidlist = list()
    
    def _activateBids(self):
        ''' activate bids in pipeline: set the indicator, and append it to
                RSU. update bid state information for CompetitorLearningModel.
            called in Vehicle.step.
        '''
        while (len(self.activeBid)<self.maxNrBids and len(self.bidPipeline)>0
               and self.bidCount<=self.totalNrBids):
            bid = self.model.bidlist[self.bidPipeline.pop(0)]
            if bid.model is None:
                continue
            bid.activate()
            bid.rsu = self.rsu
            self.activeBid.append(bid.unique_id)
            self.model.rsu[self.rsu].bids.append(bid.unique_id)
            if self.transitionTbl is not None:
                self.model.transitionTbl[
                        self.transitionTbl].updateBidStateInfo(bid.unique_id)
    
    def _getAction(self):
        lower = 0
        upper = 1
        mu = self.stagingThreshold
        sigma = max(mu-lower,upper-mu)/2
        return truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,
                             loc=mu,scale=sigma)
    
    def _rebid(self,bid):
        ''' put rejected bids back into staging area. called at the end of
                Vehicle._collectRewards in Vehicle.advance.
        '''
        bid.updateResult(BidStatus.PREP,self.model.schedule.steps)
        bid.payment = -1
        bid.price = 0
        bid.rsu = self.rsu
        bid.siteBinding = self.rsu
        bid.isActive = 0

        bidVectorNew = list()
        for i in range(self.budgetRenewal):
            bidInfo = {k:0 for k in cbd.keys}
            taskSeq = [0] * len(cbd.services)
            if i==0:
                bidInfo[(bid.learningKey,bid.qostype)] += 1
                taskSeq = bid.taskSeq.values()
            bidVectorNew += list(bidInfo.values())
            bidVectorNew += taskSeq

        batchNew = [(None,0)] * self.budgetRenewal
        batchNew[0] = (bid.unique_id,bid.qosRequired)
        self.bidStaging.append((self.model.schedule.steps,
                            batchNew,bidVectorNew,
                            self.model.schedule.steps))
        

    def deactivateBid(self,bidId):
        ''' 
        helper function to deactivate a given bid. currently removing it 
            from the activeBid list will give space to create more bids.
        currently reward is collected immediately after admission
            decision in RSU.allocate, and not when the bids are serviced. 
            Therefore reward is not collected AGAIN if payment has already
            been determined in RSU.allocate.
        called from RSU._rejectBids in RSU.allocate. also called from
            Bid.estimate, to catch finished bids from 
            Service._unloadService in Service.step.
        @param bidId: unique_id of the bid to be deactivated.
        '''
        bid = self.model.bidlist[bidId]
        bid.isActive = 0
        try:
            self.activeBid.remove(bidId)
        except:
            pass        
        bidStatus = dict(Counter([x[0] for x in bid.statusHistory]))
        if (BidStatus.BID in bidStatus.keys() and 
            bidStatus[BidStatus.BID]>=bidStatus[BidStatus.PREP]):
            # if not directly from bidStaging:
            if bid.payment==-1 and bid.failReason!=bfr.DUPLICATE:
                if np.isnan(bid.price):
                    bid.price = -1
                # if the bid is not re-allocated before rejected:
                # (currently only recording admission rewards. if admitted
                # the reward is collected in rsu.allocate)
                if self.transitionTbl is not None:
                    transTbl = self.model.transitionTbl[self.transitionTbl]
                    try:
                        idx = list(transTbl.stateChangeFlag.keys()).index(
                            (bid.siteBinding,(bid.learningKey,bid.qostype)))
                        history = [x[idx] * self.totalBudget 
                                               for x in transTbl.stateHistory]
                        start = max(0,len(history)-self.lamChangeInterval)
                        history = max(history[start:])
                    except:
                        history = (self.totalBudget / self.budgetRenewal)
                    high = max(history,bid.price+2)
                    bid.payment = np.random.randint(low=bid.price+1,high=high)                   
                    transTbl.updateStateInfo(bidId)
                else:
                    bid.payment = np.random.randint(low=bid.price+1,
                                        high=max(self.totalBudget,bid.price+2))
                    
                self.rewardBidlist.append(bidId)

        
        items = utCom.listClassItems(bid)
        self.bids_random = self.bids_random.append(items,ignore_index=True)    
        for task in bid.chain:
            task_items = utCom.listClassItems(task)
            self.tasks_random = self.tasks_random.append(task_items,
                                                         ignore_index=True)   

        # rebid or clean up
        if (self.rebid and bid.status in [BidStatus.REJECT] and 
            bid.failReason not in [bfr.NA,bfr.DUPLICATE,bfr.ACTUAL_DUE_TIME] 
            and self.model.schedule.steps < bid.dueTime
            and bid.nrRebid>0):
            bid.nrRebid -= 1
            self.rebidlist.append(bidId)
        else:
            bid.finishTime = self.model.schedule.steps
            nrRebid = bp.nrRebid
            bidfile = self.model.filedict[FILES.BID_FILE[0]]
            bidfile.write(
             '{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}\n'.format(
             self.model.schedule.steps,bidId,bid.status,bid.isSuccess,
             bid.learningKey,bid.qostype,bid.price,bid.payment,bid.availBudget,
             bid.batchCreateTime,bid.createTime,bid.bestProposal.proposedDest,
             bid.finishTime,nrRebid-bid.nrRebid,bid.failReason,bid.value))

            # clean up
            bid.model = None
            bid.proposals = None
            self.bidHistory.append(bidId)
        
    def collectPriceModelNextState(self,stateVec,pos):
        ''' 
        helper to record target state data in PriceLearningModel.
            the data is from TransitionTbl object while creating state
            vector for CompetitorLearningModel in 
            TransitionTbl._addStateRecord from TransitionTbl.advance.
        @param stateVec: actual competitor state from TransitionTbl
        @param pos: corresponding position of data in priceMdl.
        '''
        self.priceMdl.collectNextState(stateVec,pos)
    
    def step(self):
        if self.isActive:
            self.availLifespan -= 1
            self.countdown -= 1
            if self.availLifespan <= 0:
                self.isActive = False
            
            if self.rsu is None:
                self._findRSU()
                self._chooseRSU()

            self._activateBids()
            
            if ((not self.model.trainable) or (self.model.evaluation)
                or (self.model.schedule.steps>=pmp.evaluation_start)):
                return
            
            if (int(self.learningDataPos)-self.prevRecordNr
                                                >=self.plmTrainingThres):
                self.priceMdl.train(self.model.schedule.steps,
                                    self.model.filedict[FILES.PLM_FILE[0]])
                self.superviseMdl.train(self.model.schedule.steps,
                                        self.model.filedict[FILES.SL_FILE[0]])
                
                self.prevRecordNr = int(self.learningDataPos)

    def estimate(self):
        if self.isActive:
            self._fillPipeline()
    
    def allocate(self):
        # rsu allocates bids and adds rewards
        if (self.model.schedule.steps>0 and 
            np.mod(self.model.schedule.steps,self.lamChangeInterval)==0):
            self._updateLambda()
    
    def advance(self):
        if (self.transitionTbl is None 
            and len(self.bidHistory) >= self.competitorDataThres):
            self._createTransitionTbl()
            bidlist = self.bidHistory + self.activeBid
            self.model.transitionTbl[self.transitionTbl].populateTbl(bidlist)
        self._collectRewards()
        
    def move(self):
        new_position = (self.pos[0]+self.speed,self.pos[1])
        self.model.grid.move_agent(self, new_position)
        self.pos = new_position
        
    @classmethod
    def getInstances(cls):
        nonexistent = set()
        for ref in cls._instances:
            obj = ref()
            if obj is not None:
                yield obj
            else:
                nonexistent.add(ref)
        cls._instances -= nonexistent


class Bid(Agent):
    '''
    step:.
    estimate: 
        1. catch finished bids from Service._unloadService in Service.step, to
            deactivate the bid. Currently rewards are collected only upon
            admission in RSU.allocate (only when payment is determined
            for the first time, through a if-clause in the 
            Vehicle.deactivateBid function). There is the possibility of 
            implementing delayed reward on successfully finishing the bid. 
        2. collect proposals from all resource sites in range (currently
            all resource site are in range). pre-select the best proposal 
            by ordering rules defined in ServiceProposal class.
        
        the bids will be allocated in RSU.allocate after this step.
    allocate:.
    advance:.
    '''
    QoS1 = bp.QoS1
    QoS2 = bp.QoS2
    servAmount1 = bp.servAmount1
    servAmount2 = bp.servAmount2
    minDataSize = bp.minDataSize
    maxDataSize = bp.maxDataSize
    nrRebid = bp.nrRebid

    def __init__(self,bidId,qostype,servAmount,user,budget,
                 createTime,coord,model,savedRecord=None):
        self.unique_id = bidId + '_' + str(qostype)
        super().__init__(self.unique_id, model)
        self.batch_id = bidId # bids with same batch id can only be 
                              # allocated once.
        self.user = user # associated vehicle.
        self.rsu = None # associated RSU.
        self.pos = coord # coordination of the bid, not used.
        self.availBudget = budget # vehicle total budget / budget renewal
        self.batchCreateTime = createTime # time bid is created
        self.finishTime = -1 # time bid is done
        self.createTime = createTime # time bid is sent to RSU for allocation
        self.siteBinding = None # current resource site processing the bid
        self.transmissionTimeTotal = 0 # transmission time if different sites
        self.transmissionTimeToGo = 0 # if zero then can be process by site.
        self.datasize = 0 # packet size relevant if allocated site changes
        self.price = 0 # total bidding price
        self.priceLeftover = 0 # calculated after each task is finished
        self.priority = 0 # if admitted by RSU but cannot be served, increase 
                          # priority to be considered earlier by RSU.
        self.qostype = qostype # qos type as part of bid category
        self.qosRequired = self.QoS1 if qostype==QoSType.HIGH else self.QoS2
        self.servAmountLvl = servAmount # service amount level description
        
        self.servAmount = self.servAmount2
        self.chain = list() # chain of tasks (instances of services)
        self.status = BidStatus.PREP # stage the bid is in
        self.statusHistory = [(self.status,self.createTime)] # for debugging
        self.dueTime = 0 # required due time calculated from qos.
        self.estimFinishTime = 0 # estimated time for finishing the bid
        self.currentTask = 0 # current task in progress
        self.qosActual = -1 # actual qos needed
        self.isSuccess = 0 # indicator if successfully finished
        self.isActive = 0 # indicator if still active (in queue or process)
        self.proposals = dict() # duration and cost proposals from sites
        self.bestProposal = ServiceProposal(sys.maxsize,None,
                            self.dueTime,None,0,dict()) # best proposal
        self.payment = -1 # second price as payment
        self.addReward = None # if admitted: additional reward from 
                              # free capacity of sites, calculated in 
                              # vehicle._collectRewards: 
                              # freeCapa,maxAmount,addReward
        self.environment = 0 # e.g. nr bids in rsu at time of creation
        self.competitorEnv = 0 # e.g. nr bids in rsu at time of allocation
        # to help find position for reward data in model
        self.priceLearningDataPos = -1 # key to connect input, output and 
                                       # reward in PriceLearningModel
        self.failReason = bfr.NA # failed reason if rejected
        
        if savedRecord is None:
            self._createPacket()
            self.savedRecord = self._saveRecord()       
            self._createTaskChain()
        else:
            self.datasize = savedRecord['datasize']
            self._createTaskChainFromRecord(savedRecord)
        
        self.learningKey,self.taskSeq = self._createLearningKey() 
                        # bid category identifier, 
                        # also used as a state id for CompetitorLearningModel
        
        self.value = self._getBidValue()
        
    def _getBidValue(self):
        maxServiceAmount = max(self.servAmount1,self.servAmount2)
        maxQoS = max(1/self.QoS1,1/self.QoS2)
        maxTaskChainLen = len(cbd.services)
        
        proportion = ( (self.servAmount / maxServiceAmount) 
            * (1/self.qosRequired / maxQoS) 
            * (len(self.chain) / maxTaskChainLen) )
        
        minProportion = 0.75
        value = (minProportion * self.availBudget 
                 + (1 - minProportion) * self.availBudget * proportion)
        return value
        
    def _createPacket(self):
        ''' immitate transmission time between sites by creating data packets
            sized 300-1200 bytes.
        '''
        self.datasize = (np.random.randint(self.minDataSize,
                                           self.maxDataSize) * 8)
        
    def _createTaskChain(self):
        ''' create chain of tasks with random service type and sequence.
                resource profile associated with the service is also 
                randomly generated based on a normal distribution defined
                in DefaultProfile.
        '''
        services = cbd.services
        prob = [expon.pdf(x) for x in np.arange(1,len(services)+1)]
        prob = [x/sum(prob) for x in prob]
        length = np.random.randint(1,len(services)+1)
        chain = [services[i] for i in np.random.choice(range(len(services)), 
                 size=length,replace=False,p=prob)]
        self.savedRecord['chain'] = chain

        savedProfiles = []
        for pos, tup in enumerate(chain):
            serv, servType = tup
            # estim. resource profile created based on standard type of site:
            resProfile = DefaultProfile.randomPopulate(SiteType.TYPE1,servType)
            task = Task(pos,serv,servType,self.servAmount,self.unique_id,
                        self.batchCreateTime,self.qosRequired,resProfile) 
            savedProfiles.append(resProfile)
            self.chain.append(task)
        
        self.savedRecord['resProfile'] = savedProfiles
    
    def _createTaskChainFromRecord(self,savedRecord):
        ''' for repeatability, create bids from historical records '''
        chain = savedRecord['chain']
        for pos, tup in enumerate(chain):
            serv, servType = tup
            resProfile = savedRecord['resProfile'][pos]
            task = Task(pos,serv,servType,self.servAmount,self.unique_id,
                        self.batchCreateTime,self.qosRequired,resProfile) 
            self.chain.append(task)        
        
    def _createLearningKey(self):
        ''' helper function to create key indicating the bid category, used 
                in CompetitorLearningModel 
        '''
        key = list()
        bidTasks = dict([((x.serviceName,x.serviceType), x.pos)
                                    for x in self.chain[self.currentTask:]])
        taskSequence = dict()
        for service in cbd.services:
            if service in bidTasks.keys():
                key.append((service,self.servAmountLvl))
                taskSequence[service] = bidTasks[service] + 1
            else: 
                key.append((service,ServiceAmount.LOW))
                taskSequence[service] = 0
        if len(key)==1:
            return key[0],taskSequence
        else:
            return tuple(key),taskSequence
    
    def _saveRecord(self):
        ''' save for repeatability'''
        locations = [x.span() for x in re.finditer('_',self.unique_id)]
        user = self.unique_id[0:locations[1][1]]
        batch = self.unique_id[locations[2][1]:locations[3][0]]
        savedRecord = {'qostype':self.qostype,'datasize':self.datasize,
                       'servAmount':self.servAmountLvl,'user':user,
                       'batch':batch,'batchCreateTime':self.batchCreateTime}
        return savedRecord

    def setPrice(self,price,budget):
        '''
        placeholder for setting bidding price (if deviation from given
            price). called from Vehicle._fillPipeline in Vehicle.estimate.
        @param price: learned bidding price for the bid from Vehicle.priceMdl.
        '''
        # if the bid is from rebidding, deduct incurred costs
        self.price = min(price,budget - sum([t.cost for t in self.chain]))
        self.priceLeftover = self.price
    
    def activate(self):
        ''' 
        helper function called from Vehicle._activateBids in Vehicle.step
        '''
        self.isActive = 1
        self.createTime = self.model.schedule.steps
        self.dueTime = self.batchCreateTime + self.qosRequired
        self.estimFinishTime = self.dueTime
        self.competitorEnv = self.model.rsu[self.rsu].nrActiveBids
        self.updateResult(BidStatus.BID,self.model.schedule.steps)
        
    def updateResult(self,status,time,qosActual=-1,reason=None):
        ''' called in every stage of the bid lifecycle for a complete
                record. 
            @param status: keyword of the stage
            @param time: status change time
            @param qosActual: actual time the bid is closed
            @param reason: fail reason if the bid is rejected.
        '''
        if self.status == status:
            return
        self.status = status
        self.statusHistory.append((self.status,time))
        if status==BidStatus.FINISH:
            self.qosActual = qosActual
            if qosActual<=0:
                self.failReason = bfr.ACTUAL_DUE_TIME
            elif self.qosRequired<qosActual:
                self.failReason = bfr.ESTIM_DUE_TIME
            elif self.priceLeftover<0:
                self.failReason = bfr.COST
            else:
                self.isSuccess = 1
                
        if status==BidStatus.REJECT:
            self.failReason = reason
        if status in (BidStatus.FINISH,BidStatus.REJECT):
            self.isActive = 0

    def acceptProposal(self,siteId,rsuId,rsuHomesiteId):
        ''' 
        called by RSU.allocate after evaluation of all proposals. update 
            data object of the bid for debugging.
        @param siteId: unique_id of the resource site in RSU's allocation 
            decision.
        @param rsuId: unique_id of the RSU
        @param rsuHomesiteId: unique_id of the resource site where RSU resides.
        '''
        proposal = self.proposals[siteId]

        for pos in proposal.proposedResProfile.keys():
            self.chain[
              int(pos)].resProfileSelected = proposal.proposedResProfile[pos]

        self.siteBinding = proposal.proposedDest
        self.estimFinishTime = proposal.proposedEstimFinishTime
        self.transmissionTimeTotal = proposal.proposedTransmitTime
        self.transmissionTimeToGo = proposal.proposedTransmitTime
            
    def step(self):
        pass

    def estimate(self):
        '''
        1. catch finished bids from Service._unloadService in Service.step, to
            deactivate the bid. Currently rewards are collected only upon
            admission in RSU.allocate (only when payment is determined
            for the first time, through a if-clause in the 
            Vehicle.deactivateBid function). There is the possibility of 
            implementing delayed reward on successfully finishing the bid. 
        2. collect proposals from all resource sites in range (currently
            all resource site are in range). pre-select the best proposal 
            by ordering rules defined in ServiceProposal class.
        
        the bids will be allocated in RSU.allocate after this step.
        '''
        if self.siteBinding is None:
            self.siteBinding = self.rsu

        # some task of the bid is being processed:
        if self.status in (BidStatus.PREP,BidStatus.PROCESS,
                           BidStatus.TRANSMIT) or self.model is None:
            return
        if self.status in (BidStatus.REJECT,BidStatus.FINISH):
            self.model.rsu[self.rsu].removeBids(self.siteBinding,
                                                          self.unique_id)
            self.model.vehicles[self.user].deactivateBid(self.unique_id)
            return

        self.proposals = dict()
        self.bestProposal = ServiceProposal(sys.maxsize,None,
                                            self.dueTime,None,0,dict())
        for siteId in self.model.rsu[self.rsu].sites:
            s = self.model.sites[siteId]
            (duration,leftover,resProfile,
             cost,transmitTime) = s.estimTimeInSystem(self.unique_id)
            
            self.proposals[siteId] = ServiceProposal(cost,siteId,
                self.dueTime-leftover,self.unique_id,transmitTime,resProfile)

            if (self.proposals[siteId] > self.bestProposal 
                and cost <= self.priceLeftover):
                self.bestProposal = self.proposals[siteId]   
    
    def allocate(self):
        pass
    
    def advance(self):
        pass
            
    def __eq__(self, other):
        if isinstance(other, Bid):
            return (self.batch_id==other.batch_id 
                    and self.qostype==other.qostype) or (
                            self.unique_id==other.unique_id)
        return False

class TransitionTbl(Agent):
    '''
    step: 1. vehicle._activateBids triggers self.updateBidStateInfo to record
        actual bidding behavior of time t, which will become part of the 
        inputVec(t) to CompetitorLearningModel. Full record will be created in 
        self.advance in t, and collected into learning model 
        in self.step in t+1.
        2. self._chooseNextState is called to collect inputVec from t-1, and 
        to "guess" current competitor state (output from the model) for time t.
        The output will be used in vehicle.estimate, where 
        vehicle._fillPipeline will be called to run the PriceLearningModel
        and set prices for the next batch of bids. Note that due to the 
        lag between pipeline and activation, these bids may not necessarily be
        activated in the time unit t+1. 
    estimate: 1. vehicle._fillPipeline will create bids and assign budget to 
        each bid based on guesses of current competitor state and the bids' 
        service requirements, for activation in time t+1.
        2. activated bids will be roll-calling sites for offering prices. 
        3. train competitor learning model in self.estimate.
    allocate: 1. rsu.allocate will allocate activated bids to sites or reject. 
        admission result and payment in each bid is calculated for time t. 
        2. self.updateStateInfo is called by rsu.allocate to update "actual" 
        competitor states for time t. This competitor state is a sampling of 
        the actual distribution only with payment information. The information 
        will become part of the inputVec in self.advance.
    advance: 1. self._addStateRecord creates competitor state record 
        for time t based on activated bid info of time t from self.step, 
        and actual competitor state info of time t from self.allocate. 
        2. self._addStateRecord also adds actual competitor state info to 
        CompetitorLearningModel as result of the previous inputVec (for
        future supervised learning). 
        3. self.populateTbl can be run in this stage too for batch creation of
        dataset for training purposes.
        4. self._addStateRecord also calls Vehicle.collectPriceModelNextState
        to add next stateVec to price learning model for $\hat V(S_{t+1},w)$
    '''
    trainingThres = tp.trainingThres
    historyPeriods = tp.historyPeriods
    
    def __init__(self,vehicleId,budget,benchmarkNrBid,model,clmfile):
        self.unique_id = vehicleId + '_transition'
        super().__init__(self.unique_id,model)
        self.availBudget = budget # vehicle total budget / budget renewal
        self.benchmarkNrBid = benchmarkNrBid # for normalization of env. var.
        self.vehicle = vehicleId # associated vehicle
        self.stateChangeFlag = {(s,k):0 for k in cbd.keys for s in self.model.sites.keys()} # indicator if state 
                                                       # has changed
        self.states = dict() # competitor states, categorized by 
                             # bid categories (service amount and qos type)
        self._resetCompetitorStates() # initiate state info
        self.bidStates = dict() # bid states, categorized by bid categories
        self._resetBidStates() # initiate bid state info
        self.estimNextState = list() # estimated next state 
        self.competitorMdl = clm(uniqueId=self.unique_id,
                dimOutput=len(self.states),
                evaluation=self.model.evaluation) # CompetitorLearningModel object
        self.clmfile = clmfile # output file path and name
        self.stateHistory = list() # for creating input vector. keeping the 
                                   # history enables creating input vector 
                                   # from more than one previous state records        
        self.newRecordFlag = False # flag if there are new state records.
        self.prevRecordNr = 0 # previous record number in clm
        self.learningDataPos = str(0) # current record number in clm
        self.bidlist_open = list() # for debugging
        self.bidlist_closed = list() # for debugging
        self.priceLearningDataPos = 0 # for updating the next state in plm
    
    def _resetCompetitorStates(self):
        ''' initial setup '''
        for siteId in self.model.sites.keys():
            for k in cbd.keys:
                self.states[(siteId,k)] = list()
    
    def _getCompetitorStateRecord(self,rsuId=None):
        ''' simple processing of state inputs to avoid duplicate. e.g. if
                state information from different bids are different, simply
                take the maximum payment as the actual state.
            called by TransitionTbl._addStateRecord in 
                TransitionTbl.advance 
        '''
        if rsuId is not None:
            secondPrices = self.model.rsu[rsuId].secondPrices
            state = list()
            for siteId in self.model.sites.keys():
                for k in cbd.keys:
                    if (siteId,k) in secondPrices.keys():
                        state.append(secondPrices[(siteId,k)])
                    else:
                        state.append(0)
            return state
        
        state = list()
        for siteId in self.model.sites.keys():
            for k in cbd.keys:
                if len(self.states[(siteId,k)])>0:
                    state.append(max(self.states[(siteId,k)]))
                else:
                    state.append(0)
        return state
        
    def _resetBidStates(self):
        ''' reset after each TransitionTbl._addStateRecord in 
                TransitionTbl.advance 
        '''
        for k in cbd.keys:
            if k not in self.bidStates.keys():
                self.bidStates[k] = BidState()
            else:
                self.bidStates[k].reset()

    def _chooseNextState(self,infer=False):
        '''
        create TransitionTbl.estimNextState by running the 
            CompetitorLearningModel's inference function. run in each time
            step in TransitionTbl.step. The estimated next
            state will be used in Vehicle._fillPipeline from Vehicle.estimate.
        also used in TransitionTbl.populateTbl, when batch inputs and target
            outputs are created before the TransitionTbl is created. In this 
            case the function is used for generating input vectors, but the
            next state estimation will not be necessary.
        @param infer: if True, TransitionTbl.estimNextState is created 
            normally. if False, it's only for populateTbl.
        '''
        # if no updates, omit the step.
        if self.newRecordFlag is False:
            return
        # omit the time step record in history
        inputVec = [x[:-1] for x in self.stateHistory[-self.historyPeriods:]]
        if self.historyPeriods > 1:
            inputVec = [x for y in inputVec for x in y] #flatten
        else:
            inputVec = inputVec[0]
        # output: estimated current competitor state
        if infer:
            self.estimNextState = (
                    self.competitorMdl.inference(inputVec)).tolist()
        else:
            self.estimNextState = self.competitorStates
        
        self.learningDataPos = self.competitorMdl.collectInput(inputVec)
        self.newRecordFlag = False
        
    def _addStateRecord(self,nrActiveBids,timestep,rsuId=None):
        '''
        called in TransitionTbl.advance. consolidates state information from 
            individual bids (in RSU.allocate), bid state information, and 
            environment variables to create input vectors and target outputs.
            also provide next state to vehicle's PriceLearningModel's critic.
            all variables and target outputs are normalized.
        @param nrActiveBids: environment variable. In reality this should be
            equivalent to channel occupancy and can be 
            obtained by pinging the RSU and perceiving the delay.
        @param timestep: state record's time step. Normally this is the 
            current time step. in populateTbl this is the bid creation time.
        '''
        if sum(self.stateChangeFlag.values())==0: # no change in this time step
            self._resetBidStates()
            return
        if (len(self.stateHistory)>0 
            and self.stateHistory[-1][-1]==self.model.schedule.steps):
            # prevent duplicates in the same time step
            self._resetBidStates()
            return
        normalizedNrBids = nrActiveBids / self.benchmarkNrBid
        bidStates = [y for x in self.bidStates.values() 
                                            for y in x.getRecord()]

        # if rsuId is given, the competitor state is from the rsu failed 
        #   bid information (rsu.secondPrices)
        # if rsuId is not given, then an estimation from the bid payment 
        #   information is used (self.state)
        # estimate of next state: if self.step has infer=False, 
        #   then use the current state. 
        # if self.step has infer=True, then use the competitor model inference.
        self.competitorStates = self._getCompetitorStateRecord(rsuId=rsuId)
        self.stateHistory.append(self.competitorStates + bidStates 
                                 + [normalizedNrBids,timestep])
        for key in self.stateChangeFlag.keys():
            self.stateChangeFlag[key] = 0
        self._resetBidStates()
        self.newRecordFlag = True
        self.competitorMdl.collectOutput(self.competitorStates,
                                         self.learningDataPos)
        
        self.model.vehicles[self.vehicle].collectPriceModelNextState(
                self.competitorStates,self.priceLearningDataPos)
    
    def populateTbl(self,bidlist):
        '''batch-create data record for CompetitorLearningModel'''
        if (len(bidlist)==0):
            return
        bidDict = dict([(x.unique_id,(x.createTime,x.environment))
                        for x in self.model.bidlist.values() 
                        if x.unique_id in bidlist])
        sortedBids = [(k,v) for k,v in sorted(bidDict.items(),
                                          key=lambda item:item[1][0])]
        createTime = sortedBids[0][1][0]
        for bidId,(bidCreateTime,env) in sortedBids:
            if bidCreateTime>createTime:
                # collect output for training
                self._addStateRecord(env,bidCreateTime)
                # collect input for training
                self._chooseNextState(infer=False)
                createTime = bidCreateTime
            self.updateStateInfo(bidId)
            self.updateBidStateInfo(bidId)
        self._addStateRecord(env,bidCreateTime)
        
        self.competitorMdl.train(time=self.model.schedule.steps,
                                 clmfile=self.clmfile)
    
    def updateBidStateInfo(self,bidId):
        '''
        collect bid information.
        called by Vehicle._activateBids in Vehicle.step. 
        @param bidId: unique_id of the bid in question.
        '''
        self.bidlist_open.append(bidId)
        bid = self.model.bidlist[bidId]
        key = (bid.learningKey,bid.qostype)
        self.bidStates[key].nrBidsInCategory += 1
        if bid.price / self.availBudget > self.bidStates[key].highestPrice:
            self.bidStates[key].highestPrice = bid.price / self.availBudget
        if bid.price / self.availBudget < self.bidStates[key].lowestPrice:
            self.bidStates[key].lowestPrice = bid.price / self.availBudget
        for pos in range(bid.currentTask,len(bid.chain)):
            servNameType = (bid.chain[pos].serviceName,
                            bid.chain[pos].serviceType)
            if pos==bid.currentTask:
                self.bidStates[key].currentService[servNameType] += 1
            else:
                self.bidStates[key].futureService[servNameType] += 1
        
        self.priceLearningDataPos = bid.priceLearningDataPos
        
    def updateStateInfo(self,bidId):
        '''record actual competitor states after rsu.allocate'''
        self.bidlist_closed.append(bidId)
        bid = self.model.bidlist[bidId]
        rsu = self.model.vehicles[self.vehicle].rsu
        if bid.siteBinding is None or bid.siteBinding==rsu:
            return
        
        currentStateKey = (bid.siteBinding,(bid.learningKey,bid.qostype))
        if self.stateChangeFlag[currentStateKey]==0:
            self.states[currentStateKey] = list()
            self.stateChangeFlag[currentStateKey] = 1

        # if admitted is true, price is bid payment. else it is bidding price.
        bidStatus = dict(Counter([x[0] for x in bid.statusHistory]))
        if (BidStatus.ADMIT in bidStatus.keys() and 
                bidStatus[BidStatus.ADMIT]>=bidStatus[BidStatus.PREP]):
            self.states[currentStateKey].append(bid.payment/self.availBudget)
        else: 
            # use average of past guesses as state. for rejected bids, 
            # payments are calculated in vehicle.deactivateBid
            self.states[currentStateKey].append((bid.payment)/self.availBudget) 
        
    def step(self):
        '''
        bids are activated from Vehicle._activateBids. bid information is
            also collected there by calling self.updateBidStateInfo. 
        competitor states are collected in self.advance, either from 
            rsu.secondPrices directly, or estimated from bid payment info
            and stored in self.states.
        If in self._chooseNextState infer=True, for the next step of 
            setting bidding price, next competitor state is from model
            inference. Otherwise it's the copy of current competitor state.
        '''
        self._chooseNextState(infer=False)
    
    def estimate(self):
        ''' bidding prices are set in Vehicle.estimate. '''
        if ((not self.model.trainable) or (self.model.evaluation)
            or (self.model.schedule.steps>=pmp.evaluation_start)):
            return
        
        if (int(self.learningDataPos)-self.prevRecordNr>=self.trainingThres):
            self.competitorMdl.train(time=self.model.schedule.steps,
                                     clmfile=self.clmfile)
            self.prevRecordNr = int(self.learningDataPos)
    
    def allocate(self):
        '''
        bid info with reward info collected here by rsu.allocate.
        updateStateInfo is called from Vehicle.allocate.
        '''
        pass
    
    def advance(self):
        rsu = self.model.vehicles[self.vehicle].rsu
        env = self.model.rsu[rsu].nrActiveBids
        timestep = self.model.schedule.steps
        self._addStateRecord(env,timestep,rsuId=rsu)
        

class V2XModel(Model):
    width = mp.width
    height = mp.height
    rsuPos = mp.rsuPos
    rsuInterval = mp.rsuInterval
    resSitePos = mp.resSitePos
    resSiteInterval = mp.resSiteInterval
    vehicleYposChoice = list(np.arange(*mp.vehicleYposChoice))
    vehicleInterval = mp.vehicleInterval
    lam = mp.lam
    totalSimTime = mp.totalSimTime
    timeForNewVehicles = mp.timeForNewVehicles
    recent = mp.recent
    
    def __init__(self,filedict,nrSites=mp.nrSites,initVehicles=mp.initVehicles,
                 nrRsu=mp.nrRsu,train=True,evaluation=False,repeatable=True,
                 loadModel=False,resourceCapaIdx=0,rebid=False):
        self.nrSites = nrSites # number of resource sites to create
        self.nrRsu = nrRsu # number of RSUs to create
        self.totalNrVehicles = 0 # for naming vehicles
        self.sites = dict() # list of ResourceSite objects
        self.rsu = dict() # list of RSU objects
        self.vehicles = dict() # list of Vehicle objects
        self.bidlist = dict() # list of Bid objects
        self.servicelist = dict() # list of Service objects
        self.transitionTbl = dict() # list of TransitionTbl objects
        self.filedict = filedict # all paths and names of output files
        self.trainable = train # if to run the training algorithms
        self.evaluation = evaluation # if only to run inference with already
                                     # trained model
        self.repeatable = repeatable # if create same bids with same interval
        self.loadModel = loadModel # if to use pre-trained models
        self.resourceCapaIdx = resourceCapaIdx
        self.rebid = rebid
        
        self._printHeader() # print all file headers
        # requires each agent to implement all methods 
        # step, estimate, allocate, advance:
        self.schedule = SimultaneousActivation(self)
        # for visualization. not used.
        self.grid = MultiGrid(self.width, self.height, torus = False)
        
        # create sites
        for i in range(nrSites):
            if i==0: # 
                sitetype = SiteType.TYPE3 # cloud
            elif i==1:
                sitetype = SiteType.TYPE1 # standard server
            elif i==2:
                sitetype = SiteType.TYPE2 # slow server
            else:
                sitetype = SiteType.TYPE1
            if nrSites==1:
                sitetype = SiteType.TYPE1
            agent = ResourceSite('site'+str(i),sitetype,self.resourceCapaIdx,
                                 self)
            agent.pos = (self.resSitePos[0]
                                  +self.resSiteInterval*i, self.resSitePos[1])
            self.sites[agent.unique_id] = agent
            self.schedule.add(agent)
        self.siteId = [s.unique_id for s in self.sites.values()]
        self.siteDist = ResourceSiteDistance(self.siteId)
        
        for s in self.sites.values():
            s.siteInfo = self.siteDist
        
        # create RSU
        for i in range(nrRsu):
            agent = RSU(uniqueId='RSU'+str(i),siteMat=self.siteDist,model=self)
            agent.pos = (self.rsuPos[0]
                                  +self.rsuInterval*i, self.rsuPos[1])
            self.rsu[agent.unique_id] = agent
            self.schedule.add(agent)
        
        # create cars
        if self.schedule.steps < self.timeForNewVehicles:
            self._createNewVehicles(initVehicles)
    
    def _createNewVehicles(self,nrArrivals):
        
        addId = '_' + str(self.nrSites) + '_' + str(self.trainable) # for identifying different models
        vehicleBudgetPath = os.path.join(MODEL_DIR,'vehicleBudget.pkl')
        saveVehicleBudgets = False
        try:
            vehicleBudgets = pickle.load(open(vehicleBudgetPath,'rb'))
        except:
            vehicleBudgets = None
            saveVehicleBudgets = True
        
        for i in range(nrArrivals):
            try:
                budget = vehicleBudgets[i]
            except:
                budget = None
                saveVehicleBudgets = True
            agent = Vehicle('vehicle'+str(self.totalNrVehicles)+addId,
                            self.schedule.steps,self,budget,self.rebid)
            agent.pos = (0+self.vehicleInterval*i,
                         np.random.choice(self.vehicleYposChoice))
            self.vehicles[agent.unique_id] = agent
            self.schedule.add(agent)
            self.totalNrVehicles += 1
        
        if saveVehicleBudgets:
            vehicleBudgets = [x.totalBudget for x in self.vehicles.values()]
            pickle.dump(vehicleBudgets,open(vehicleBudgetPath,'wb'))
    
    def _printHeader(self):
        perfile = self.filedict[FILES.PERF_FILE[0]]
        perfTitle = 'step;occupied;utilization;maxAmount;sitetype;pmdlRecord;'
        perfTitle += 'totalBid;active;success;recentSuccess;accepted;'
        perfTitle += 'recentAccepted;rejectedBid;recentReject;finishedBid;'
        perfTitle += 'totalSuccessRatio;totalAcceptRatio;totalNonRejectRatio;'
        perfTitle += 'recentSuccessRatio;recentAcceptRatio;recentNonRejectRatio;'
        perfTitle += 'modelType;capacity;markovProb;budget\n'
        if os.path.getsize(perfile.name) == 0:
            perfile.write(perfTitle)
        
        logfile = self.filedict[FILES.ALLOC_FILE[0]]
        logTitle = 'step;totalActiveBids;sortIdx;bidId;status;'
        logTitle += 'bidServiceKey;bidQoS;bidPrice;bidPayment;carBudget;'
        logTitle += 'batchCreate;createTime;site;sitetype;'
        logTitle += 'finishTime;nrRebid;decision;bidValue\n'
        if os.path.getsize(logfile.name) == 0:
            logfile.write(logTitle)

        bidfile = self.filedict[FILES.BID_FILE[0]]
        bidTitle = 'step;bidId;status;success;'
        bidTitle += 'bidServiceKey;bidQoS;bidPrice;bidPayment;carBudget;'
        bidTitle += 'batchCreate;createTime;site;'
        bidTitle += 'finishTime;nrRebid;failReason;bidValue\n'
        if os.path.getsize(bidfile.name) == 0:
            bidfile.write(bidTitle)
        
        plmfile = self.filedict[FILES.PLM_FILE[0]]
        plmTitle = 'step;vehicle;epoch;nrInput;avgReward;'
        plmTitle += 'criticLoss;actorLoss;restart\n'
        if os.path.getsize(plmfile.name) == 0:
            plmfile.write(plmTitle)
        
        clmfile = self.filedict[FILES.CLM_FILE[0]]
        clmTitle = 'step;vehicle;epoch;nrInput;competitorLoss;restart\n'
        if os.path.getsize(clmfile.name) == 0:
            clmfile.write(clmTitle)

        slfile = self.filedict[FILES.SL_FILE[0]]
        slTitle = 'step;vehicle;epoch;nrInput;avgReward\n'
        if os.path.getsize(slfile.name) == 0:
            clmfile.write(slTitle)
        
    def _print(self,perfile):
        if np.mod(self.schedule.steps,100)==0:
            print("step:{}".format(self.schedule.steps))
        recent = min(self.schedule.steps,self.recent)

        totalBid = len(set([x.batch_id for x in self.bidlist.values()]))
        if totalBid==0:
            totalBid = 1
        
        # active bids being processed by rsu or servers 
        # (not equal to unfinished. when bids are in pipeline
        # they are inactive).
        activeBid = len([x for x in self.bidlist.values() if x.isActive==1])

        # finished bids
        finishedBid = len(set([x.batch_id for x in self.bidlist.values() 
                                        if x.finishTime>x.batchCreateTime] ))
        
        # successfully finished bids
        successBid = sum([x.isSuccess for x in self.bidlist.values()])
        
        # nr bids at least admitted once (but may have failed at the end)
        # regardless if the bid is finished
        acceptedBid = len([y[0] for x in self.bidlist.values()
                            for y in x.statusHistory if y[0]==BidStatus.ADMIT])
        
        # nr bids at least rejected once (but may be successful at the end)
        # until the bid is finished
        rejectedBid = len(set([x.batch_id for x in self.bidlist.values()
                          for y in x.statusHistory if y[0]==BidStatus.REJECT 
                          and x.finishTime>x.batchCreateTime]))

        if finishedBid==0:
            finishedBid = 1
        
        # nr of bids which are finished within the recent period of time
        recentBids = [x for x in self.bidlist.values() 
                        if x.finishTime>=self.schedule.steps-recent]
        recentTotal = len(set([x.batch_id for x in recentBids]))
        if recentTotal==0:
            recentTotal = 1
        recentSuccess = sum([x.isSuccess for x in recentBids])
        recentAccept = len([y[0] for x in recentBids
                            for y in x.statusHistory if y[0]==BidStatus.ADMIT])
        recentReject = len(set([x.batch_id for x in recentBids
                            for y in x.statusHistory if y[0]==BidStatus.REJECT]))
        
        perfile.write('{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}\n'.format(
              self.schedule.steps,
              [x.currentOccupied for x in self.sites.values()],
              [x.currentUtilization for x in self.sites.values()],
              [x.maxAmounts for x in self.sites.values()],
              [x.sitetype for x in self.sites.values()],
              pmp.history_record,totalBid,activeBid,successBid,recentSuccess,
              acceptedBid,recentAccept,rejectedBid,recentReject,finishedBid,
              successBid/totalBid,acceptedBid/totalBid,1-rejectedBid/totalBid,
              recentSuccess/recentTotal,recentAccept/recentTotal,1-recentReject/recentTotal,
              pmp.critic_type+'_'+pmp.actor_type,
              rsp.serviceCapa,vp.ph,vp.totalBudget
            ))
    
    def step(self):
        nrArrivals = np.random.poisson(self.lam)
        self._createNewVehicles(nrArrivals)
        
        for agent in self.schedule.agents[:]:
            agent.step()

        for agent in self.schedule.agents[:]:
            agent.estimate()
        
        for agent in self.schedule.agents[:]:
            agent.allocate()
        
        for agent in self.schedule.agents[:]:
            agent.advance()    
        
        self._print(self.filedict[FILES.PERF_FILE[0]])
        
        self.schedule.steps += 1
        self.schedule.time += 1

        
def calWin(model):
    totalWinning = 0
    for agent in model.schedule.agents:
        if isinstance(agent, Vehicle):
            won = [x.isSuccess for x in agent.bidHistory]
            totalWinning = totalWinning + np.sum(won)
        return totalWinning


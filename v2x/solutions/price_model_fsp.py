# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch import tensor,nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from ..config.config import (PRICE_MODEL_PARAMS as pmdl, 
                             VEHICLE_PARAMS as vp, DEVICE, MODEL_DIR)

class PriceLearningModel():
    '''
    vehicle price learning model, A2C. input is competitor state, new bid
        info and environment variables. output is whether to hold bids for 
        the current round, and budget allocation to each bid. number of 
        bids sharing the same budget pool is currently fixed.
    inputs, outputs and rewards are all normalized.
    '''
    batch_size = pmdl.batch_size
    epoch = pmdl.epoch
    history_record = pmdl.history_record
    train_all_records = pmdl.train_all_records

    actor_learning_rate = pmdl.actor_learning_rate
    actor_lr_min = pmdl.actor_lr_min
    actor_lr_reduce_rate = pmdl.actor_lr_reduce_rate
    critic_learning_rate = pmdl.critic_learning_rate
    critic_lr_min = pmdl.critic_lr_min
    critic_lr_reduce_rate = pmdl.critic_lr_reduce_rate
    critic_pretrain_nr_record = pmdl.critic_pretrain_nr_record
    actor_pretrain_nr_record = pmdl.actor_pretrain_nr_record
    reward_rate = pmdl.reward_rate
    reward_min = pmdl.reward_min
    reward_reduce_rate = pmdl.reward_reduce_rate
    add_randomness = pmdl.add_randomness
    exploration = pmdl.exploration
    
    actor_type = pmdl.actor_type
    critic_type = pmdl.critic_type
    
    def __init__(self,uniqueId,dimOutput=1,evaluation=False,loadModel=False):
        self.unique_id = uniqueId + '_plm'
        self.inputVec = dict()
        self.nextStateVec = dict()
        self.output = dict()
        self.reward = dict()
        self.dimOutput = dimOutput
        self.actor = None
        self.critic = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.avg_reward = 0
        self.evaluation = evaluation
        self.loadModel = loadModel
        self.criticpath = os.path.join(MODEL_DIR,
                                       self.unique_id+'_critic_train_fsp.pkl')
        self.actorpath = os.path.join(MODEL_DIR,
                                      self.unique_id+'_actor_train_fsp.pkl')
        
        self.inputCounter = 0
        self.firstInput = 0
    
    def _initBudget(self):
        ''' random budget split if model is not available '''
        return list(np.random.rand(self.dimOutput))

    def _initActor(self,inputDim,outputDim):
        paramDim = int(outputDim + outputDim + (outputDim**2 - outputDim) / 2)
        if self.actor_type=='Actor':
            self.actor = MLP_Wrapper(inputDim,paramDim,
                            pmdl.actor_hidden_size1,pmdl.actor_hidden_size2)
        else:
            self.actor = CNNHighway(inputDim,paramDim,pmdl.actor_num_filter,
                            pmdl.actor_dropout_rate,pmdl.actor_hidden_size1,
                            pmdl.actor_hidden_size2)
        if DEVICE!=torch.device('cpu'):
            self.actor = nn.DataParallel(self.actor)
        self.actor.to(DEVICE)       
        self.actor_params = list(filter(lambda p: p.requires_grad, 
                                    self.actor.parameters()))
        self.actor_optimizer = torch.optim.SGD(self.actor_params, 
                                           lr=self.actor_learning_rate)
        if self.evaluation or self.loadModel:
            # evaluation: only run inference with previously trained model
            # loadModel: load pre-trained model
            self._reload(self.actorpath)
    
    def _initCritic(self,inputDim,outputDim):
        if self.critic_type=='Critic':
            self.critic = MLP_Wrapper(inputDim,outputDim,
                            pmdl.critic_hidden_size1,pmdl.critic_hidden_size2)
        else:
            self.critic = CNNHighway(inputDim,outputDim,pmdl.critic_num_filter,
                            pmdl.critic_dropout_rate,pmdl.critic_hidden_size1,
                            pmdl.critic_hidden_size2)            
        if DEVICE!=torch.device('cpu'):
            self.critic = nn.DataParallel(self.critic)
        self.critic.to(DEVICE)
        self.critic_params = list(filter(lambda p: p.requires_grad, 
                                    self.critic.parameters()))
        self.critic_optimizer = torch.optim.SGD(self.critic_params, 
                                           lr=self.critic_learning_rate)
        
        if self.evaluation or self.loadModel:
            # evaluation: only run inference with previously trained model
            # loadModel: load pre-trained model
            self._reload(self.criticpath)
        
    def _reload(self,path):
        try:
            checkpoint = torch.load(path)
            if path==self.criticpath:
                self.critic.load_state_dict(checkpoint)
            else:
                self.actor.load_state_dict(checkpoint)
        except:
            pass
        
    def _updateLearningRate(self):
        currentIdx = max([int(k) for k in self.reward.keys()])
        if currentIdx < self.exploration:
            critic_lr_reduce_rate = 1
            actor_lr_reduce_rate = 1
            reward_reduce_rate = 1
        else:
            critic_lr_reduce_rate = self.critic_lr_reduce_rate
            actor_lr_reduce_rate = self.actor_lr_reduce_rate
            reward_reduce_rate = self.reward_reduce_rate
        
        self.critic_learning_rate = max(self.critic_lr_min,
                        self.critic_learning_rate * critic_lr_reduce_rate)
        if self.actor is not None:            
            self.actor_learning_rate = max(self.actor_lr_min,
                        self.actor_learning_rate * actor_lr_reduce_rate)
            self.actor_optimizer = torch.optim.SGD(self.actor_params, 
                                               lr=self.actor_learning_rate)
            self.reward_rate = max(self.reward_min,
                                   self.reward_rate * reward_reduce_rate)

    def _prepInput(self,inputVec,pos=None,var=True):
        if pos is None:
            pos = list(range(len(inputVec)))
        if isinstance(inputVec,dict):
            x = np.array([v for k,v in inputVec.items() if int(k) in pos])
        else:
            x = np.array([x for i,x in enumerate(inputVec) if i in pos])
        if var:
            return Variable(tensor(x,device=DEVICE,dtype=torch.float))
        else:
            return tensor(x,device=DEVICE,dtype=torch.float)
    
    def _removeRecord(self):
        pos = str(self.firstInput)
        _ = self.inputVec.pop(pos)
        _ = self.output.pop(pos)
        _ = self.reward.pop(pos)
        _ = self.nextStateVec.pop(pos)
        self.firstInput += 1
    
    def _critic_loss_func(self,value,next_value,reward,
			  avg_reward,rate):
        advantage = reward + next_value - value
        for i in range(len(advantage)):
            advantage[i] -= avg_reward
            if not torch.isnan(advantage[i]): 
                avg_reward += rate * advantage[i].item()
        return advantage.pow(2).mean(),advantage,avg_reward

    def _createCovMat(self,diag,tril):
        z = torch.zeros(size=[diag.size(0)],
			device=DEVICE,dtype=torch.float) # with batchsize
        diag = 1E-7 + diag # strictly positive
        elements = []
        trilPointer = 0
        for i in range(diag.shape[1]):
            for j in range(diag.shape[1]):
                if j<i:
                    elements.append(tril[:,trilPointer])
                    trilPointer += 1
                elif j==i:
                    elements.append(diag[:,i])
                else:
                    elements.append(z)
        scale_tril = torch.stack(elements,dim=-1).view(-1,self.dimOutput,
                                                            self.dimOutput)
        return scale_tril

    def _actor_loss_func(self,log_prob_actions,mean,advantage):
        return (advantage.detach() * -log_prob_actions).mean()

    def _chooseAction(self,params):
        ''' 
        action space is multi-dimentional continuous variables. therefore use
            parameterized action estimators, and a multivariate gaussian 
            distribution to output joint probability of the actions. 
            parameters in this case includes N means and N*N covariance 
            matrix elements. Therefore this solution is not scalable when N
            increases. Another solution would be to use a RNN, such as in 
            https://arxiv.org/pdf/1806.00589.pdf
            or http://papers.nips.cc/paper/6398-learning-multiagent-communication-with-backpropagation.pdf
            or https://arxiv.org/pdf/1705.05035.pdf
        derivatives of a multivariate gaussian: see matrix cookbook chapter 8:
            http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3274/pdf/imm3274.pdf
        params are split into mean, covariance matrix diagonal, 
            cov matrix triangle lower half (since cov matrix is symmetric). 
            also make sure cov is positive definite. 
        '''
        mean,diag,tril = params.split([self.dimOutput,self.dimOutput,
                         params.shape[1]-2*self.dimOutput],dim=-1)
        scale_tril = self._createCovMat(diag,tril)
        dist = MultivariateNormal(loc=mean,scale_tril=scale_tril)
        # https://pytorch.org/docs/stable/distributions.html#pathwise-derivative
        actions = dist.rsample()
        log_prob_actions = dist.log_prob(actions)
        return actions,log_prob_actions,mean

    def _saveModel(self,critic=True,actor=True):
        if critic:
            torch.save(self.critic.state_dict(),self.criticpath)
        if actor:
            torch.save(self.actor.state_dict(),self.actorpath)

    def inference(self,inputVec,randomness=None):
        if self.critic is None or self.actor is None:
            return tensor(self._initBudget(),device=DEVICE,dtype=torch.float)
        if randomness is None:
            randomness = self.add_randomness * self.actor_learning_rate
        nr = np.random.rand()
        if nr<randomness:
            return tensor(self._initBudget(),device=DEVICE,dtype=torch.float)
        
        fullInput = list(self.inputVec.values())[
                                        -(self.batch_size-1):] + [inputVec]
        x = self._prepInput(inputVec=fullInput)
        self.actor.eval()
        with torch.no_grad():
            params = self.actor(x)
            actions,_,_ = self._chooseAction(params)
        return torch.clamp(actions[0],0,1)
    

    def train(self,time,plmfile):
        try:
            currentIdx = max([int(k) for k in self.reward.keys()])
        except:
            print(self.unique_id)
            currentIdx = max([int(k) for k in self.reward.keys()])
        if (currentIdx<max(self.critic_pretrain_nr_record,self.batch_size)):
            plmfile.write('{};{};{};too few data points.\n'.format(
                                                    time,self.unique_id,0))
            return
        length = len([v for k,v in self.reward.items() 
                                    if v is not None and not np.isnan(v)])
        if length>=self.train_all_records:
            length = min(length,self.history_record)
        
        pos = [int(k) for k,v in self.reward.items() 
                                    if v is not None and not np.isnan(v)]
        pos = pos[-length:]
        r = [v for k,v in self.reward.items() 
                                    if v is not None and not np.isnan(v)]
        r = tensor(r[-length:],device=DEVICE,dtype=torch.float)
        r = r + (torch.abs(r+1E-7)/100)**0.5 * torch.randn(len(r),
					device=DEVICE,dtype=torch.float)
        maxReward = torch.max(torch.abs(r))
        r = r / maxReward
        r = r.view(-1,1)
        
        x = self._prepInput(self.inputVec,pos,var=True)
        y = self._prepInput(self.nextStateVec,pos,var=False)
        try:
            assert len(x)==len(y)==len(r)
        except:
            print(self.unique_id)
            assert len(x)==len(y)==len(r)

        if self.critic is None:
            self._initCritic(x.shape[1],1)        
        if self.actor is None and currentIdx>=self.actor_pretrain_nr_record:
            self._initActor(x.shape[1],self.dimOutput)
        
        if self.actor is None:
            self._saveModel(critic=True,actor=False)
        else:
            self._saveModel(critic=True,actor=True)

        for epoch in range(self.epoch):
            pointer = 0
            epoch_loss_critic = []
            epoch_loss_actor = []
            while pointer+self.batch_size<=len(x):
                idx = range(pointer,pointer+self.batch_size)

                x_batch = x[idx]
                y_batch = y[idx]
                r_batch = r[idx]
                
                values = self.critic(x_batch)
                next_values = self.critic(y_batch)                
                
                critic_loss,advantage,self.avg_reward = self._critic_loss_func(
                  values,next_values,r_batch,self.avg_reward,self.reward_rate)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                epoch_loss_critic.append(critic_loss)
                
                if self.actor is not None:
                    action_params = self.actor(x_batch)
                    actions,log_prob_actions,mean = self._chooseAction(
                                                       action_params)
                    actor_loss = self._actor_loss_func(log_prob_actions,
                                                       mean,advantage)
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()
                    epoch_loss_actor.append(actor_loss)
                
                pointer += 1
            
            avgLossCritic = sum(epoch_loss_critic)
            if len(epoch_loss_critic) > 0:
                avgLossCritic /= len(epoch_loss_critic)
            avgLossActor = sum(epoch_loss_actor)
            if len(epoch_loss_actor) > 0:
                avgLossActor /= len(epoch_loss_actor)
                
            plmfile.write('{};{};{};{};{};{};{}\n'.format(time,
                    self.unique_id,epoch,len(x),self.avg_reward,
                    avgLossCritic, avgLossActor))
                    
            
            if avgLossCritic!=0 and torch.isnan(avgLossCritic):
                plmfile.write(
                    '{};{};{};{};{};{};{};critic restarted.\n'.format(
                    time,self.unique_id,epoch,len(x),self.avg_reward,
                    avgLossCritic,avgLossActor))
                self._initCritic(x.shape[1],1)
                self._reload(self.criticpath)
            
            if avgLossActor!=0 and torch.isnan(avgLossActor):
                plmfile.write(
                    '{};{};{};{};{};{};{};actor restarted.\n'.format(
                    time,self.unique_id,epoch,len(x),self.avg_reward,
                    avgLossCritic,avgLossActor))
                self._initActor(x.shape[1],self.dimOutput)
                self._reload(self.actorpath)
        
        self._updateLearningRate()
    
    def collectInput(self,inputVec):
        self.inputVec[str(self.inputCounter)] = inputVec
        self.inputCounter += 1
        if len(self.inputVec)>max(self.history_record+50,
                                  self.train_all_records+50):
            self._removeRecord()
        
        return str(self.inputCounter-1) # id for matching output and reward
    
    def collectNextState(self,stateVec,idx):
        envVec = self.inputVec[idx][len(stateVec):]
        nextState = stateVec + envVec
        self.nextStateVec[idx] = nextState
    
    def collectOutput(self,output,idx):
        self.output[idx] = output
        
    def collectReward(self,reward,idx):
        if (idx not in self.reward.keys() or self.reward[idx] is None 
                                          or np.isnan(self.reward[idx])):
            if idx in self.inputVec.keys():
                self.reward[idx] = reward
        else:
           self.reward[idx] += reward

class MLP(nn.Module):
    '''multilayer perceptron as another form of highway
    '''    
    def __init__(self,inputDim,outputDim,hidden_size1,hidden_size2):
        super().__init__()
        self.batchNorm = nn.BatchNorm1d(inputDim)
        self.hidden1 = nn.Linear(inputDim,hidden_size1)
        self.hidden2 = nn.Linear(hidden_size1,hidden_size2)
        self.batchNorm2  = nn.BatchNorm1d(hidden_size2)
        self.hidden3 = nn.Linear(hidden_size2,outputDim)
        
    def forward(self,x):
        batchnorm = self.batchNorm(x)
        hidden1 = F.relu(self.hidden1(batchnorm))
        hidden2 = F.relu(self.batchNorm2(self.hidden2(hidden1)))
        hidden3 = self.hidden3(hidden2)
        return hidden3


class MLP_Wrapper(nn.Module):
    '''value function estimator. sigmoid layer is used for output to
            control the output range. 
    '''    
    def __init__(self,inputDim,outputDim,hidden_size1,hidden_size2):
        super().__init__()
        self.mlp = MLP(inputDim,outputDim,hidden_size1,hidden_size2)
        self.predict = nn.Sigmoid() # output between 0 and 1
        
    def forward(self,x):
        mlp = self.mlp(x)
        predict = self.predict(mlp)
        return predict


class Highway(nn.Module):
    def __init__(self,in_features,out_features,num_layers=1,bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.bias = bias
        self.cells = nn.ModuleList()
        for idx in range(self.num_layers):
            g = nn.Sequential(
                    nn.Linear(self.in_features, self.out_features),
                    nn.ReLU(inplace=True)
                    )
            t = nn.Sequential(
                    nn.Linear(self.in_features, self.out_features),
                    nn.Sigmoid()
                    )
            self.cells.append(g)
            self.cells.append(t)
        
    def forward(self,x):
        for i in range(0,len(self.cells),2):
            g = self.cells[i]
            t = self.cells[i+1]
            nonlinearity = g(x)
            transformGate = t(x) + self.bias
            x = nonlinearity * transformGate + (1-transformGate) * x
        return x
    

class CNNHighway(nn.Module):
    filter_size = list(np.arange(1,pmdl.batch_size,step=2,dtype=int))
    
    def __init__(self,inputDim,outputDim,num_filter,dropout_rate,
                 hidden_size1,hidden_size2):
        super().__init__()

        self.num_filter = ([num_filter] 
              + [num_filter * 2] * int(len(self.filter_size)/2)
              + [num_filter] * len(self.filter_size))[0:len(self.filter_size)]

        self.num_filter_total = sum(self.num_filter)
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.seqLength = pmdl.batch_size
        self.dropout_rate = dropout_rate
        
        self.batchNorm = nn.BatchNorm1d(inputDim)
        self.convs = nn.ModuleList()
        for fsize, fnum in zip(self.filter_size, self.num_filter):
            # kernel_size = depth, height, width
            conv = nn.Sequential(
                nn.Conv2d(in_channels=1,out_channels=fnum,
                         kernel_size=(fsize,inputDim),
                         padding=0,stride=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(fnum),
                nn.MaxPool2d(kernel_size=(self.seqLength-fsize+1,1),stride=1)
                )
            self.convs.append(conv)
        
        self.highway = Highway(self.num_filter_total,self.num_filter_total,
                               num_layers=1, bias=0)
        self.mlp = MLP(inputDim,outputDim,hidden_size1,hidden_size2)
        # p: probability of an element to be zeroed. Default: 0.0
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc_conv = nn.Linear(sum(self.num_filter),outputDim)
        self.fc = nn.Linear(pmdl.batch_size*outputDim+outputDim,outputDim)
        self.predict = nn.Sigmoid()

    def forward(self,x):
        batchnorm = self.batchNorm(x)
        xs = list()
        for i,conv in enumerate(self.convs):
            x0 = conv(batchnorm.view(-1,1,self.seqLength,self.inputDim))
            x0 = x0.view((x0.shape[0],x0.shape[1]))
            xs.append(x0)
        cats = torch.cat(xs,1)
        highway = self.highway(cats)
        dropout = F.relu(self.dropout(highway))
        fc_conv = F.relu(self.fc_conv(dropout))
        mlp = self.mlp(x)
        cats2 = torch.cat([fc_conv,mlp.view(1,-1)],1)
        fc = F.relu(self.fc(cats2))
        predict = self.predict(fc)
        return predict

class SupervisedModel():

    supervise_learning_rate = pmdl.supervise_learning_rate
    supervise_hidden_size1 = pmdl.supervise_hidden_size1
    supervise_hidden_size2 = pmdl.supervise_hidden_size2

    batch_size = pmdl.batch_size
    epoch = pmdl.epoch
    pretrain_nr_record = pmdl.critic_pretrain_nr_record
    history_record = pmdl.history_record
    train_all_records = pmdl.train_all_records
    targetUpperBound = vp.totalBudget[0]
        
    def __init__(self,uniqueId,dimOutput=1,evaluation=False,loadModel=False):
        self.unique_id = uniqueId + '_supervised'
        self.inputVec = dict()
        self.output = dict()
        self.dimOutput = dimOutput
        self.evaluation = evaluation
        self.loadModel = loadModel
        self.supervise = None # the model
        self.supervisepath = os.path.join(MODEL_DIR,
                                       self.unique_id+'_train_fsp.pkl')
        self.inputCounter = 0
        self.firstInput = 0

    def _prepInput(self,inputVec,pos=None,var=True):
        if pos is None:
            pos = list(range(len(inputVec)))
        if isinstance(inputVec,dict):
            x = np.array([v for k,v in inputVec.items() if int(k) in pos])
        else:
            x = np.array([x for i,x in enumerate(inputVec) if i in pos])
        if var:
            return Variable(tensor(x,device=DEVICE,dtype=torch.float))
        else:
            return tensor(x,device=DEVICE,dtype=torch.float)

    def _removeRecord(self):
        ''' remove both the average and best response records '''
        pos = self.firstInput
        _ = self.inputVec.pop(str(pos))
        _ = self.inputVec.pop(str(pos+1))
        _ = self.output.pop(str(pos))
        _ = self.output.pop(str(pos+1))
        self.firstInput += 2

    def _initBudget(self):
        ''' random budget split if model is not available '''
        return list(np.random.rand(self.dimOutput))
        #return np.random.rand(self.dimOutput)

    def _reload(self,path):
        try:
            checkpoint = torch.load(path)
            if path==self.supervisepath:
                self.supervise.load_state_dict(checkpoint)
        except:
            pass
        
    def _saveModel(self,supervise=True):
        if supervise:
            torch.save(self.supervise.state_dict(),self.supervisepath)

    def _initSupervise(self,inputDim,outputDim):
        self.supervise = MLP_Wrapper(inputDim,outputDim,
                    self.supervise_hidden_size1,self.supervise_hidden_size2)
        if DEVICE!=torch.device('cpu'):
            self.supervise = nn.DataParallel(self.supervise)
        self.supervise.to(DEVICE)
        self.supervise_params = list(filter(lambda p: p.requires_grad, 
                                    self.supervise.parameters()))
        self.supervise_optimizer = torch.optim.SGD(self.supervise_params, 
                                           lr=self.supervise_learning_rate)
        self.loss_func = torch.nn.MSELoss()
        
        if self.evaluation or self.loadModel:
            # evaluation: only run inference with previously trained model
            # loadModel: load pre-trained model
            self._reload(self.supervisepath)

    def collectInput(self,inputVec):
        self.inputVec[str(self.inputCounter)] = inputVec
        self.inputCounter += 1
        if len(self.inputVec)>max(self.history_record+100,
                                  self.train_all_records+100):
            self._removeRecord()
        
        return str(self.inputCounter-1) # id for matching output and reward
    
    def collectBehavior(self,output,idx):
        self.output[idx] = output    
 
    def inference(self,inputVec):
        if self.supervise is None:
            return tensor(self._initBudget(),device=DEVICE,dtype=torch.float)
        x = self._prepInput(inputVec).reshape(1,-1)
        self.supervise.eval()
        actions = self.supervise(x)
        return torch.clamp(actions[0],0,1)

    def train(self,time,supervisefile):
        try:
            currentIdx = max([int(k) for k in self.output.keys()])
        except:
            print(self.unique_id)
            currentIdx = max([int(k) for k in self.output.keys()])
        if (currentIdx<max(self.pretrain_nr_record,self.batch_size)):
            supervisefile.write('{};{};{};too few data points.\n'.format(
                                                    time,self.unique_id,0))
            return
        length = len([v for k,v in self.output.items() 
                            if v[0] is not None and not np.isnan(v[0])])
        if length>=self.train_all_records:
            length = min(length,self.history_record)
        
        pos = [int(k) for k,v in self.output.items() 
                            if v[0] is not None and not np.isnan(v[0])]
        pos = pos[-length:]
        y = [v for k,v in self.output.items() 
                            if v[0] is not None and not np.isnan(v[0])]
        y = tensor(y[-length:],device=DEVICE,dtype=torch.float)
        
        x = self._prepInput(self.inputVec,pos,var=True)
        try:
            assert len(x)==len(y)
        except:
            print(self.unique_id)
            assert len(x)==len(y)

        if self.supervise is None:
            self._initSupervise(x.shape[1],self.dimOutput)
        self._saveModel()

        for epoch in range(self.epoch):
            pointer = 0
            epoch_loss_supervise = []
            while pointer+self.batch_size<=len(x):
                idx = range(pointer,pointer+self.batch_size)

                x_batch = x[idx]
                y_batch = y[idx]
                
                prediction = self.supervise(x_batch)
                supervise_loss = self.loss_func(prediction, y_batch)
                
                self.supervise_optimizer.zero_grad()
                supervise_loss.backward()
                self.supervise_optimizer.step()
                epoch_loss_supervise.append(supervise_loss)
                
                pointer += 1
            
            avgLossSupervise = sum(epoch_loss_supervise)
            if len(epoch_loss_supervise) > 0:
                avgLossSupervise /= len(epoch_loss_supervise)
                
            supervisefile.write('{};{};{};{};{}\n'.format(time,
                    self.unique_id,epoch,len(x),avgLossSupervise))
                    
            if avgLossSupervise!=0 and torch.isnan(avgLossSupervise):
                supervisefile.write(
                    '{};{};{};{};{};supervised learning restarted.\n'.format(
                    time,self.unique_id,epoch,len(x),avgLossSupervise))
                self._initSupervise(x.shape[1],self.dimOutput)
                self._reload(self.supervisepath)

    

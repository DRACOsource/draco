# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch import tensor,nn
from torch.autograd import Variable
from torch.nn import functional as F
# from torch.nn.parallel import DistributedDataParallel
from ..config.config import COMP_MODEL_PARAMS as cmdl, DEVICE, MODEL_DIR

class CompetitorLearningModel():
    ''' 
    model inside TransitionTbl class to guess competitor state transition
        probabilities and predict the next competitor state (i.e. payment
        value in each service/qos category).
    simple supervised learning model. no RL needed for guessing 
        next competitor state. input: previous competitor state and
        previous bid information, as well as env. variables. output:
        next competitor state. actual nextState is collected after 
        RSU.allocate into nextStateVec.
    inputs and targets are normalized. 
    '''
    batch_size = cmdl.batch_size
    epoch = cmdl.epoch
    learning_rate = cmdl.learning_rate
    pretrain_nr_record = cmdl.pretrain_nr_record
    history_record = cmdl.hitory_record
    
    def __init__(self,uniqueId,dimOutput=1,evaluation=False):
        self.unique_id = uniqueId + '_clm'
        self.inputVec = dict()
        self.net = None
        self.optimizer = None
        self.loss_func = None
        self.output = dict()
        self.dimOutput = dimOutput
        self.evaluation = evaluation
        self.modelpath = os.path.join(MODEL_DIR,self.unique_id+'_net.pkl')

        self.inputCounter = 0
        self.firstInput = 0
    
    def _initNet(self,inputDim,outputDim):
        self.net = Net(inputDim,outputDim)
        if DEVICE!=torch.device('cpu'):
            self.net = nn.DataParallel(self.net)
        self.net.to(DEVICE)
        # get only the trainable parameters: 
        params = list(filter(lambda p: p.requires_grad, self.net.parameters()))
        self.optimizer = torch.optim.SGD(params, lr=self.learning_rate)
        self.loss_func = torch.nn.MSELoss()
        
        if self.evaluation:
            # evaluation without training:
            self._reload()
            
    
    def _prepInput(self,inputVec,length=1):
        if isinstance(inputVec,dict):
            x = np.array([v for k,v in inputVec.items()])
        else:
            x = np.array(inputVec)
        x = Variable(tensor(x[-length:],device=DEVICE,dtype=torch.float))
        return x
    
    def _reload(self):
        try:
            checkpoint = torch.load(self.modelpath)
            self.net.load_state_dict(checkpoint)
        except:
            pass

    def _removeRecord(self):
        pos = str(self.firstInput)
        _ = self.inputVec.pop(pos)
        _ = self.output.pop(pos)
        self.firstInput += 1
    
    def _saveModel(self):
        torch.save(self.net.state_dict(),self.modelpath)
    
    def inference(self,inputVec):
        if self.net is None:
            return tensor(np.array(inputVec[0:self.dimOutput]),
				device=DEVICE,dtype=torch.float)
            
        x = self._prepInput([inputVec],length=1)
        self.net.eval()
        with torch.no_grad():
            y_pred = self.net(x)[0]
        return torch.clamp(y_pred,0,1)
    
    def train(self,time,clmfile):        
        y = np.array(list(self.output.values()))
        length = min(len(self.inputVec),len(y))
        if length < self.pretrain_nr_record:
            clmfile.write('{};{};{};too few data points.\n'.format(
                                                    time,self.unique_id,0))
            return
        length = min(length,self.history_record)
        
        x = self._prepInput(self.inputVec,length=length)
        y = tensor(y[-length:],device=DEVICE,dtype=torch.float)
        if self.net is None:
            self._initNet(x.shape[1],y.shape[1])
        
        self._saveModel()
        for epoch in range(self.epoch):
            pointer = 0
            epoch_loss = []
            while pointer<=len(x):
                if pointer+self.batch_size<=len(x):
                    idx = range(pointer,pointer+self.batch_size)
                else:
                    idx = (list(range(pointer,len(x))) 
                         + list(np.random.randint(low=0,high=len(x),
                                        size=self.batch_size+pointer-len(x))))
                x_batch = x[idx]
                y_batch = y[idx]
                y_pred = self.net(x_batch)
                loss = self.loss_func(y_pred,y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pointer = pointer + self.batch_size
                epoch_loss.append(loss.item())
            
            clmfile.write('{};{};{};{};{}\n'.format(time,self.unique_id,
                          epoch,len(x),sum(epoch_loss)/len(epoch_loss)))
            if np.isnan(sum(epoch_loss)):
                clmfile.write('{};{};{};{};{};restarted.\n'.format(
                    time,self.unique_id,epoch,len(x),
                    sum(epoch_loss)/len(epoch_loss)))
                self._initNet(x.shape[1],y.shape[1])
                self._reload()

    def collectInput(self,inputVec):
        self.inputVec[str(self.inputCounter)] = inputVec
        self.inputCounter += 1
        if len(self.inputVec)>self.history_record+5:
            self._removeRecord()
        
        return str(self.inputCounter-1) # id for matching output
    
    def collectOutput(self,output,idx):
        if int(idx) >= 0:
            self.output[idx] = output


class Net(nn.Module):
    hidden_size1 = cmdl.hidden_size1
    hidden_size2 = cmdl.hidden_size2
    def __init__(self,inputDim,outputDim):
        super(Net,self).__init__()
        self.batchNorm = nn.BatchNorm1d(inputDim)
        self.hidden1 = nn.Linear(inputDim,self.hidden_size1)
        self.hidden2 = nn.Linear(self.hidden_size1,self.hidden_size2)
        self.batchNorm2  = nn.BatchNorm1d(self.hidden_size2)
        self.predict = nn.Linear(self.hidden_size2,outputDim)
        
    def forward(self,x):
        batchnorm = self.batchNorm(x)
        hidden1 = F.relu(self.hidden1(batchnorm))
        hidden2 = F.relu(self.batchNorm2(self.hidden2(hidden1)))
        predict = self.predict(hidden2)
        return predict
    
    
    
    

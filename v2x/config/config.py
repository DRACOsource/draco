# -*- coding: utf-8 -*-

import os
import sys
import torch
torch.set_num_threads(1)

ROOT_DIR         = os.path.normpath(os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), '../..'))
LOGS_DIR         = os.path.join(ROOT_DIR, 'logs')
RESOURCES_DIR    = os.path.join(ROOT_DIR, 'resources')
MODEL_DIR        = os.path.join(ROOT_DIR, 'models')
GRAPH_DIR        = os.path.join(ROOT_DIR, 'graphs')
PSTATS_FILE      = os.path.join(LOGS_DIR, 'profile.stats')
COUNTDOWN_FILE   = os.path.join(MODEL_DIR, 'countdownHistory')
BATCHCREATE_FILE = os.path.join(MODEL_DIR, 'batchCreateHistory')
#DEVICE          = torch.device('cuda:0' if torch.cuda.is_available() 
#                               else 'cpu')
DEVICE = torch.device('cpu')
#print(DEVICE)

class FILES():
    ALLOC_FILE = ('allocfile',os.path.join(LOGS_DIR,'allocation'))
    PERF_FILE =('perfile',os.path.join(LOGS_DIR,'performance'))
    CLM_FILE = ('clmfile',os.path.join(LOGS_DIR,'competitorLearningModel'))
    PLM_FILE = ('plmfile',os.path.join(LOGS_DIR,'priceLearningModel'))
    BID_FILE = ('bidfile',os.path.join(LOGS_DIR,'finishedBids'))
    SL_FILE = ('supervisefile',os.path.join(LOGS_DIR,'supervisedLearningModel'))
#    CRITIC_FILE = ('criticfile',os.path.join(LOGS_DIR, 'plm_critic'))

class RESOURCE_SITE_PARAMS():
    serviceCapa = [(25,26),(25,26),
                   (25,26)]                 # Range of random service capacity. 
                                            # when services are created. items. 
                                            # for different types of services.
    resourceCapa = [[(0,1),(0,1),(0,1)],
                    [(100,101),(100,101),(250,251)],
                    [(200,201),(200,201),(250,251)],
                    [(500,501),(500,501),(1000,1001)],
                    [(1000,1001),(1000,1001),(2000,2001)]]
                                            # Range of random resource capacity 
                                            # when resources are created. 
                                            # Items are for different types of 
                                            # resources.
    transCost = 0.01                        # Transmission cost between sites.
    burnIn = 150                             # Frozen period for capacity
                                            # adjustment.
    lowerCapaLimit = 0.3                    # For capacity adjustment.
    upperCapaLimit = 0.7                    # For capacity adjustment.
    resProfileSelectionThreshold = 2        # Before collecting this much
                                            # of data, take user's resource
                                            # profile. After that trust 
                                            # the service's estimated profile.
    randomizeQueueTimeThres = 10            # Min sample size for getting
                                            # queue length distribution
    

class SERVICE_PARAMS():
    avgCapacityPeriod = 50                  # For calculation of average 
                                            # capacity.
    avgCostPeriod = 10                      # For calculation of average cost.
    discount = 0.8                          # Price discount of slower servers.
    utilPredictionWeights = [1/50] * 50
    capacityBuffer = 0.1                    # Buffer in capacity adjustment.
    resProfileUpdateThreshold = 20          # At least to collect so many data 
                                            # points before curve fitting 
                                            # for resNeedsEstim.
    null = 1E-7                             # Anything below is null.
 
    
class RSU_PARAMS():
    secondPriceThres = 3                    # Least number of failed bids 
                                            # record for regression
    overload = 0.1                          # % overload allowed on resource 
                                            # sites at time of allocation.

class VEHICLE_PARAMS():
    totalBudget = (2000,1500)               # Total budget at time of creation.
    minNrBids = 5                           # Minimum number of simultaneous 
                                            # bids a vehicle can activate.
                                            # Whenever active nr. bids drops
                                            # below this, new bids are 
                                            # created.
    maxNrBids = 100                         # Maximum nr of simultaneous bids
                                            # a vehicle can activate.
    totalNrBids = sys.maxsize               # Maxixmum total nr. bids a
                                            # vehicle can create.
    maxSensingDist = 1000                   # Distance for sensing the RSU.
                                            # Can be useful when there are
                                            # multiple RSUs.
    budgetRenewal = 1                       # Number of bids to share the 
                                            # available budget. 
    lifespan = (30000,50000)                # Vehicle lifespan range
    envCategory = 5                         # Discretize nr. bids as env. input
    priceCategory = 5                       # Discretize price
    competitorDataThres = 0                 # Nr bids to collect before 
                                            # creating transitionTbl
    plmTrainingThres = 3                    # Interval (in terms of new 
                                            # priceLearningModel inputs)
    lamMax = 0.6                            # Max. lambda of the exponential
                                            # distribution for 
                                            # generating new bids
                                            # e.g. 0.2 means 1 batch every 5 
                                            # time steps.
    lamChangeInterval = 100                 # Interval of changing lambda.
    lamMin = 0.02                           # Min. lambda.
    ph = 0.6                                # transition prob. high to low
    pl = 0.6                                # transition prob. low to high
    lamCategory = 5                         # Nr. of lambda categories. 
                                            # the higher the number, the more
                                            # extreme the lambdas.
    stagingMaxsize = 50                     # Maxsize for staged batches.
    stagingMaxtime = 250                    # Longest time to be staged.
    stagingThreshold = [0.6]                # An indicator of higher than 
                                            # threshold will be submitted.
    stagingPeriod = 10                      # Max time to wait until next
                                            # evaluation.
                                            
class BID_PARAMS():
    QoS1 = 50                       # High performance E2E processing time.
    QoS2 = 300                      # Low performance E2E processing time.
    servAmount1 = 1                 # Low service amount
    servAmount2 = 3                 # High service amount
    minDataSize = 300               # Min data size in bytes.
    maxDataSize = 1200              # Max data size in bytes.
    nrRebid = 5                     # Max number of rebids.

class MDL_PARAMS():
    width = 200                     # Visual param
    height = 200                    # Visual param
    rsuPos = (100,100)              # Visual param
    rsuInterval = 20                # Visual param
    resSitePos = (80,150)           # Visual param
    resSiteInterval = 20            # Visual param
    vehicleYposChoice = (50,65)     # Visual param
    vehicleInterval = 1             # Visual param
    lam = 0                         # Lambda of the poisson distribution for 
                                    # generating new vehicles.
    totalSimTime = 10000             # Simulation time
    timeForNewVehicles = 10000       # Latest time to create new vehicles
    nrSites = 4                     # Default number of sites to create
    initVehicles = 30              # Initial number of vehicles to create.
                                    # When lam==0, no new vehicles will be 
                                    # created after initialization, to give
                                    # the vehicles the chance to learn.
    nrRsu = 1                       # Default number of RSUs to create.
    recent = 350                    # Performance indicator for only
                                    # the recent periods
    
class RESOURCE_PARAMS():
    defaultMaxAmount = 200          # default maximum resource amount
    unitCost = [(2,2),(20,5)]       # cost per resource unit per time unit

class TRANSITION_PARAMS():
    trainingThres = 3               # Interval (in terms of new nr. inputs)
                                    # to train the competitorLearnModel.
    historyPeriods = 1              # Number of historical states to use 
                                    # for forecast of next state.

class COMP_MODEL_PARAMS():
    hidden_size1 = 64
    hidden_size2 = 64
    batch_size = 10
    hitory_record = 10
    epoch = 10
    learning_rate = 0.3
    pretrain_nr_record = 2          # No training until this nr. inputs. 
                                    # Afterwards train on the most recent 
                                    # history_record number of records 
                                    # every TRANSITION_PARAMS.trainingThres

class PRICE_MODEL_PARAMS():
    evaluation_start = 7000
    batch_size = 16
    history_record = 16             # total size of input
    epoch = 1
    train_all_records = 16          # Before this nr. inputs, train on all 
                                    # records. after this, train on the 
                                    # most recent history_record number of
                                    # records every 
                                    # VEHICLE_PARAMS.plmTrainingThres
    
    critic_type = 'ConvCritic'      # 'ConvCritic' or 'Critic' class
    critic_num_filter = 32
    critic_hidden_size1 = 128
    critic_hidden_size2 = 128
    critic_lr_min = 0.1
    critic_lr_reduce_rate = 0.99
    critic_learning_rate = 0.9
    critic_dropout_rate = 0.0
    reward_rate = 0.01              # Continuing tasks with function estimator 
                                    # should not use discount. 
                                    # Use average reward instead.
    reward_min = 0.01
    reward_reduce_rate = 1
    critic_pretrain_nr_record = 16 # no training until this nr. inputs
    
    actor_type = 'ConvActor'        # 'ConvActor' or 'Actor' class
    actor_num_filter = 64
    actor_hidden_size1 = 128
    actor_hidden_size2 = 128
    actor_lr_min = 0.1
    actor_lr_reduce_rate = 0.99
    actor_learning_rate = 0.9
    actor_dropout_rate = 0.0
    actor_pretrain_nr_record = 16  # no training until this nr. inputs
    
    add_randomness = 0              # Valued between 0 and 1. if greater 
                                    # than zero, then in inference function,
                                    # action is randomly chosen
                                    # when generated random number is smaller
                                    # than add_randomness * learning rate
    exploration = 16               # Before the model accumulated this 
                                    # number of records, the 
                                    # learning rate does not reduce.
                                    
    supervise_learning_rate = 0.1   # learning rate for supervised learning 
    supervise_hidden_size1 = 64     # MLP hidden layer 
    supervise_hidden_size2 = 128    # MLP hidden layer 
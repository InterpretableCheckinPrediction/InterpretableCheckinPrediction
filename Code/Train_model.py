import helper as hp
from NN_models import Model
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import numpy as np

## Args
parser = argparse.ArgumentParser()
parser.add_argument("--id",
                    default='0',
                    type=int,
                    help="ID for saving",
                    required=True)
parser.add_argument("--city_id",
                    default='1',
                    type=int,
                    help="1 for Las Vegas, 2 for Toronto and 3 for Phoenix",
                    required=True)
parser.add_argument("--radius",
                    default='100',
                    type=float,
                    help="Radius for influence measure",
                    required=True)
parser.add_argument("--dropout_rate",
                    default='0.1',
                    type=float,
                    help="Dropout for training",
                    required=True)
parser.add_argument("--initial_lr",
                    default='0.001',
                    type=float,
                    help="Initial learning rate for training",
                    required=True)
parser.add_argument("--beta_1",
                    default='0.9',
                    type=float,
                    help="Beta parameter (Adam) for training",
                    required=True)
parser.add_argument("--random_seed",
                    default='0',
                    type=int,
                    help="Random seed for computations")
args = parser.parse_args()

## Training

# Get parameters
id_str=str(args.id)
city_id=args.city_id
radius=args.radius
dropout_rate=args.dropout_rate
initial_lr=args.initial_lr
random_seed=args.random_seed
beta_1=args.beta_1
device=torch.device("cuda")
criterion=nn.MSELoss(size_average=False)

# Load data
if city_id==1:
    with open('../Data/Data_LasVegas','rb') as f:
        data=pickle.load(f)
    save_str='LasVegas'
if city_id==2:
    with open('../Data/Data_Toronto','rb') as f:
        data=pickle.load(f)
    save_str='Toronto'
if city_id==3:
    with open('../Data/Data_Phoenix','rb') as f:
        data=pickle.load(f)
    save_str='Phoenix'

# Compute influence factors on training set
if os.path.exists('../Data/Influence_Factors/influence_factors_'+str(save_str)+'_'+str(int(radius)))==0:
    hp.Compute_Influence_Factors_Train_Set(data,device,radius,save_str)

with open('../Data/Influence_Factors/influence_factors_'+str(save_str)+'_'+str(int(radius)),'rb') as f:
    influence_factors=pickle.load(f)

# Set varaibles for Training
num_of_covariates=data['X_Train'][0].shape[0]
index_of_categorial_features=3
batch_size=len(data['X_Train'])

early_stopping_bound=20
warm_up_flag=True
num_of_warmup_epochs=10
max_num_of_epochs=10000

# Set random seed
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Create Model
NN=Model(num_of_covariates,dropout_rate,data['X_Train'],data['S_Train'],data['Business_IDs_Train'],
         radius,influence_factors,index_of_categorial_features)
NN=NN.to(device)
optimizer=optim.Adam(NN.parameters(),lr=initial_lr,betas=(beta_1, 0.999))

# Train Model
training_output=hp.Training(NN,data,device,optimizer,criterion,batch_size,warm_up_flag,
                            num_of_warmup_epochs,early_stopping_bound,max_num_of_epochs,
                            id_str,random_seed,False)

# Load best model and compute validation-loss
loadeddict=torch.load("..//Data//NN_models//current_model_"+id_str)
NN.load_state_dict(loadeddict)
val_loss=hp.validationstep(NN,data,False,device,batch_size,criterion,random_seed,0)

# Save training output
with open('../Data/Training_output/training_output_'+id_str,'wb') as f:
    pickle.dump(training_output,f)
    
with open('../Data/Results/results_'+id_str,'wb') as f:
    pickle.dump(val_loss,f)
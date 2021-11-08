import helper as hp
from NN_models import BaselineModel
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
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
parser.add_argument("--batchsize",
                    default='128',
                    type=int,
                    help="Batchsize for training",
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
dropout_rate=args.dropout_rate
initial_lr=args.initial_lr
random_seed=args.random_seed
beta_1=args.beta_1
batch_size=args.batchsize

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

# Set varaibles for Training
device=torch.device("cuda")
criterion=nn.MSELoss(size_average=False)

num_of_covariates=data['X_Train'][0].shape[0]
index_of_categorial_features=3
if batch_size==-1:
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
NN=BaselineModel(num_of_covariates,dropout_rate)
NN=NN.to(device)
optimizer=optim.Adam(NN.parameters(),lr=initial_lr,betas=(beta_1, 0.999))

# Train Model
training_output=hp.Training(NN,data,device,optimizer,criterion,batch_size,warm_up_flag,
                            num_of_warmup_epochs,early_stopping_bound,max_num_of_epochs,
                            id_str,random_seed,True)

# Load best model and compute validation-loss
loadeddict=torch.load("..//Data//NN_models//current_model_"+id_str)
NN.load_state_dict(loadeddict)
val_loss=hp.validationstep(NN,data,True,device,batch_size,criterion,random_seed,0)

# Save training output
with open('../Data/Training_output/training_output_'+str(id_str),'wb') as f:
    pickle.dump(training_output,f)
    
with open('../Data/Results/results_'+id_str,'wb') as f:
    pickle.dump(val_loss,f)
import subprocess
from itertools import product
import pandas as pd
import pickle
import torch
import helper_GP as hp_GP

random_seed=0
city_ids=[1,2,3]
device=torch.device("cuda")

## Hyperparameter Tuning

radius_grid=[150.0,200.0,250.0,300.0,350.0,400.0,450.0,500.0,750.0,1000.0]
dropout_grid=[0.1,0.15,0.2]
lr_grid=[0.01,0.001,0.0001]
beta_grid=[0.9,0.99]
batchsize_grid=[128,512,-1]

df_model = pd.DataFrame(columns={'id','city_id','radius','dropout','lr','beta','val_loss'})
df_baseline = pd.DataFrame(columns={'id','city_id','dropout','lr','beta','bs','val_loss'})

id_str=0

for city_id in city_ids:

    if city_id==1:
        save_str='LasVegas'
    if city_id==2:
        save_str='Toronto'
    if city_id==3:
        save_str='Phoenix'

    # Latent customer flow model (without GP)
    for (radius, dr, lr, beta) in product(radius_grid,dropout_grid,lr_grid,beta_grid):
        
        res=subprocess.run(["python",
                        "Train_model.py",
                        "--id", str(id_str),
                        "--city_id", str(city_id),
                        "--radius", str(radius),
                        "--dropout_rate", str(dr),
                        "--initial_lr", str(lr),
                        "--beta_1", str(beta),
                        "--random_seed", str(random_seed)])
        
        print(save_str+": Finished Model with radius: "+str(radius)+", dropout: "+str(dr)+", learning rate: "+str(lr)+", beta: "+str(beta))
        
        with open('../Data/Results/results_'+str(id_str),'rb') as f:
            val_loss=pickle.load(f)
            
        df_model = df_model.append({'id' : id_str,
                                    'city_id': city_id,
                                    'radius' : radius,
                                    'dropout' : dr,
                                    'lr' : lr,
                                    'beta' : beta,
                                    'val_loss' : val_loss},ignore_index=True)
                         
        id_str+=1
    
    # Baseline NN
    for (dr, lr, beta, bs) in product(dropout_grid,lr_grid,beta_grid,batchsize_grid):
        
        res=subprocess.run(["python",
                        "Train_NN_baseline.py",
                        "--id", str(id_str),
                        "--city_id", str(city_id),
                        "--dropout_rate", str(dr),
                        "--initial_lr", str(lr),
                        "--beta_1", str(beta),
                        "--batchsize", str(bs),
                        "--random_seed", str(random_seed)])
        
        print(save_str+": Finished Baseline with dropout: "+str(dr)+", learning rate: "+str(lr)+", beta: "+str(beta)+", batchsize: "+str(bs))
        
        with open('../Data/Results/results_'+str(id_str),'rb') as f:
            val_loss=pickle.load(f)
            
        df_baseline = df_baseline.append({'id' : id_str,
                                          'city_id': city_id,
                                          'dropout' : dr,
                                          'lr' : lr,
                                          'beta' : beta,
                                          'bs': bs,
                                          'val_loss' : val_loss},ignore_index=True)
                         
        id_str+=1
        
with open('../Data/Results/results_df_model','wb') as f:
    pickle.dump(df_model,f)
with open('../Data/Results/results_df_baseline','wb') as f:
    pickle.dump(df_baseline,f)

## Find best models and compute GP
for city_id in city_ids:
    
    if city_id==1:
        save_str='LasVegas'
    if city_id==2:
        save_str='Toronto'
    if city_id==3:
        save_str='Phoenix'
    
    df_model_city=df_model[df_model.city_id==city_id]
    df_baseline_city=df_baseline[df_baseline.city_id==city_id]
    
    opt_model=df_model_city[df_model_city.val_loss==df_model_city.val_loss.min()]
    opt_model=opt_model.iloc[0]
    
    opt_baseline=df_baseline_city[df_baseline_city.val_loss==df_baseline_city.val_loss.min()]
    opt_baseline=opt_baseline.iloc[0]
    
    with open('../Data/Model_info_'+save_str,'wb') as f:
        pickle.dump(opt_model,f)
    with open('../Data/Baseline_info_'+save_str,'wb') as f:
        pickle.dump(opt_baseline,f)
    
    hp_GP.Train_GP(opt_model,False,device,random_seed)
    print(save_str+": Finished GP for LCFM")
    hp_GP.Train_GP(opt_baseline,True,device,random_seed)
    print(save_str+": Finished GP for Baseline")


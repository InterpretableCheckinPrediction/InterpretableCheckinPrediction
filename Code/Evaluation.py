import pickle
import pandas as pd
import helper_GP as hp_GP
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from NN_models import Model, BaselineModel

device=torch.device("cuda")

def Eval_Kriging_SVR(data,save_str):

    X_Test_baseline=np.array(data['X_Test'])
    Y_Test_baseline=np.array(data['Y_Test'])
    X_GP_Test=np.array(data['S_Test']).reshape(len(data['S_Test']),2)
    
    with open('../Data/Baseline_Models/Kriging_SVR_SVR_'+save_str,'rb') as f:
        Kriging_SVR_SVR=pickle.load(f)
    with open('../Data/Baseline_Models/Kriging_SVR_GP_'+save_str,'rb') as f:
        Kriging_SVR_GP=pickle.load(f)
        
    pred=Kriging_SVR_SVR.predict(X_Test_baseline)+Kriging_SVR_GP.predict(X_GP_Test)
    # If the GP_Term makes the output negative --> predict zero (Note that the target variable is scaled to [0,1])
    pred[pred<0.0]=0.0
    
    mse=mean_squared_error(Y_Test_baseline,pred)
    
    return mse

def Eval_Kriging_RF(data,save_str):

    X_Test_baseline=np.array(data['X_Test'])
    Y_Test_baseline=np.array(data['Y_Test'])
    X_GP_Test=np.array(data['S_Test']).reshape(len(data['S_Test']),2)

    with open('../Data/Baseline_Models/Kriging_RF_RF_'+save_str,'rb') as f:
        Kriging_RF_RF=pickle.load(f)
    with open('../Data/Baseline_Models/Kriging_RF_GP_'+save_str,'rb') as f:
        Kriging_RF_GP=pickle.load(f)
        
    pred=Kriging_RF_RF.predict(X_Test_baseline)+Kriging_RF_GP.predict(X_GP_Test)
    # If the GP_Term makes the output negative --> predict zero (Note that the target variable is scaled to [0,1])
    pred[pred<0.0]=0.0
    
    mse=mean_squared_error(Y_Test_baseline,pred)
    
    return mse

def Eval_Kriging_NN(data,save_str):
    
    with open('../Data/Baseline_info_'+save_str,'rb') as f:
        baseline_info=pickle.load(f)
    
    num_of_covariates=data['X_Train'][0].shape[0]
      
    ## Load Baseline (NN)
    Baseline_NN=BaselineModel(num_of_covariates,baseline_info['dropout'])
    Baseline_NN=Baseline_NN.to(device)
    loadeddict=torch.load("..//Data//NN_models//current_model_"+str(int(baseline_info['id'])))
    Baseline_NN.load_state_dict(loadeddict)
    Baseline_NN.eval()
    
    ## Load Baseline (GP)
    with open('../Data/GP_Baseline_'+save_str,'rb') as f:
        Baseline_GP=pickle.load(f)
    

    pred=hp_GP.Evaluate_Baseline_Kriging(Baseline_NN,Baseline_GP,device,data['X_Test'],data['S_Test'])
    
    mse=mean_squared_error(data['Y_Test'],pred)
    
    return mse

def prepare_input_for_GWR(X_Train_baseline,X_Validation_baseline,X_Test_baseline):
    # This function aggregates the categroical features to a single class.
    # That is, all unique combinations are determined and then each category
    # receives a unique value that is scaled to lie in the same range as the
    # non-categorical features.
    
    val_mat=np.vstack([X_Train_baseline[:,:2],X_Validation_baseline[:,:2],X_Test_baseline[:,:2]])
    X=np.vstack([X_Train_baseline[:,2:],X_Validation_baseline[:,2:],X_Test_baseline[:,2:]])
    unique_comb,ind=np.unique(X,axis=0,return_index=True)
    
    X_Train_baseline_GWR=X_Train_baseline[:,:2]
    X_Validation_baseline_GWR=X_Validation_baseline[:,:2]
    X_Test_baseline_GWR=X_Test_baseline[:,:2]
    vals=np.linspace(val_mat.min(),val_mat.max(),len(list(ind)))
    
    X_Train_baseline_GWR=np.hstack([X_Train_baseline_GWR,np.zeros((X_Train_baseline.shape[0],1))])
    X_Validation_baseline_GWR=np.hstack([X_Validation_baseline_GWR,np.zeros((X_Validation_baseline.shape[0],1))])
    X_Test_baseline_GWR=np.hstack([X_Test_baseline_GWR,np.zeros((X_Test_baseline.shape[0],1))])
    
    for i in range(X_Train_baseline.shape[0]):
        category_ind=0
        for j in range(len(list(ind))):
            if (X_Train_baseline[i,2:]==unique_comb[j,:]).all():
                category_ind=j
        X_Train_baseline_GWR[i,2]=vals[category_ind]
        
    for i in range(X_Validation_baseline.shape[0]):
        category_ind=0
        for j in range(len(list(ind))):
            if (X_Validation_baseline[i,2:]==unique_comb[j,:]).all():
                category_ind=j
        X_Validation_baseline_GWR[i,2]=vals[category_ind]
        
    for i in range(X_Test_baseline.shape[0]):
        category_ind=0
        for j in range(len(list(ind))):
            if (X_Test_baseline[i,2:]==unique_comb[j,:]).all():
                category_ind=j
        X_Test_baseline_GWR[i,2]=vals[category_ind]
        
    return [X_Train_baseline_GWR,X_Validation_baseline_GWR,X_Test_baseline_GWR]

def Eval_GWR(data,save_str):

    X_Train_baseline=np.array(data['X_Train'])
    X_Validation_baseline=np.array(data['X_Validation'])
    X_Test_baseline=np.array(data['X_Test'])
    Y_Test_baseline=np.array(data['Y_Test'])  
    
    [X_Train_baseline_GWR,X_Validation_baseline_GWR,X_Test_baseline_GWR]=prepare_input_for_GWR(X_Train_baseline,X_Validation_baseline,X_Test_baseline)
    
    loc_Test=np.array(data['S_Test']).reshape(len(data['S_Test']),2)  

    with open('../Data/Baseline_Models/GWR_'+save_str,'rb') as f:
        GWR=pickle.load(f)
    with open('../Data/Baseline_Models/GWR_params_'+save_str,'rb') as f:
        [scale,residuals]=pickle.load(f)
        
    pred_results = GWR.predict(loc_Test, X_Test_baseline_GWR, scale, residuals)
    pred=pred_results.predictions
    
    mse=mean_squared_error(Y_Test_baseline,pred)
    
    return mse

def Eval_GEO_LR(data,save_str):    

    with open('../Data/Baseline_Models/GEO_LR_'+save_str,'rb') as f:
        GEO_LR=pickle.load(f)
    with open('../Data/Baseline_Models/GEO_LR_radius_'+save_str,'rb') as f:
        GEO_LR_radius=pickle.load(f)
    with open('../Data/Geographic_features/Data_'+save_str+'_'+str(int(GEO_LR_radius)),'rb') as f:
        geo_data=pickle.load(f)
        
    pred=GEO_LR.predict(geo_data['geo_features_Test'])
    
    mse=mean_squared_error(data['Y_Test'],pred)
    
    return mse
 
def Eval_GEO_DT(data,save_str):    
    
    with open('../Data/Baseline_Models/GEO_DT_'+save_str,'rb') as f:
        GEO_DT=pickle.load(f)
    with open('../Data/Baseline_Models/GEO_DT_radius_'+save_str,'rb') as f:
        GEO_DT_radius=pickle.load(f)
    with open('../Data/Geographic_features/Data_'+save_str+'_'+str(int(GEO_DT_radius)),'rb') as f:
        geo_data=pickle.load(f)
        
    pred=GEO_DT.predict(geo_data['geo_features_Test'])
    
    mse=mean_squared_error(data['Y_Test'],pred)
    
    return mse

def Eval_GEO_SVR(data,save_str):

    with open('../Data/Baseline_Models/GEO_SVR_'+save_str,'rb') as f:
        GEO_SVR=pickle.load(f)
    with open('../Data/Baseline_Models/GEO_SVR_radius_'+save_str,'rb') as f:
        GEO_SVR_radius=pickle.load(f)
    with open('../Data/Geographic_features/Data_'+save_str+'_'+str(int(GEO_SVR_radius)),'rb') as f:
        geo_data=pickle.load(f)
        
    pred=GEO_SVR.predict(geo_data['geo_features_Test'])
    
    mse=mean_squared_error(data['Y_Test'],pred)
    
    return mse

def Eval_LCFM(data,save_str):
    
    with open('../Data/Model_info_'+save_str,'rb') as f:
        model_info=pickle.load(f)
        
    with open('../Data/Influence_Factors/influence_factors_'+str(save_str)+'_'+str(int(model_info['radius'])),'rb') as f:
        influence_factors=pickle.load(f)
    
    num_of_covariates=data['X_Train'][0].shape[0]
    num_of_categorial_features=3
    
    ## Load LCFM (NN)
    NN=Model(num_of_covariates,model_info['dropout'],data['X_Train'],data['S_Train'],data['Business_IDs_Train'],
         model_info['radius'],influence_factors,num_of_categorial_features)
    NN=NN.to(device)
    loadeddict=torch.load("..//Data//NN_models//current_model_"+str(int(model_info['id'])))
    NN.load_state_dict(loadeddict)
    NN.eval()
    
    ## Load LCFM (GP)
    with open('../Data/GP_'+save_str,'rb') as f:
        GP=pickle.load(f)
    
    pred=hp_GP.Evaluate_LCFM(NN,GP,device,data['X_Test'],data['S_Test'])
    
    mse=mean_squared_error(data['Y_Test'],pred)
    
    return mse


# =============================================================================
# Evaluate
# =============================================================================

city_ids=[1,2,3]
round_digits=4
results = pd.DataFrame(columns={'city','model','mse','rmse'})

for city_id in city_ids:
    
    ## Load data
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

    mse=Eval_Kriging_SVR(data,save_str)
    results = results.append({'city' : save_str,
                              'model' : 'Kriging_SVR',
                              'mse' : round(mse,round_digits),
                              'rmse': round(np.sqrt(mse),round_digits)},ignore_index=True)
    print(save_str+": evaluated SVR Kriging.")
    
    mse=Eval_Kriging_RF(data,save_str)
    results = results.append({'city' : save_str,
                              'model' : 'Kriging_RF',
                              'mse' : round(mse,round_digits),
                              'rmse': round(np.sqrt(mse),round_digits)},ignore_index=True)
    print(save_str+": evaluated RF Kriging.")
    
    mse=Eval_Kriging_NN(data,save_str)
    results = results.append({'city' : save_str,
                              'model' : 'Kriging_NN',
                              'mse' : round(mse,round_digits),
                              'rmse': round(np.sqrt(mse),round_digits)},ignore_index=True)
    print(save_str+": evaluated NN Kriging.")
    
    mse=Eval_GWR(data,save_str)
    results = results.append({'city' : save_str,
                              'model' : 'GWR',
                              'mse' : round(mse,round_digits),
                              'rmse': round(np.sqrt(mse),round_digits)},ignore_index=True)
    print(save_str+": evaluated GWR.")
    
    mse=Eval_GEO_LR(data,save_str)
    results = results.append({'city' : save_str,
                              'model' : 'GEO_LR',
                              'mse' : round(mse,round_digits),
                              'rmse': round(np.sqrt(mse),round_digits)},ignore_index=True)
    print(save_str+": evaluated GEO LR.")
    
    mse=Eval_GEO_DT(data,save_str)
    results = results.append({'city' : save_str,
                              'model' : 'GEO_DT',
                              'mse' : round(mse,round_digits),
                              'rmse': round(np.sqrt(mse),round_digits)},ignore_index=True)
    print(save_str+": evaluated GEO DT.")
    
    mse=Eval_GEO_SVR(data,save_str)
    results = results.append({'city' : save_str,
                              'model' : 'GEO_SVR',
                              'mse' : round(mse,round_digits),
                              'rmse': round(np.sqrt(mse),round_digits)},ignore_index=True)
    print(save_str+": evaluated GEO SVR.")
    
    mse=Eval_LCFM(data,save_str)
    results = results.append({'city' : save_str,
                              'model' : 'LCFM',
                              'mse' : round(mse,round_digits),
                              'rmse': round(np.sqrt(mse),round_digits)},ignore_index=True)
    print(save_str+": evaluated LCFM.")
    
print(results[results.city=='LasVegas'])
print(results[results.city=='Phoenix'])
print(results[results.city=='Toronto'])
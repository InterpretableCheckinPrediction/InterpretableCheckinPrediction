from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel as WK
import pickle
import torch
import numpy as np

import helper as hp
from NN_models import Model, BaselineModel

def Train_GP(opt_model,baseline_flag,device,random_seed):
    
    city_id=int(opt_model['city_id'])
    if not baseline_flag:
        radius=float(opt_model['radius'])
    dropout=float(opt_model['dropout'])
    id_str=str(int(opt_model['id']))
    
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
    
    if not baseline_flag:
        with open('../Data/Influence_Factors/influence_factors_'+str(save_str)+'_'+str(int(radius)),'rb') as f:
            influence_factors=pickle.load(f)
    
    ## Load model
    num_of_covariates=data['X_Train'][0].shape[0]
    index_of_categorial_features=3
    
    if baseline_flag:
        NN=BaselineModel(num_of_covariates,dropout)
        NN=NN.to(device)
        loadeddict=torch.load("..//Data//NN_models//current_model_"+id_str)
        NN.load_state_dict(loadeddict)
        NN.eval()
    else:
        NN=Model(num_of_covariates,dropout,data['X_Train'],data['S_Train'],data['Business_IDs_Train'],
             radius,influence_factors,index_of_categorial_features)
        NN=NN.to(device)
        loadeddict=torch.load("..//Data//NN_models//current_model_"+id_str)
        NN.load_state_dict(loadeddict)
        NN.eval()

    ## Compute residuals
    if baseline_flag:
        batchsize=2000
        batch=hp.get_batch(data['X_Train'],[],data['Y_Train'],[],batchsize,random_seed,True,0)
        residuals=[]
        with torch.no_grad():
            for (X,Y) in batch:
                X, Y = X.to(device), Y.to(device)
                output=NN(X)
                residuals.append(Y-output)
    else:
        batchsize=2000
        batch=hp.get_batch(data['X_Train'],data['S_Train'],data['Y_Train'],data['Business_IDs_Train'],batchsize,random_seed,False,0)
        residuals=[]
        with torch.no_grad():
            for (X,S,Y,ID) in batch:
                X, S, Y = X.to(device), S.to(device), Y.to(device)
                output=NN(X,S,ID)
                residuals.append(Y-output)
    
    ## Compute GP input
    X_GP=np.array(data['S_Train']).reshape(len(data['S_Train']),2)
    Y_GP=np.zeros((len(data['S_Train']),1))

    if len(residuals)>1:
        for i in range(len(residuals)-1):
            Y_GP[i*batchsize:(i+1)*batchsize,0]=np.array(residuals[i].cpu()).reshape(residuals[i].shape[0])
        Y_GP[(i+1)*batchsize:,0]=np.array(residuals[i+1].cpu()).reshape(residuals[i+1].shape[0])
    else:
        Y_GP=np.array(residuals[0].cpu()).reshape(residuals[0].shape[0])
    
    ## Fit GP
    kernel=C(1e-3) * RBF([1e-1,1e-1],(1e-8,1e0))+WK(0.2)
    GP=GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,random_state=random_seed)
    GP=GP.fit(X_GP,Y_GP)
    
    if baseline_flag:
        with open('../Data/GP_Baseline_'+save_str,'wb') as f:
            pickle.dump(GP,f)
    else:
        with open('../Data/GP_'+save_str,'wb') as f:
            pickle.dump(GP,f)
        
def Evaluate_LCFM(nn_model,gp_model,device,X,S):
    
    ID=['NewID' for i in range(len(S))]
    X_tensor=torch.stack([torch.Tensor(x) for x in X])
    S_tensor=torch.stack([torch.Tensor(s[0]) for s in S])
    
    X_tensor, S_tensor = X_tensor.to(device), S_tensor.to(device)    
    S_input_for_GP=np.reshape(np.array(np.ravel(S)),(int(np.shape(np.array(np.ravel(S)))[0]/2),2))
    
    with torch.no_grad():
        NN_Term=nn_model(X_tensor,S_tensor,ID)
    
    NN_Term=np.array(NN_Term.cpu())
    GP_Term=np.array(gp_model.predict(S_input_for_GP)).reshape((-1,1))
    Value=NN_Term+GP_Term
    # If the GP_Term makes the output negative --> predict zero (Note that the target variable is scaled to [0,1])
    Value[Value<0]=0
    
    return Value

def Evaluate_Baseline_Kriging(nn_model,gp_model,device,X,S):
    
    X_tensor=torch.stack([torch.Tensor(x) for x in X])
    X_tensor = X_tensor.to(device)
    
    S_input_for_GP=np.reshape(np.array(np.ravel(S)),(int(np.shape(np.array(np.ravel(S)))[0]/2),2))
    
    with torch.no_grad():
        NN_Term=nn_model(X_tensor)
    
    NN_Term=np.array(NN_Term.cpu())
    GP_Term=np.array(gp_model.predict(S_input_for_GP)).reshape((-1,1))
    Value=NN_Term+GP_Term
    # If the GP_Term makes the output negative --> predict zero (Note that the target variable is scaled to [0,1])
    Value[Value<0]=0
    
    return Value
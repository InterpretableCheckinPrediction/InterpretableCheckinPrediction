import numpy as np
import pickle
import helper as hp
import math
import os
import pandas as pd
from itertools import product
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel as WK
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import Gaussian

# =============================================================================
# Kriging
# =============================================================================

def Train_SVR_Kriging_Baseline(data,random_seed,save_str):

    X_Train_baseline=np.array(data['X_Train'])
    X_Validation_baseline=np.array(data['X_Validation'])
    Y_Train_baseline=np.array(data['Y_Train'])
    Y_Validation_baseline=np.array(data['Y_Validation'])
        
    ## SVR
    min_loss_svr=float("inf")
    kernels=['rbf','linear','poly']
    costs=[1e-3,1e-2,1e-1,1.0,1e1,1e2,1e3]
    for (kernel,cost) in product(kernels,costs):
        svr_model=SVR(kernel=kernel,C=cost,gamma='auto')
        svr_model.fit(X_Train_baseline,np.ravel(Y_Train_baseline))
        current_loss=hp.calculate_MSE(svr_model,X_Validation_baseline,np.ravel(Y_Validation_baseline),False)
        if current_loss<min_loss_svr:
            min_loss_svr=current_loss
            best_svr_model=svr_model
    
    pred=best_svr_model.predict(X_Train_baseline)
    
    X_GP_Train=np.array(data['S_Train']).reshape(len(data['S_Train']),2)
    X_GP_Validation=np.array(data['S_Validation']).reshape(len(data['S_Validation']),2)
    Y_GP=pred-np.ravel(Y_Train_baseline)
    
    kernel=C(1e-3) * RBF([1e-1,1e-1],(1e-8,1e0))+WK(0.2)
    GP1=GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,random_state=random_seed)
    GP1=GP1.fit(X_GP_Train,Y_GP)
    
    kernel=C()*RBF()+WK()
    GP2=GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,random_state=random_seed)
    GP2=GP2.fit(X_GP_Train,Y_GP)
    
    pred1=best_svr_model.predict(X_Validation_baseline)+GP1.predict(X_GP_Validation)
    # If the GP_Term makes the output negative --> predict zero (Note that the target variable is scaled to [0,1])
    pred1[pred1<0.0]=0.0
    pred2=best_svr_model.predict(X_Validation_baseline)+GP2.predict(X_GP_Validation)
    # If the GP_Term makes the output negative --> predict zero (Note that the target variable is scaled to [0,1])
    pred2[pred2<0.0]=0.0
    
    Kriging_SVR_val_loss_1=hp.calculate_MSE(np.nan,pred1,np.ravel(Y_Validation_baseline),True)
    Kriging_SVR_val_loss_2=hp.calculate_MSE(np.nan,pred2,np.ravel(Y_Validation_baseline),True)
    
    if Kriging_SVR_val_loss_1<Kriging_SVR_val_loss_2:
        with open('../Data/Baseline_Models/Kriging_SVR_SVR_'+save_str,'wb') as f:
            pickle.dump(best_svr_model,f)
        with open('../Data/Baseline_Models/Kriging_SVR_GP_'+save_str,'wb') as f:
            pickle.dump(GP1,f)
    else:
        with open('../Data/Baseline_Models/Kriging_SVR_SVR_'+save_str,'wb') as f:
            pickle.dump(best_svr_model,f)
        with open('../Data/Baseline_Models/Kriging_SVR_GP_'+save_str,'wb') as f:
            pickle.dump(GP2,f)

def Train_RF_Kriging_Baseline(data,random_seed,save_str):

    X_Train_baseline=np.array(data['X_Train'])
    X_Validation_baseline=np.array(data['X_Validation'])
    Y_Train_baseline=np.array(data['Y_Train'])
    Y_Validation_baseline=np.array(data['Y_Validation'])

    min_loss_randomforest=float("inf")
    treenumbers=[100,200,500]
    maximumdepths=[2,5,10,50,None]
    numbersofrandomlysampledvariables=[1,3,5,10]
    for (num_of_trees,max_depth,num_var) in product(treenumbers,maximumdepths,numbersofrandomlysampledvariables):
        randomforest_model=RandomForestRegressor(n_estimators=num_of_trees, max_depth=max_depth, min_samples_leaf=num_var, random_state=random_seed)
        randomforest_model.fit(X_Train_baseline,np.ravel(Y_Train_baseline))
        current_loss=hp.calculate_MSE(randomforest_model,X_Validation_baseline,np.ravel(Y_Validation_baseline),False)
        if current_loss<min_loss_randomforest:
            min_loss_randomforest=current_loss
            best_randomforest_model=randomforest_model
            
    pred=best_randomforest_model.predict(X_Train_baseline)
    
    X_GP_Train=np.array(data['S_Train']).reshape(len(data['S_Train']),2)
    X_GP_Validation=np.array(data['S_Validation']).reshape(len(data['S_Validation']),2)
    Y_GP=pred-np.ravel(Y_Train_baseline)
    
    kernel=C(1e-3) * RBF([1e-1,1e-1],(1e-8,1e0))+WK(0.2)
    GP1=GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,random_state=random_seed)
    GP1=GP1.fit(X_GP_Train,Y_GP)
    
    kernel=C()*RBF()+WK()
    GP2=GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,random_state=random_seed)
    GP2=GP2.fit(X_GP_Train,Y_GP)
    
    pred1=best_randomforest_model.predict(X_Validation_baseline)+GP1.predict(X_GP_Validation)
    # If the GP_Term makes the output negative --> predict zero (Note that the target variable is scaled to [0,1])
    pred1[pred1<0.0]=0.0
    pred2=best_randomforest_model.predict(X_Validation_baseline)+GP2.predict(X_GP_Validation)
    # If the GP_Term makes the output negative --> predict zero (Note that the target variable is scaled to [0,1])
    pred2[pred2<0.0]=0.0
    
    Kriging_RF_val_loss_1=hp.calculate_MSE(np.nan,pred1,np.ravel(Y_Validation_baseline),True)
    Kriging_RF_val_loss_2=hp.calculate_MSE(np.nan,pred2,np.ravel(Y_Validation_baseline),True)
    
    if Kriging_RF_val_loss_1<Kriging_RF_val_loss_2:
        with open('../Data/Baseline_Models/Kriging_RF_RF_'+save_str,'wb') as f:
            pickle.dump(best_randomforest_model,f)
        with open('../Data/Baseline_Models/Kriging_RF_GP_'+save_str,'wb') as f:
            pickle.dump(GP1,f)
    else:
        with open('../Data/Baseline_Models/Kriging_RF_RF_'+save_str,'wb') as f:
            pickle.dump(best_randomforest_model,f)
        with open('../Data/Baseline_Models/Kriging_RF_GP_'+save_str,'wb') as f:
            pickle.dump(GP2,f)

# =============================================================================
# GWR
# =============================================================================

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
    
def Train_GWR_Baseline(data,save_str):
    
    X_Train_baseline=np.array(data['X_Train'])
    X_Validation_baseline=np.array(data['X_Validation'])
    X_Test_baseline=np.array(data['X_Test'])
    Y_Train_baseline=np.array(data['Y_Train'])
    
    [X_Train_baseline_GWR,X_Validation_baseline_GWR,X_Test_baseline_GWR]=prepare_input_for_GWR(X_Train_baseline,X_Validation_baseline,X_Test_baseline)
    
    loc_Train=np.array(data['S_Train']).reshape(len(data['S_Train']),2)
    coords = list(zip(loc_Train[:,0],loc_Train[:,1]))
    
    bw = Sel_BW(coords,Y_Train_baseline,X_Train_baseline_GWR,family=Gaussian(),spherical=False,kernel='gaussian').search(criterion='CV')
    model = GWR(coords,Y_Train_baseline,X_Train_baseline_GWR,bw)
    gwr_results = model.fit()
    
    scale = gwr_results.scale
    residuals = gwr_results.resid_response
    
    with open('../Data/Baseline_Models/GWR_'+save_str,'wb') as f:
        pickle.dump(model,f)
    with open('../Data/Baseline_Models/GWR_params_'+save_str,'wb') as f:
        pickle.dump([scale,residuals],f)

# =============================================================================
# Geographic Features
# =============================================================================

def compute_distance(s_1,s_2):
    lat1=s_1[0]
    lat2=s_2[0]
    lon1=s_1[1]
    lon2=s_2[1]
    dlon=hp.deg2rad(lon2-lon1)
    dlat=hp.deg2rad(lat2-lat1)
    a=(math.sin(dlat/2))**2 + math.cos(hp.deg2rad(lat1)) * math.cos(hp.deg2rad(lat2))*(math.sin(dlon/2))**2
    c = 2 * math.atan2( math.sqrt(a), math.sqrt(1-a) ) 
    distance = 6373 * c # distance in kilometers
    distance= 1000 * distance # distance in meters
    return distance

def compute_num_of_neighbours_with_given_category(X,category,current_loc_within_radius):
    N=0
    for i in current_loc_within_radius:
        if X[i,2]==category:
            N=N+1
    return N
    
def compute_geographic_features(data,radius,save_str):

    ## Use GWR data as it has aggregated unique categories
    X_Train_baseline=np.array(data['X_Train'])
    X_Validation_baseline=np.array(data['X_Validation'])
    X_Test_baseline=np.array(data['X_Test'])
    [X_Train_baseline_GWR,X_Validation_baseline_GWR,X_Test_baseline_GWR]=prepare_input_for_GWR(X_Train_baseline,X_Validation_baseline,X_Test_baseline)
    # Combine all sets
    X=np.vstack([X_Train_baseline_GWR,X_Validation_baseline_GWR,X_Test_baseline_GWR])
    # Compute categories
    categories=list(np.unique(X[:,2]))
    # Get locations
    loc_Train=np.array(data['S_Train']).reshape(len(data['S_Train']),2)
    loc_Validation=np.array(data['S_Validation']).reshape(len(data['S_Validation']),2)
    loc_Test=np.array(data['S_Test']).reshape(len(data['S_Test']),2)
    loc=np.vstack([loc_Train,loc_Validation,loc_Test])
    
    ## Compute distance matrix
    D=np.zeros((loc.shape[0],loc.shape[0]))
    for i in range(loc.shape[0]):
        for j in range(loc.shape[0]):
            D[i,j]=compute_distance(loc[i,:],loc[j,:])
    
    ## Compute locations within radius for each location    
    loc_within_radius=[]
    for i in range(loc.shape[0]):
        locs=[]
        for j in range(loc.shape[0]):
            if D[i,j]<radius and i!=j:
                locs.append(j)
        loc_within_radius.append(locs)
    
    ## Compute N
    N=np.zeros(X.shape[0])
    for ind in range(X.shape[0]):
        N[ind]=len(loc_within_radius[ind])
    
    ## Compute N_gamma
    N_gamma=np.zeros((X.shape[0],len(categories)))
    for ind in range(X.shape[0]):
        ind2=0
        current_loc_within_radius=loc_within_radius[ind]
        for category in categories:
            N_gamma[ind,ind2]=compute_num_of_neighbours_with_given_category(X,category,current_loc_within_radius)
            ind2+=1
    
    ## Compute total places of given category and total places
    total_places_per_category=np.zeros(len(categories))
    for i in range(len(categories)):
        current_category=categories[i]
        total_places_per_category[i]=np.where(X[:,2]==current_category)[0].shape[0]
    total_places=total_places_per_category.sum()
    
    ## Compute average venue appearance per categories
    average_venue_appearance=np.zeros((len(categories),len(categories)))
    for i in range(len(categories)):
        current_category=categories[i]
        places_ind_of_current_category=list(np.where(X[:,2]==current_category)[0])
        for j in range(len(categories)):
            average_venue_appearance[i,j]=np.mean(N_gamma[places_ind_of_current_category,j])
    
    ## Compute kappa
    kappa=np.zeros((len(categories),len(categories)))
    for p in range(len(categories)):
        for l in range(len(categories)):
            kappa[p,l]=(total_places-total_places_per_category[p])/(total_places_per_category[p]*total_places_per_category[l])
            
            val=0
            places_ind_of_current_category=list(np.where(X[:,2]==categories[p])[0])
            for p2 in places_ind_of_current_category:
                if N[p2]-N_gamma[p2,p]!=0.0:
                    val+=N_gamma[p2,l]/(N[p2]-N_gamma[p2,p])
                    
            kappa[p,l]=kappa[p,l]*val
            
    ## Geographic features
    
    # Density
    density=N
    
    # Neighbours Entropy
    neighbours_entropy=np.zeros(X.shape[0])
    
    for i in range(X.shape[0]):
        help_vec=np.zeros(len(categories))
        for j in range(len(categories)):
            if N[i]==0.0:
                N_gamma_N=0.0
            else:
                N_gamma_N=N_gamma[i,j]/N[i]
            
            if N_gamma_N==0.0:
                help_vec[j]=0.0
            else:
                help_vec[j]=N_gamma_N*np.log(N_gamma_N)
        neighbours_entropy[i]=-help_vec.sum()
        
    # Competitivness
    competitivness=np.zeros(X.shape[0])
    
    for i in range(X.shape[0]):
        category_ind=int(np.where(categories==X[i,2])[0])
        if N[i]==0.0:
            competitivness[i]=0.0
        else:
            competitivness[i]=-N_gamma[i,category_ind]/N[i]
            
    # Quality by Jensen
    quality=np.zeros(X.shape[0])
    
    for i in range(X.shape[0]):
        category_ind=int(np.where(categories==X[i,2])[0])
        help_vec=np.zeros(len(categories))
        for j in range(len(categories)):
            help_vec[j]=N_gamma[i,j]-average_venue_appearance[category_ind,j]
            if kappa[j,category_ind]!=0.0:
                help_vec[j]*=np.log(kappa[j,category_ind])
            else:
                help_vec[j]=0.0
        quality[i]=help_vec.sum()
    
    ## Scale geo features and append to non categorical features
    geo_features=np.hstack([X[:,:2],density.reshape(-1,1),
                            neighbours_entropy.reshape(-1,1),
                            competitivness.reshape(-1,1),
                            quality.reshape(-1,1)])
    
    geo_features_Train=geo_features[:X_Train_baseline.shape[0],:]
    geo_features_Validation=geo_features[X_Train_baseline.shape[0]:X_Train_baseline.shape[0]+X_Validation_baseline.shape[0],:]
    geo_features_Test=geo_features[X_Train_baseline.shape[0]+X_Validation_baseline.shape[0]:,:]
    
    scale_inst_X_geo=RobustScaler()
    scale_inst_X_geo.fit(geo_features_Train)
    geo_features_Train=scale_inst_X_geo.transform(geo_features_Train)
    geo_features_Validation=scale_inst_X_geo.transform(geo_features_Validation)
    geo_features_Test=scale_inst_X_geo.transform(geo_features_Test)
    
    geo_data={'geo_features_Train': geo_features_Train,
              'geo_features_Validation': geo_features_Validation,
              'geo_features_Test': geo_features_Test}
    
    with open('../Data/Geographic_features/Data_'+save_str+'_'+str(int(radius)),'wb') as f:
        pickle.dump(geo_data,f)

def Train_geo_baseline(data,save_str):
    
    radius_grid=[150.0,200.0,250.0,300.0,350.0,400.0,450.0,500.0,750.0,1000.0]
    
    df = pd.DataFrame(columns={'model','radius','val_loss','hp1','hp2'})
    
    for radius in radius_grid:
    
        if os.path.exists('../Data/Geographic_features/Data_'+save_str+'_'+str(int(radius)))==0:
            compute_geographic_features(data,radius,save_str)
        
        with open('../Data/Geographic_features/Data_'+save_str+'_'+str(int(radius)),'rb') as f:
            geo_data=pickle.load(f)
        
        ## Linear Model
        lr_model=LinearRegression()
        lr_model.fit(geo_data['geo_features_Train'],np.ravel(data['Y_Train']))
        current_loss=hp.calculate_MSE(lr_model,geo_data['geo_features_Validation'],np.ravel(data['Y_Validation']),False)
        
        df = df.append({'model' : 'lr',
                        'radius' : radius,
                        'val_loss' : current_loss,
                        'hp1': np.nan,
                        'hp2': np.nan},ignore_index=True)
        
        print(save_str+" (Geographic Features) Finished linear regression with radius: "+str(int(radius)))
                        
        ## Decision Tree
        dt_model=tree.DecisionTreeRegressor()
        dt_model.fit(geo_data['geo_features_Train'],np.ravel(data['Y_Train']))
        current_loss=hp.calculate_MSE(dt_model,geo_data['geo_features_Validation'],np.ravel(data['Y_Validation']),False)
        
        df = df.append({'model' : 'dt',
                        'radius' : radius,
                        'val_loss' : current_loss,
                        'hp1': np.nan,
                        'hp2': np.nan},ignore_index=True)
        
        print(save_str+" (Geographic Features) Finished decision tree with radius: "+str(int(radius)))
                
        ## SVR
        min_loss_svr=float("inf")
        kernels=['rbf','linear','poly']
        costs=[1e-3,1e-2,1e-1,1.0,1e1,1e2,1e3]
        for (kernel,cost) in product(kernels,costs):
            svr_model=SVR(kernel=kernel,C=cost,gamma='auto')
            svr_model.fit(geo_data['geo_features_Train'],np.ravel(data['Y_Train']))
            current_loss=hp.calculate_MSE(svr_model,geo_data['geo_features_Validation'],np.ravel(data['Y_Validation']),False)
            if current_loss<min_loss_svr:
                min_loss_svr=current_loss
                hp1=kernel
                hp2=cost
                
        df = df.append({'model' : 'svr',
                        'radius' : radius,
                        'val_loss' : min_loss_svr,
                        'hp1': hp1,
                        'hp2': hp2},ignore_index=True)
        
        print(save_str+" (Geographic Features) Finished SVR with radius: "+str(int(radius)))
                    
    ## Save linear model
    current_df=df[df.model=='lr']
    current_df=current_df[current_df.val_loss==current_df.val_loss.min()].iloc[0]
    best_radius=current_df.radius
    
    with open('../Data/Geographic_features/Data_'+save_str+'_'+str(int(best_radius)),'rb') as f:
            geo_data=pickle.load(f)
    
    lr_model=LinearRegression()
    lr_model.fit(geo_data['geo_features_Train'],np.ravel(data['Y_Train']))
    
    with open('../Data/Baseline_Models/GEO_LR_'+save_str,'wb') as f:
        pickle.dump(lr_model,f)
    with open('../Data/Baseline_Models/GEO_LR_radius_'+save_str,'wb') as f:
        pickle.dump(best_radius,f)
    
    ## Save decision tree
    current_df=df[df.model=='dt']
    current_df=current_df[current_df.val_loss==current_df.val_loss.min()].iloc[0]
    best_radius=current_df.radius
    
    with open('../Data/Geographic_features/Data_'+save_str+'_'+str(int(best_radius)),'rb') as f:
            geo_data=pickle.load(f)
    
    dt_model=tree.DecisionTreeRegressor()
    dt_model.fit(geo_data['geo_features_Train'],np.ravel(data['Y_Train']))
    
    with open('../Data/Baseline_Models/GEO_DT_'+save_str,'wb') as f:
        pickle.dump(dt_model,f)
    with open('../Data/Baseline_Models/GEO_DT_radius_'+save_str,'wb') as f:
        pickle.dump(best_radius,f)
    
    ## Save SVR
    current_df=df[df.model=='svr']
    current_df=current_df[current_df.val_loss==current_df.val_loss.min()].iloc[0]
    best_radius=current_df.radius
    best_hp1=current_df.hp1
    best_hp2=current_df.hp2
    
    with open('../Data/Geographic_features/Data_'+save_str+'_'+str(int(best_radius)),'rb') as f:
            geo_data=pickle.load(f)
    
    svr_model=SVR(kernel=best_hp1,C=best_hp2,gamma='auto')
    svr_model.fit(geo_data['geo_features_Train'],np.ravel(data['Y_Train']))
    
    with open('../Data/Baseline_Models/GEO_SVR_'+save_str,'wb') as f:
        pickle.dump(svr_model,f)
    with open('../Data/Baseline_Models/GEO_SVR_radius_'+save_str,'wb') as f:
        pickle.dump(best_radius,f)

# =============================================================================
# Train
# =============================================================================

city_ids=[1,2,3]
random_seed=0

for city_id in city_ids:

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
    
    Train_SVR_Kriging_Baseline(data,random_seed,save_str)
    print(save_str+" (Kriging) Finished SVR")
    
    Train_RF_Kriging_Baseline(data,random_seed,save_str)
    print(save_str+" (Kriging) Finished RF")
    
    Train_GWR_Baseline(data,save_str)
    print(save_str+" (GWR) Finished GWR")
    
    Train_geo_baseline(data,save_str)
    print(save_str+" (Geographic Features) Finished")
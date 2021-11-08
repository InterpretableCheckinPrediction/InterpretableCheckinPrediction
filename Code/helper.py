import pickle
import math
import numpy as np
import torch
from sklearn.utils import shuffle

def get_batch(X_Set,S_Set,Y_Set,ID_Set,batch_size,random_seed,baseline_flag,epochcount):
    num_of_datapoints=len(X_Set)
    index_shuffeld=[ind for ind in range(num_of_datapoints)]
    index_shuffeld=shuffle(index_shuffeld,random_state=random_seed+epochcount)
    if baseline_flag:
        for i in range(num_of_datapoints // batch_size):
            batch=index_shuffeld[i*batch_size:(i+1)*batch_size]
            x_batch=torch.stack([torch.Tensor(X_Set[ind]) for ind in batch])
            y_batch=torch.stack([torch.Tensor(Y_Set[ind]) for ind in batch])
            yield (x_batch, y_batch)
        if num_of_datapoints<batch_size:
            i=-1
        if num_of_datapoints % batch_size != 0:
            batch=index_shuffeld[(i+1)*batch_size:]
            x_batch=torch.stack([torch.Tensor(X_Set[ind]) for ind in batch])
            y_batch=torch.stack([torch.Tensor(Y_Set[ind]) for ind in batch])
            yield (x_batch, y_batch)
    else:
        for i in range(num_of_datapoints // batch_size):
            batch=index_shuffeld[i*batch_size:(i+1)*batch_size]
            x_batch=torch.stack([torch.Tensor(X_Set[ind]) for ind in batch])
            s_batch=torch.stack([torch.Tensor(S_Set[ind][0]) for ind in batch])
            y_batch=torch.stack([torch.Tensor(Y_Set[ind]) for ind in batch])
            id_batch=[ID_Set[ind] for ind in batch]
            yield (x_batch, s_batch, y_batch, id_batch)
        if num_of_datapoints<batch_size:
            i=-1
        if num_of_datapoints % batch_size != 0:
            batch=index_shuffeld[(i+1)*batch_size:]
            x_batch=torch.stack([torch.Tensor(X_Set[ind]) for ind in batch])
            s_batch=torch.stack([torch.Tensor(S_Set[ind][0]) for ind in batch])
            y_batch=torch.stack([torch.Tensor(Y_Set[ind]) for ind in batch])
            id_batch=[ID_Set[ind] for ind in batch]
            yield (x_batch, s_batch, y_batch, id_batch)
        
def deg2rad(deg):
    value=deg * (math.pi/180)
    return value

def torchdeg2rad(deg,device):
    constant=float(math.pi/180)
    multiple=torch.Tensor([constant])
    multiple=multiple.repeat(deg.shape[0],1)
    multiple=multiple.to(device)
    value=deg * multiple
    return value

def InfluenceFactor(s_1,s_2,radius,device):
    
    if len(s_2.shape)==1:
        lat1=s_1[0]
        lat2=s_2[0]
        lon1=s_1[1]
        lon2=s_2[1]
        dlon=deg2rad(lon2-lon1)
        dlat=deg2rad(lat2-lat1)
        a=(math.sin(dlat/2))**2 + math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2))*(math.sin(dlon/2))**2
        c = 2 * math.atan2( math.sqrt(a), math.sqrt(1-a) ) 
        distance = 6373 * c # distance in kilometers
        distance= 1000 * distance # distance in meters

        if distance<radius:
            distance_squared=distance**2
            radius_squared=radius**2
            return math.exp(-distance_squared/(radius_squared*(radius_squared-distance_squared)))
        else:
            return 0
    else:
        lat1=s_1[0].repeat(s_2.shape[0],1)
        lat2=s_2[:,0].unsqueeze(1)
        lon1=s_1[1].repeat(s_2.shape[0],1)
        lon2=s_2[:,1].unsqueeze(1)
        dlon=torchdeg2rad(lon2-lon1,device)
        dlat=torchdeg2rad(lat2-lat1,device)
        a=(torch.sin(dlat/2))*(torch.sin(dlat/2)) + torch.cos(torchdeg2rad(lat1,device))*torch.cos(torchdeg2rad(lat2,device)) * (torch.sin(dlon/2))*(torch.sin(dlon/2))
        c = 2 * torch.atan2( torch.sqrt(a), torch.sqrt(1-a) ) 
        distance = 6373 * c # distance in kilometers
        distance = 1000 * distance # distance in meters
        
        if len(torch.nonzero(distance<radius,as_tuple=False).shape)>1:
            values_smaller_radius=torch.nonzero(distance<radius,as_tuple=False)[:,0]
            radius_squared=radius**2
            distance_smaller_radius=distance[values_smaller_radius,0]
            distance_smaller_radius_squared=distance_smaller_radius*distance_smaller_radius
            nonzeroreturn=torch.exp(-distance_smaller_radius_squared/(radius_squared*(radius_squared-distance_smaller_radius_squared)))
            returnvec=torch.zeros(s_2.shape[0],1).to(device)
            returnvec[values_smaller_radius,:]=nonzeroreturn.unsqueeze(1)
            returnvec=returnvec.transpose(0,1)
            return returnvec
        else:
            returnvec=torch.zeros(s_2.shape[0],1).to(device)
            returnvec=returnvec.transpose(0,1)
            return returnvec
        
def Compute_Influence_Factors_Train_Set(data,device,radius,save_str):
    SData=np.array(data['S_Train']).reshape(len(data['S_Train']),2)
    size_of_training_set=len(SData)

    influence_factor_tensor=torch.zeros([size_of_training_set,size_of_training_set])

    for i in range(size_of_training_set):
        for j in range(size_of_training_set):
            influence_factor_tensor[i,j]=InfluenceFactor(SData[i,:],SData[j,:],radius,device)
            
    with open('../Data/Influence_Factors/influence_factors_'+str(save_str)+'_'+str(int(radius)),'wb') as f:
        pickle.dump(influence_factor_tensor,f)
        
def trainingstep(model,data,baseline_flag,device,batch_size,optimizer,criterion,random_seed,epochcount): 
    model.train()
    train_loss = 0
    if baseline_flag:
        batch=get_batch(data['X_Train'],[],data['Y_Train'],
                        [],batch_size,random_seed,baseline_flag,epochcount)
        for (X,Y) in batch:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            output=model(X)
            loss=criterion(output,Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
    else:
        batch=get_batch(data['X_Train'],data['S_Train'],data['Y_Train'],
                    data['Business_IDs_Train'],batch_size,random_seed,baseline_flag,epochcount)
        for (X,S,Y,ID) in batch:
            X, S, Y = X.to(device), S.to(device), Y.to(device)
            optimizer.zero_grad()
            output=model(X,S,ID)
            loss=criterion(output,Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
    train_loss /= len(data['X_Train'])
    return train_loss

def validationstep(model,data,baseline_flag,device,batch_size,criterion,random_seed,epochcount):
    model.eval()
    validation_loss = 0
    if baseline_flag:
        batch=get_batch(data['X_Validation'],[],data['Y_Validation'],
                        [],batch_size,random_seed,baseline_flag,epochcount)
        with torch.no_grad():
            for (X,Y) in batch:
                X, Y = X.to(device), Y.to(device)
                output = model(X)
                validation_loss += criterion(output, Y).item() # sum up batch loss
    else:
        batch=get_batch(data['X_Validation'],data['S_Validation'],data['Y_Validation'],
                    data['Business_IDs_Validation'],batch_size,random_seed,baseline_flag,epochcount)
        with torch.no_grad():
            for (X,S,Y,ID) in batch:
                X, S, Y = X.to(device), S.to(device), Y.to(device)
                output = model(X,S,ID)
                validation_loss += criterion(output, Y).item() # sum up batch loss
    validation_loss /= len(data['X_Validation'])
    return validation_loss

def teststep(model,data,baseline_flag,device,batch_size,criterion,random_seed,epochcount):
    model.eval()
    test_loss = 0
    if baseline_flag:
        batch=get_batch(data['X_Test'],[],data['Y_Test'],
                        [],batch_size,random_seed,baseline_flag,epochcount)
        with torch.no_grad():
            for (X,Y) in batch:
                X, Y = X.to(device), Y.to(device)
                output = model(X)
                test_loss += criterion(output, Y).item() # sum up batch loss
    else:
        batch=get_batch(data['X_Test'],data['S_Test'],data['Y_Test'],
                    data['Business_IDs_Test'],batch_size,random_seed,baseline_flag,epochcount)
        with torch.no_grad():
            for (X,S,Y,ID) in batch:
                X, S, Y = X.to(device), S.to(device), Y.to(device)
                output = model(X,S,ID)
                test_loss += criterion(output, Y).item() # sum up batch loss
    test_loss /= len(data['X_Test'])
    return test_loss

def Training(NN,data,device,optimizer,criterion,batch_size,warm_up_flag,
             num_of_warmup_epochs,early_stopping_bound,max_num_of_epochs,
             save_str,random_seed,baseline_flag):
    
    list_of_train_loss=[]
    list_of_validation_loss=[]
    countflag=False
    continueflag=True
    counter=0
    epochcount=0
    
    while continueflag and epochcount<max_num_of_epochs:
        train_loss=trainingstep(NN,data,baseline_flag,device,batch_size,optimizer,criterion,random_seed,epochcount)
        validation_loss=validationstep(NN,data,baseline_flag,device,batch_size,criterion,random_seed,epochcount)
        list_of_train_loss.append(train_loss)
        list_of_validation_loss.append(validation_loss)
        
        print('Epoch '+str(epochcount)+' finished')
        print("Train Loss: "+str(train_loss))
        print("Validation Loss: "+str(validation_loss))
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        
        if warm_up_flag:
            if epochcount>num_of_warmup_epochs:
                warm_up_flag=False
            if list_of_validation_loss[-1]<=min(list_of_validation_loss):
                torch.save(NN.state_dict(),"..//Data//NN_models//current_model_"+save_str)
        else:
            if list_of_validation_loss[-1]>min(list_of_validation_loss):
                countflag=True
            else:
                countflag=False
                counter=0
                torch.save(NN.state_dict(),"..//Data//NN_models//current_model_"+save_str)
            
            if countflag or math.isnan(train_loss):
                counter+=1
            
            if counter==early_stopping_bound:
                continueflag=False
            
        epochcount+=1
        
    return [list_of_train_loss,list_of_validation_loss]

def calculate_MSE(model,data,target,NN_flag):
    if NN_flag:
        loss=np.sum((data-target)**2)/len(target)
    else:
        output=model.predict(data)
        loss=np.sum((output-target)**2)/len(target)
    return loss
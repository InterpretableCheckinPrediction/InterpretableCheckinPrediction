import torch
import torch.nn as nn
from helper import InfluenceFactor

device=torch.device("cuda")

class Model(nn.Module):
    
    def __init__(self,num_of_covariates,dropout_rate,X_Train,S_Train,Business_IDs_Train,Radius,influence_factor_tensor_of_Training_set,index_of_categorial_features):
        super(Model, self).__init__()
    
        self.num_of_covariates=num_of_covariates
        self.num_of_observations_in_training_set=len(X_Train)
        self.X_Train=torch.stack([torch.Tensor(x) for x in X_Train]).to(device)
        self.S_Train=torch.stack([torch.Tensor(s[0]) for s in S_Train]).to(device)
        self.Business_IDs_Train=Business_IDs_Train
        self.Radius=Radius
        self.influence_factor_tensor_of_Training_set=influence_factor_tensor_of_Training_set
        self.index_of_categorial_features=index_of_categorial_features
        self.dropout_rate=dropout_rate
        
        self.B_parameter=nn.Parameter(torch.rand(1))
        
        self.MUTerm=nn.Sequential(
            nn.Linear(self.num_of_covariates,512),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512,512),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256,1),
            nn.Sigmoid()
            )
    
        self.NTerm_Vector=nn.Sequential(
            nn.Linear(self.num_of_observations_in_training_set,512),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512,512),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512,512),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512,self.num_of_observations_in_training_set),
            nn.Tanh()
            )
    
        self.BTerm_Vector=nn.Sequential(
            nn.Linear(self.num_of_observations_in_training_set,512),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512,512),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512,512),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512,self.num_of_observations_in_training_set),
            nn.Sigmoid()
            )
    
    def forward_MUTerm(self,X):
        return self.MUTerm(X)
    
    def get_cosinesimilarity(self,X):
        # Compute the cosine similarity only on the business categories
        X_category=X[:,self.index_of_categorial_features:]
        X_Train_category=self.X_Train[:,self.index_of_categorial_features:]
        
        # Compute Cosine similarity for each x_c in the batch with all x_c in the trainingset
        X_category_norm=X_category / X_category.norm(dim=1)[:, None]
        X_Train_category_norm=X_Train_category / X_Train_category.norm(dim=1)[:, None]
        cosinesimilarity=torch.mm(X_category_norm, X_Train_category_norm.transpose(0,1))
        
        return cosinesimilarity
    
    def get_influencefactors(self,S):
        num_of_observations_in_batch=S.shape[0]
        if len(S.shape)==1:
            num_of_observations_in_batch=1
            
        # Compute influence factors for each business in batch
        influence_factor_tensor=torch.zeros(num_of_observations_in_batch,self.num_of_observations_in_training_set).to(device)
        
        if num_of_observations_in_batch<self.num_of_observations_in_training_set:
            for i in range(num_of_observations_in_batch):
                    influence_factor_tensor[i,:]=InfluenceFactor(S[i,:],self.S_Train[:,:],self.Radius,device)
        else:
            influence_factor_tensor=self.influence_factor_tensor_of_Training_set.to(device)
        
        return influence_factor_tensor
            
    def get_IDfactors(self,ID):
        num_of_observations_in_batch=len(ID)
        if num_of_observations_in_batch<self.num_of_observations_in_training_set:
            IDfactors=torch.zeros(num_of_observations_in_batch,self.num_of_observations_in_training_set).to(device)
            for i in range(num_of_observations_in_batch):
                current_business_ID=ID[i]
                helplist=[float(ID_Train!=current_business_ID) for ID_Train in self.Business_IDs_Train]
                helptensor=torch.Tensor(helplist).to(device)
                IDfactors[i,:]=helptensor
        else:
            #IDfactors=torch.eye(num_of_observations_in_batch,self.num_of_observations_in_training_set).to(device)
            IDfactors=torch.ones(self.num_of_observations_in_training_set,self.num_of_observations_in_training_set)-torch.eye(self.num_of_observations_in_training_set,self.num_of_observations_in_training_set)
            IDfactors=IDfactors.to(device)
        return IDfactors
        
    def forward_NTerm_Vector(self,cosinesimilarity,influencefactors,IDfactors):
        # Give cosinesimilarity as an input for the NTerm_Vector neural net
        # and perform elementwise multiplication with the influencefactors afterwards
        computed_NTerm_Vector=self.NTerm_Vector(cosinesimilarity)
        computed_NTerm_Vector=computed_NTerm_Vector*influencefactors*IDfactors
        
        # Additional weight of cosine similarity to get a higher interaction between similar firms and a lower interaction for different firms
        computed_NTerm_Vector=computed_NTerm_Vector*cosinesimilarity
            
        return computed_NTerm_Vector
        
    def forward_BTerm_Vector(self,cosinesimilarity,influencefactors,para):
        # Give cosinesimilarity as an input for the BTerm_Vector neural net
        # and perform elementwise multiplication with the influencefactors and the B_parameter afterwards
        computed_BTerm_Vector=self.BTerm_Vector(cosinesimilarity)
        computed_BTerm_Vector=computed_BTerm_Vector*influencefactors
        computed_BTerm_Vector=computed_BTerm_Vector*para
        
        # Additional weight of cosine similarity to get a higher interaction between similar firms and a lower interaction for different firms
        computed_BTerm_Vector=computed_BTerm_Vector*cosinesimilarity
        
        return computed_BTerm_Vector
    
    def forward_NTerm(self,cosinesimilarity,influencefactors,IDfactors):
        computed_NTerm_Vector=self.forward_NTerm_Vector(cosinesimilarity,influencefactors,IDfactors)
        NTerm=torch.sum(computed_NTerm_Vector,1,keepdim=True)
        return NTerm
        
    def forward_BTerm(self,cosinesimilarity,influencefactors,para):
        computed_BTerm_Vector=self.forward_BTerm_Vector(cosinesimilarity,influencefactors,para)
        MU_of_Training_Set=self.forward_MUTerm(self.X_Train)
        cosinesimilarity_of_Training_set=self.get_cosinesimilarity(self.X_Train)
        influencefactors_of_Training_set=self.get_influencefactors(self.S_Train)
        IDfactors_of_Training_set=self.get_IDfactors(self.Business_IDs_Train)
        N_of_Training_Set=self.forward_NTerm(cosinesimilarity_of_Training_set,influencefactors_of_Training_set,IDfactors_of_Training_set)
        sum_of_MU_and_N_of_Training_set=MU_of_Training_Set+N_of_Training_Set
        BTerm=torch.mm(computed_BTerm_Vector,sum_of_MU_and_N_of_Training_set)
        return BTerm
    
    def forward(self,X,S,ID):
        
        MUTerm=self.forward_MUTerm(X)
        para=torch.clamp(self.B_parameter,min=0.0)
        
        cosinesimilarity=self.get_cosinesimilarity(X)
        influencefactors=self.get_influencefactors(S)
        IDfactors=self.get_IDfactors(ID)
        
        NTerm=self.forward_NTerm(cosinesimilarity,influencefactors,IDfactors)
        BTerm=self.forward_BTerm(cosinesimilarity,influencefactors,para)
        
        return MUTerm+NTerm+BTerm
    
class BaselineModel(nn.Module):
    
    def __init__(self,num_of_covariates,dropout_rate):
        super(BaselineModel, self).__init__()
    
        self.num_of_covariates=num_of_covariates
        self.dropout_rate=dropout_rate
        
        self.FFNN=nn.Sequential(
            nn.Linear(self.num_of_covariates,512),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512,512),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256,1),
            nn.Sigmoid()
            )
    
    def forward(self,X):
        return self.FFNN(X)
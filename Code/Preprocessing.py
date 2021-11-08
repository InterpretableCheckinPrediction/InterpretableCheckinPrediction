import pandas as pd
import numpy as np
import pickle
import math
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer

## Cities to preprocessed
list_of_cities=['Las Vegas','Phoenix','Toronto']
list_of_business_IDs=[]
list_of_covariates=[]
list_of_locations=[]
list_of_targets=[]

## Load and filter data

latitude_lower_bound=[36,-math.inf,-math.inf]
longitude_lower_bound=[-115.35,-112.3,-math.inf]
latitude_upper_bound=[36.31,33.75,43.85]
longitude_upper_bound=[-115,-111.9,math.inf]

current_city_id=-1

for current_city in list_of_cities:
    
    current_city_id+=1
    print(current_city)
    
    business=pd.read_csv('..\Data\yelp-dataset\yelp_business.csv',sep=',')
    checkin=pd.read_csv('..\Data\yelp-dataset\yelp_checkin.csv',sep=',')
    review=pd.read_csv('..\Data\yelp-dataset\yelp_review.csv',sep=',')
    
    if current_city=='Las Vegas':
        searchfor=['Las Vegas','North Las Vegas']
        business_nationwide=business
        business=business[business.city.isin(searchfor)]
    else:
        business_nationwide=business
        business=business[business.city==current_city]
    searchfor=['Shopping','Grocery','Food']
    business=business[business.categories.str.contains('|'.join(searchfor))]
    business=business[business.is_open==1]
    business=business.sample(frac=1,random_state=12345).reset_index(drop=True) # Shuffle businesses
    
    checkin=checkin[checkin.business_id.isin(business.business_id)]
    review=review[review.business_id.isin(business.business_id)]
    
    ## Get business categories
    categories=[]
    for i in range(business.shape[0]):
        currentstring=business.categories.iloc[i]
        counter=0
        continuerflag=True
        while continuerflag:
            wordbegin=counter
            if ';' in currentstring[wordbegin:]:
                while currentstring[counter]!=';':
                    counter+=1
                categories.append(currentstring[wordbegin:counter])
                counter+=1
            else:
                categories.append(currentstring[wordbegin:])
                continuerflag=False

    unique_categories=[]

    for x in categories:
        if x not in unique_categories:
            if '(' not in x:
                if ')' not in x:
                    if '"' not in x:
                        unique_categories.append(x)
                    
                    
    categorycount=[]
    for c in unique_categories:
        categorycount.append(business[business.categories.str.contains(c)].shape[0])
    
    max_indices=np.argsort(categorycount)
    num_of_categories=80
    feature_categories=[]
    for i in range(1,num_of_categories+1):
        feature_categories.append(unique_categories[max_indices[-i]])
       
    
    ## Get chain affiliation
    chain=[]

    for i in range(business.shape[0]):
        current_id=business.business_id.iloc[i]
        current_business=business[business.business_id==current_id]
        current_name=current_business.name.iloc[0]
        num_of_such_businesses=business_nationwide[business_nationwide.name==current_name].shape[0]
        if num_of_such_businesses>14:
            chain.append(1)
        else:
            chain.append(0)
        print(i)
    
    print('Chain affiliations calculated.')
        
        
    ## Get business age
    ages=[]
    # Last review in december 2017
    end_year=2017
    end_month=12 
    for i in range(business.shape[0]):
        current_id=business.business_id.iloc[i]
        review_for_current_id=review[review.business_id==current_id]
        dates=review_for_current_id.date
        min_year=2100
        min_month=13
        for j in range(dates.shape[0]):
            current_year=int(dates.iloc[j][:4])
            if current_year<min_year:
                min_year=current_year
        dates=dates[dates.str.contains(str(min_year))]
        for j in range(dates.shape[0]):
            current_month=int(dates.iloc[j][5:7])
            if current_month<min_month:
                min_month=current_month
        if min_year==end_year:
            ages.append(end_month-min_month)
        elif min_year==end_year-1:
            ages.append(12-min_month+end_month)
        else:
            ages.append(12-min_month+(end_year-min_year-1)*12+end_month)
    
        print(i)
    
    print('Business ages calculated.')
    
    ## Create Datasets with locations S, covariates X and target variables Y
    num_of_variables_in_S=2
    # latitude, longitude
    num_of_variables_in_X=3+num_of_categories
    # average rating, age, chain, business categroy (80 categories)

    Business_IDs=[]
    S=[]
    X=[]
    Y=[]
    num_of_ignored_businesses=0
    num_of_ignored_businesses_2=0

    for i in range(business.shape[0]):
    
        out_of_scope_flag=False
    
        current_id=business.business_id.iloc[i]
        currents=np.zeros((1,num_of_variables_in_S))
        currentx=np.zeros((1,num_of_variables_in_X))

        currents[0,0]=business[business.business_id==current_id].latitude
        currents[0,1]=business[business.business_id==current_id].longitude
            
        if currents[0,0]<latitude_lower_bound[current_city_id]:
            out_of_scope_flag=True
        if currents[0,1]<longitude_lower_bound[current_city_id]:
            out_of_scope_flag=True
        if currents[0,0]>latitude_upper_bound[current_city_id]:
            out_of_scope_flag=True
        if currents[0,1]>longitude_upper_bound[current_city_id]:
            out_of_scope_flag=True
    
        currentx[0,0]=business[business.business_id==current_id].stars
        currentx[0,1]=ages[i]
        currentx[0,2]=chain[i]
    
        current_business=business[business.business_id==current_id]
    
        for j in range(len(feature_categories)):
            if current_business.categories.str.contains(feature_categories[j]).bool():
                currentx[0,3+j]=1

        checkintemp=checkin[checkin.business_id==current_id].checkins._get_numeric_data()
        num_of_checkins=checkintemp.sum()
        if num_of_checkins>=10:
            if out_of_scope_flag:
                num_of_ignored_businesses_2+=1
                print(str(i)+": "+str(num_of_ignored_businesses+num_of_ignored_businesses_2)+" "+'businesses ignored. Out of scope.')
            else:
                currenty=np.array([math.log(num_of_checkins)])
                S.append(currents)
                X.append(currentx)
                Y.append(currenty)
                Business_IDs.append(current_id)
        else:
            num_of_ignored_businesses+=1
            print(str(i)+": "+str(num_of_ignored_businesses+num_of_ignored_businesses_2)+" "+'businesses ignored.  No checkins.')
    print("  -->  "+str(num_of_ignored_businesses_2)+" "+'businesses ignored.  Out of scope.')
    print("  -->  "+str(num_of_ignored_businesses)+" "+'businesses ignored. No checkins.')
    
    print('Current city done.')
    print(' ')
        
    list_of_business_IDs.append(Business_IDs)
    list_of_covariates.append(X)
    list_of_locations.append(S)
    list_of_targets.append(Y)
    
 ## Train/Test/Validation Splitting
 
list_of_covariates_Train=[]
list_of_covariates_Validation=[]
list_of_covariates_Test=[]
list_of_targets_Train=[]
list_of_targets_Validation=[]
list_of_targets_Test=[]
list_of_locations_Train=[]
list_of_locations_Validation=[]
list_of_locations_Test=[]
list_of_business_IDs_Train=[]
list_of_business_IDs_Validation=[]
list_of_business_IDs_Test=[]

for ind in range(len(list_of_cities)):
    
    X=list_of_covariates[ind]
    Y=list_of_targets[ind]
    S=list_of_locations[ind]
    Business_IDs=list_of_business_IDs[ind]

    percentages=[0.7,0.15,0.15]

    num_of_datasets=len(X)
    num_of_datasets_Train=int(num_of_datasets*percentages[0])
    num_of_datasets_Validation=int(num_of_datasets*percentages[1])
    num_of_datasets_Test=int(num_of_datasets*percentages[2])
    while num_of_datasets_Train+num_of_datasets_Validation+num_of_datasets_Test!=num_of_datasets:
        num_of_datasets_Train+=1

    Business_IDs_Train=Business_IDs[:num_of_datasets_Train]
    Business_IDs_Validation=Business_IDs[num_of_datasets_Train:num_of_datasets_Train+num_of_datasets_Validation]
    Business_IDs_Test=Business_IDs[num_of_datasets_Train+num_of_datasets_Validation:num_of_datasets_Train+num_of_datasets_Validation+num_of_datasets_Test]
    X_Train=X[:num_of_datasets_Train]
    X_Validation=X[num_of_datasets_Train:num_of_datasets_Train+num_of_datasets_Validation]
    X_Test=X[num_of_datasets_Train+num_of_datasets_Validation:num_of_datasets_Train+num_of_datasets_Validation+num_of_datasets_Test]
    Y_Train=Y[:num_of_datasets_Train]
    Y_Validation=Y[num_of_datasets_Train:num_of_datasets_Train+num_of_datasets_Validation]
    Y_Test=Y[num_of_datasets_Train+num_of_datasets_Validation:num_of_datasets_Train+num_of_datasets_Validation+num_of_datasets_Test]
    S_Train=S[:num_of_datasets_Train]
    S_Validation=S[num_of_datasets_Train:num_of_datasets_Train+num_of_datasets_Validation]
    S_Test=S[num_of_datasets_Train+num_of_datasets_Validation:num_of_datasets_Train+num_of_datasets_Validation+num_of_datasets_Test]

    list_of_covariates_Train.append(X_Train)
    list_of_covariates_Validation.append(X_Validation)
    list_of_covariates_Test.append(X_Test)
    list_of_targets_Train.append(Y_Train)
    list_of_targets_Validation.append(Y_Validation)
    list_of_targets_Test.append(Y_Test)
    list_of_locations_Train.append(S_Train)
    list_of_locations_Validation.append(S_Validation)
    list_of_locations_Test.append(S_Test)
    list_of_business_IDs_Train.append(Business_IDs_Train)
    list_of_business_IDs_Validation.append(Business_IDs_Validation)
    list_of_business_IDs_Test.append(Business_IDs_Test)   
    
## Scaling and saving

for ind in range(len(list_of_cities)):
    
    X_Train=list_of_covariates_Train[ind]
    X_Validation=list_of_covariates_Validation[ind]
    X_Test=list_of_covariates_Test[ind]
    Y_Train=list_of_targets_Train[ind]
    Y_Validation=list_of_targets_Validation[ind]
    Y_Test=list_of_targets_Test[ind]
    S_Train=list_of_locations_Train[ind]
    S_Validation=list_of_locations_Validation[ind]
    S_Test=list_of_locations_Test[ind]
    Business_IDs_Train=list_of_business_IDs_Train[ind]
    Business_IDs_Validation=list_of_business_IDs_Validation[ind]
    Business_IDs_Test=list_of_business_IDs_Test[ind]
    
    # Scale instances
    scale_inst_X=RobustScaler()
    scale_inst_Y_1=MinMaxScaler()
    scale_inst_Y_2=PowerTransformer(method='yeo-johnson', standardize=False)
    scale_inst_Y_3=MinMaxScaler()

    # Scale only non categorial variables
    X_Train_noncategory=np.vstack(X_Train)[:,:2]
    X_Train_category=np.vstack(X_Train)[:,2:]
    X_Validation_noncategory=np.vstack(X_Validation)[:,:2]
    X_Validation_category=np.vstack(X_Validation)[:,2:]
    X_Test_noncategory=np.vstack(X_Test)[:,:2]
    X_Test_category=np.vstack(X_Test)[:,2:]

    scale_inst_X.fit(X_Train_noncategory)
    X_Train_noncategory=scale_inst_X.transform(X_Train_noncategory)
    X_Validation_noncategory=scale_inst_X.transform(X_Validation_noncategory)
    X_Test_noncategory=scale_inst_X.transform(X_Test_noncategory)

    # Merge them again
    X_Train=list(np.hstack([X_Train_noncategory,X_Train_category]))
    X_Validation=list(np.hstack([X_Validation_noncategory,X_Validation_category]))
    X_Test=list(np.hstack([X_Test_noncategory,X_Test_category]))

    # Scale target variable
    scale_inst_Y_1.fit(Y_Train)
    Y_Train=scale_inst_Y_1.transform(Y_Train)
    scale_inst_Y_2.fit(Y_Train)
    Y_Train=scale_inst_Y_2.transform(Y_Train)
    scale_inst_Y_3.fit(Y_Train)
    Y_Train=scale_inst_Y_3.transform(Y_Train)
    Y_Train=list(Y_Train)
    
    Y_Validation=scale_inst_Y_1.transform(Y_Validation)
    Y_Validation=scale_inst_Y_2.transform(Y_Validation)
    Y_Validation=scale_inst_Y_3.transform(Y_Validation)
    Y_Validation=list(Y_Validation)
    
    Y_Test=scale_inst_Y_1.transform(Y_Test)
    Y_Test=scale_inst_Y_2.transform(Y_Test)
    Y_Test=scale_inst_Y_3.transform(Y_Test)
    Y_Test=list(Y_Test)

    # Save
    save_str=str(list_of_cities[ind]).replace(" ","")
    
    data={'X_Train': X_Train,
          'X_Validation': X_Validation,
          'X_Test': X_Test,
          'Y_Train': Y_Train,
          'Y_Validation': Y_Validation,
          'Y_Test': Y_Test,
          'S_Train': S_Train,
          'S_Validation': S_Validation,
          'S_Test': S_Test,
          'scale_inst_X': scale_inst_X,
          'scale_inst_Y_1': scale_inst_Y_1,
          'scale_inst_Y_2': scale_inst_Y_2,
          'scale_inst_Y_3': scale_inst_Y_3,
          'Business_IDs_Train': Business_IDs_Train,
          'Business_IDs_Validation': Business_IDs_Validation,
          'Business_IDs_Test': Business_IDs_Test}
    
    with open('..\Data\Data_'+save_str,'wb') as f1:
        pickle.dump(data,f1)
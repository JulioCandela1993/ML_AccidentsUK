

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from scipy import stats
from xgboost import XGBClassifier
import itertools as it
import scikitplot as skplt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from lightgbm.sklearn import LGBMModel
import lightgbm as lgb
import numpy.core.defchararray as np_f

os.chdir("G:\Documentos\MasterDegree\BDMA\Classes\DataMining\FinalProject\Data")

trainFile = 'accidents_clusterized.csv'
trafficFile = 'ukTrafficAADF.csv'
accidents_train = pd.read_csv(trainFile)
traffic_data = pd.read_csv(trafficFile)

null_in_column = []
for column in accidents_train.columns:
    null_in_column.append((column, accidents_train[column].isnull().sum(),
                           str(accidents_train[column].isnull().sum()*100/len(accidents_train))+"%"))
pd.DataFrame(null_in_column, columns=['Column Name', 'Total Missing', 'Percentage Missing'])
## Set Missing Values to category according to bivariate analysis
accidents_train["Junction_Control"].fillna("Missing Info", inplace = True)

accidents_train['Light_Condition_2'] = np.where(accidents_train['Light_Conditions'] == 'Daylight: Street light present' , 'Daylight', 'Darkness')
accidents_train['Light_Condition_3'] = np.where(accidents_train['Light_Conditions'] == 'Darkeness: No street lighting' ,
               'Darkeness: Null or Low lighting', accidents_train['Light_Conditions'])
accidents_train['Light_Condition_3'] = np.where(accidents_train['Light_Condition_3'] == 'Darkness: Street lights present but unlit' ,
               'Darkeness: Null or Low lighting', accidents_train['Light_Condition_3'])

accidents_train['Pedestrian_Crossing_2'] = np.where(accidents_train['Pedestrian_Crossing-Physical_Facilities'] == 'Central refuge' ,
               'No Pedestrian Crossing', accidents_train['Pedestrian_Crossing-Physical_Facilities'])
accidents_train['Pedestrian_Crossing_2'] = np.where(accidents_train['Pedestrian_Crossing_2'] == 'Footbridge or subway' ,
               'No Pedestrian Crossing', accidents_train['Pedestrian_Crossing_2'])
accidents_train['Pedestrian_Crossing_2'] = np.where(accidents_train['Pedestrian_Crossing_2'] == 'No physical crossing within 50 meters' ,
               'No Pedestrian Crossing', accidents_train['Pedestrian_Crossing_2'])


#traffic_data = pd.DataFrame(traffic_data, columns = ['AADFYear', 'CP', 'Estimation_method', 'Estimation_method_detailed',
#       'Region', 'LocalAuthority', 'Road', 'RoadCategory', 'Location_Easting_OSGR',
#       'Location_Northing_OSGR', 'StartJunction', 'EndJunction', 'LinkLength_km',
#       'LinkLength_miles', 'PedalCycles', 'Motorcycles', 'CarsTaxis',
#       'BusesCoaches', 'LightGoodsVehicles', 'V2AxleRigidHGV',
#       'V3AxleRigidHGV', 'V4or5AxleRigidHGV', 'V3or4AxleArticHGV',
#       'V5AxleArticHGV', 'V6orMoreAxleArticHGV', 'AllHGVs', 'AllMotorVehicles',
#       'Lat', 'Lon'])
#           
#accidents_train = pd.merge(accidents_train, traffic_data, on = ['Location_Easting_OSGR','Location_Northing_OSGR'])


accidents_train['B_Latitude'] = np.array(pd.qcut(accidents_train['Latitude'], q=15, precision=1).astype(str))
accidents_train['B_Longitude'] = np.array(pd.qcut(accidents_train['Longitude'], q=10, precision=1).astype(str))

accidents_train['M_LAT_LON'] = accidents_train[['B_Latitude', 'B_Longitude']].apply(lambda x: '-'.join(x), axis=1)

accidents_train.drop(['B_Latitude', 'B_Longitude'], axis=1, inplace=True)

corr = accidents_train.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11,9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220,10,as_cmap=True)

# Draw the heatmap with he mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidth=.5, cbar_kws={"shrink":.5})

# Drop null columns
accidents_train = accidents_train.drop(['Junction_Detail'], axis=1)
# Drop string variables with a lot of different values
accidents_train.drop(['LSOA_of_Accident_Location', 'Local_Authority_(District)', 'Local_Authority_(Highway)', '1st_Road_Number', '2nd_Road_Number'], axis=1, inplace=True)
# Drop string with concentration in just one category
accidents_train.drop(['Special_Conditions_at_Site', 'Carriageway_Hazards','Pedestrian_Crossing-Human_Control'], axis=1, inplace=True)
# Drop null rows
accidents_train.dropna(axis=0, subset=['Road_Surface_Conditions', 'Time'], inplace=True)
# Drop post-accident rows
accidents_train = accidents_train.drop(['Number_of_Casualties', 'Number_of_Vehicles', 'Did_Police_Officer_Attend_Scene_of_Accident'], axis=1)
# Drop other (Index, Target Derived variable and Index of source file)
accidents_train = accidents_train.drop(['Accident_Index'], axis=1)
accidents_train = accidents_train.drop(['isFatal'], axis=1)
accidents_train = accidents_train.drop(['Unnamed: 0'], axis=1)
#accidents_train = accidents_train.drop(['Location_Northing_OSGR'], axis=1)
#accidents_train = accidents_train.drop(['Location_Easting_OSGR'], axis=1)
#accidents_train = accidents_train.drop(['Latitude'], axis=1)
#accidents_train = accidents_train.drop(['Longitude'], axis=1)

accidents_train['DateTime'] = pd.to_datetime(accidents_train['Date']+" "+accidents_train['Time'], format='%d/%m/%Y %H:%M')
accidents_train['year'] = accidents_train['DateTime'].dt.year
accidents_train['month'] = accidents_train['DateTime'].dt.month
accidents_train['day'] = accidents_train['DateTime'].dt.day
accidents_train['hour'] = accidents_train['DateTime'].dt.hour
accidents_train.drop(['Date', 'Time', 'Year', 'DateTime'], axis=1, inplace=True)
accidents_train['hour'] = accidents_train['hour'].astype(np.float32)

columnnames = [accidents_train.columns , accidents_train.dtypes]
new_categorial_analysis = ['Day_of_Week','cluster_accidents','fatal_accidents','month','hour','Road_Type',
                           'Light_Conditions','Weather_Conditions','Road_Surface_Conditions','Urban_or_Rural_Area',
                           'Light_Condition_2','Light_Condition_3','Pedestrian_Crossing_2','Junction_Control',
                           'Pedestrian_Crossing-Physical_Facilities','M_LAT_LON']

for col_cat in new_categorial_analysis:
    accidents_train[col_cat] = accidents_train[col_cat].astype(str)
    
accidents_train = pd.get_dummies(accidents_train, columns = new_categorial_analysis) 



fatal_accidents = accidents_train[accidents_train['Accident_Severity']== 1 ]
serious_accidents = accidents_train[accidents_train['Accident_Severity']== 2 ].sample(len(fatal_accidents), random_state= 42)
slight_accidents = accidents_train[accidents_train['Accident_Severity']== 3 ].sample(len(fatal_accidents), random_state= 42)
ds_undersample = pd.concat([slight_accidents,fatal_accidents,serious_accidents],axis= 0 ).sample(frac=1)

dataset_y = ds_undersample[ 'Accident_Severity' ]
dataset_x = ds_undersample.drop([ 'Accident_Severity' ], axis= 1 )
X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size= 0.20 , random_state= 42 )

##Feature Selection
## Method One: By importance values from Random Forest

model = RandomForestClassifier(random_state = 42)
model.fit(X_train, y_train)

importances = model.feature_importances_
ds_X = X_train
data = ds_X.columns
t = pd.DataFrame(data, columns = ['Features'])
t['Importances'] = importances
rank_attr = t.sort_values(by=['Importances'], ascending=False)
print(rank_attr)
top50 = rank_attr[0:50]['Features']

## Modelling



numAlgorithms = 5
numDifferentTests = 5

saveResults =  pd.DataFrame([[1.0000]*numAlgorithms]*numDifferentTests) ##[algorithm][test]
saveResults_Accuracy =  pd.DataFrame([[1.0000]*numAlgorithms]*numDifferentTests)
randomforest_CM = []
xgboost_CM = []
MLP_CM = []
lGboost_CM = []
Adaboost_CM = []

#####################################
### VALIDACIÓN DE PARÁMETROS - SEGUIMIENTO
##################################

def TestParameters(model, X_train, X_test , y_train , y_test, parameters):
    
    allNames = sorted(parameters)
    combinations = it.product(*(parameters[Name] for Name in allNames))
    diff_params = list(combinations)
    
    max_roc = 0
    max_model_params = {}
    
    for param in diff_params:
        
        dictionary = dict(zip(allNames, list(param)))
        print(dictionary)
        model.set_params(**dictionary)
        model.fit(X_train, y_train)
        
        y_probas=model.predict_proba(X_test)
        roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
        print(roc)
        if roc>max_roc:
            max_roc = roc
            max_model_params = {key:value for key, value in dictionary.items()}
            
    return model.set_params(**max_model_params)


def TestParameters_Eval(model, X_train, X_test , y_train , y_test, parameters):
    
    allNames = sorted(parameters)
    combinations = it.product(*(parameters[Name] for Name in allNames))
    diff_params = list(combinations)
    
    max_roc = 0
    max_model_params = {}
    
    for param in diff_params:
        
        dictionary = dict(zip(allNames, list(param)))
        print(dictionary)
        model.set_params(**dictionary)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
        eval_metric = 'mlogloss',early_stopping_rounds=50, verbose=True)
        
        y_probas=model.predict_proba(X_test)
        roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
        print(roc)
        if roc>max_roc:
            max_roc = roc
            max_model_params = {key:value for key, value in dictionary.items()}
            
    return model.set_params(**max_model_params)

def accuracy_score(y_true, y_pred):
    y1 = np.where(np.array(y_true) == 1 , 1 , 0)
    ypred1 = np.where(np.array(y_pred) == 1 , 1 , 0)
    y_equals = np.where(y1 == ypred1 , 1 , 0)
    return sum(y_equals)/len(y_equals)



#######################################################
#### RANDOM FOREST
#######################################################


##Test 1: Normal

dataset_y = accidents_train[ 'Accident_Severity' ]
dataset_x = accidents_train.drop([ 'Accident_Severity' ], axis= 1 )
X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size= 0.20 , random_state= 42 )

print('Random Forest - Normal')

model = RandomForestClassifier(random_state = 42)
model.fit(X_train, y_train)

y_probas=model.predict_proba(X_test)
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()
saveResults[0][0] = roc
saveResults_Accuracy[0][0] = accuracy_score(y_test , model.predict(X_test))
randomforest_CM.append(["Unbalanced" , confusion_matrix(y_test, model.predict(X_test))])

##Test 2: UnderSampling

fatal_accidents = accidents_train[accidents_train['Accident_Severity']== 1 ]
serious_accidents = accidents_train[accidents_train['Accident_Severity']== 2 ].sample(len(fatal_accidents), random_state= 42)
slight_accidents = accidents_train[accidents_train['Accident_Severity']== 3 ].sample(len(fatal_accidents), random_state= 42)
ds_undersample = pd.concat([slight_accidents,fatal_accidents,serious_accidents],axis= 0 ).sample(frac=1, random_state= 42)

dataset_y = ds_undersample[ 'Accident_Severity' ]
dataset_x = ds_undersample.drop([ 'Accident_Severity' ], axis= 1 )
X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size= 0.20 , random_state= 42 )

model = RandomForestClassifier(random_state = 42)
model.fit(X_train, y_train)

y_probas=model.predict_proba(X_test)
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()
saveResults[0][1] = roc
saveResults_Accuracy[0][1] = accuracy_score(y_test , model.predict(X_test))
randomforest_CM.append(["UnderSampling" , confusion_matrix(y_test, model.predict(X_test))])

##Test 3: Normalization

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_normalized , X_test_normalized = scaler.transform(X_train), scaler.transform(X_test)

model.fit(X_train_normalized, y_train)

y_probas=model.predict_proba(X_test_normalized)
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()
saveResults[0][2] = roc
saveResults_Accuracy[0][2] = accuracy_score(y_test , model.predict(X_test))
randomforest_CM.append(["UnderSampling" , confusion_matrix(y_test, model.predict(X_test_normalized))])

##Test 4: Hypertunning of Parameters

param_grid = {
  'n_estimators' : [500,700,1000,1500],
  'max_depth' : [24,30,35,40],
  'min_samples_split': [2,5] # 1000 ~1% of dataset
}


model = RandomForestClassifier(random_state = 42)
#grid_search = GridSearchCV(estimator = rfc, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2, scoring='roc_auc')
grid_search = GridSearchCV(estimator = model, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

#Validación mejores parámetros
#best_grid = TestParameters(model, X_train, X_test , y_train , y_test, param_grid)

grid_search.fit(X_train, y_train)
best_grid = grid_search.best_estimator_
grid_search.best_estimator_
print(best_grid)

model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=35, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=1500,
                       n_jobs=None, oob_score=False, random_state=42, verbose=0,
                       warm_start=False)

model.fit(X_train, y_train)
y_probas=model.predict_proba(X_test)
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()

saveResults[0][3] = roc
saveResults_Accuracy[0][3] = accuracy_score(y_test , model.predict(X_test))
randomforest_CM.append(["HyperTunning" , confusion_matrix(y_test, model.predict(X_test))])

##Test 5: Feature Importance / Curse Dimensionality

def getBestNumberOfFeatures(X_train, y_train, X_test, y_test, model, rank_attr):
    start = [30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,250,267]
    max_roc = 0
    max_start = 0
    save_Best = pd.DataFrame([[1.000000]*2]*(len(start)),columns = ['dim','roc'])
    for i in range(len(start)):
        model.fit(X_train[rank_attr[0:start[i]]['Features']], y_train)
        y_probas=model.predict_proba(X_test[rank_attr[0:start[i]]['Features']])
        roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
        save_Best['dim'][i] = start[i]
        save_Best['roc'][i] = roc
        if roc>max_roc:
            max_roc = roc
            max_start = start[i]
    plt.plot(save_Best['dim'].astype(int), save_Best['roc'])
    return  max_start   



model = model

best_dim = getBestNumberOfFeatures(X_train, y_train, X_test, y_test, model, rank_attr)
new_dims = rank_attr[0:best_dim]['Features']
model.fit(X_train[new_dims], y_train)

y_probas=model.predict_proba(X_test[new_dims])
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()

saveResults[0][4] = roc
saveResults_Accuracy[0][4] = accuracy_score(y_test , model.predict(X_test[new_dims]))
randomforest_CM.append(["Feature Selection" , confusion_matrix(y_test, model.predict(X_test[new_dims]))])

#######################################################
#### XG BOOST
#######################################################

##Test 1: Normal



dataset_y = accidents_train[ 'Accident_Severity' ]
dataset_x = accidents_train.drop([ 'Accident_Severity' ], axis= 1 )
X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size= 0.20 , random_state= 42 )
col = np.array(X_train.columns , dtype = str)
col = np_f.replace(col, '[', '(')
col = np_f.replace(col, ']', ')')
col = np_f.replace(col, '<', '-')

X_train.columns = col
X_test.columns = col

#1) XGBoost
print('XGBoost - Normal')

model = XGBClassifier(eval_metric='auc')
model.fit(X_train, y_train)

y_probas=model.predict_proba(X_test)
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()
saveResults[1][0] = roc
saveResults_Accuracy[1][0] = accuracy_score(y_test , model.predict(X_test))
xgboost_CM.append(["Unbalanced" , confusion_matrix(y_test, model.predict(X_test))])

##Test 2: UnderSampling

fatal_accidents = accidents_train[accidents_train['Accident_Severity']== 1 ]
serious_accidents = accidents_train[accidents_train['Accident_Severity']== 2 ].sample(len(fatal_accidents), random_state= 42)
slight_accidents = accidents_train[accidents_train['Accident_Severity']== 3 ].sample(len(fatal_accidents), random_state= 42)
ds_undersample = pd.concat([slight_accidents,fatal_accidents,serious_accidents],axis= 0 ).sample(frac=1, random_state= 42)

dataset_y = ds_undersample[ 'Accident_Severity' ]
dataset_x = ds_undersample.drop([ 'Accident_Severity' ], axis= 1 )
X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size= 0.20 , random_state= 42 )
col = np.array(X_train.columns , dtype = str)
col = np_f.replace(col, '[', '(')
col = np_f.replace(col, ']', ')')
col = np_f.replace(col, '<', '-')

X_train.columns = col
X_test.columns = col


model = XGBClassifier(random_state = 42)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
        eval_metric = 'mlogloss',early_stopping_rounds=50, verbose=True)

y_probas=model.predict_proba(X_test)
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()
saveResults[1][1] = roc
saveResults_Accuracy[1][1] = accuracy_score(y_test , model.predict(X_test))
xgboost_CM.append(["UnderSampling" , confusion_matrix(y_test, model.predict(X_test))])

##Test 3: Normalization
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_normalized , X_test_normalized = scaler.transform(X_train), scaler.transform(X_test)

model.fit(X_train_normalized, y_train)

y_probas=model.predict_proba(X_test_normalized)
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()
saveResults[1][2] = roc
saveResults_Accuracy[1][2] = accuracy_score(y_test , model.predict(X_test_normalized))
xgboost_CM.append(["Normalization" , confusion_matrix(y_test, model.predict(X_test_normalized))])

##Test 4: Hypertunning of Parameters

param_grid = {
    'gamma': [0,0.05,1],
    #'objective': ['binary:logistic'],
    'reg_alpha': [0,0.1,0.5],
    'reg_lambda': [0.6,0.8,1],
    'colsample_bytree': [0.9, 0.95 , 1],
    'subsample': [0.9, 1],
    'n_estimators': [100,300,500],
    'max_depth': [10,15,20],
    'learning_rate': [0.1,0.01],
    #'min_child_weight': [5]
}

model = XGBClassifier(random_state = 42)
#grid_search = GridSearchCV(estimator = rfc, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2, scoring='roc_auc')

best_grid = TestParameters_Eval(model, X_train, X_test , y_train , y_test, param_grid)

grid_search = GridSearchCV(estimator = model, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(X_train, y_train)


best_grid = grid_search.best_estimator_
grid_search.best_estimator_
print(best_grid)

model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.95, gamma=0.05,
              learning_rate=0.1, max_delta_step=0, max_depth=10,
              min_child_weight=1, missing=None, n_estimators=300, n_jobs=1,
              nthread=None, objective='multi:softprob', random_state=42,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=0.9, verbosity=1)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
        eval_metric = 'mlogloss',early_stopping_rounds=25, verbose=True)
y_probas=model.predict_proba(X_test)
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()

saveResults[1][3] = roc
saveResults_Accuracy[1][3] = accuracy_score(y_test , model.predict(X_test))
xgboost_CM.append(["HyperTunning" , confusion_matrix(y_test, model.predict(X_test))])

##Test 5: Feature Importance / Curse Dimensionality
rank_attr_2 = pd.DataFrame()
rank_attr_2 = rank_attr.copy()
col = np.array(rank_attr_2['Features'] , dtype = str)
col = np_f.replace(col, '[', '(')
col = np_f.replace(col, ']', ')')
col = np_f.replace(col, '<', '-')
rank_attr_2['Features'] = np.array(col)

def getBestNumberOfFeatures(X_train, y_train, X_test, y_test, model, rank_attr):
    start = [30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,250,271]
    max_roc = 0
    max_start = 0
    save_Best = pd.DataFrame([[1.000000]*2]*(len(start)),columns = ['dim','roc'])
    for i in range(len(start)):
        model.fit(X_train[rank_attr[0:start[i]]['Features']], y_train)
        y_probas=model.predict_proba(X_test[rank_attr[0:start[i]]['Features']])
        roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
        save_Best['dim'][i] = start[i]
        save_Best['roc'][i] = roc
        if roc>max_roc:
            max_roc = roc
            max_start = start[i]
    plt.plot(save_Best['dim'].astype(int), save_Best['roc'])
    return  max_start   

model = model

best_dim = getBestNumberOfFeatures(X_train, y_train, X_test, y_test, model, rank_attr_2)
new_dims = rank_attr[0:best_dim]['Features']
model.fit(X_train[new_dims], y_train)

y_probas=model.predict_proba(X_test[new_dims])
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()

saveResults[1][4] = roc
saveResults_Accuracy[1][4] = accuracy_score(y_test , model.predict(X_test[new_dims]))
xgboost_CM.append(["Feature Selection" , confusion_matrix(y_test, model.predict(X_test[new_dims]))])



#######################################################
#### LGBM
#######################################################


##Test 1: Normal

dataset_y = accidents_train[ 'Accident_Severity' ]
dataset_x = accidents_train.drop([ 'Accident_Severity' ], axis= 1 )
X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size= 0.20 , random_state= 42 )
col = np.array(X_train.columns , dtype = str)
col = np_f.replace(col, ':', '-')
col = np_f.replace(col, '[', '-')
col = np_f.replace(col, ']', '-')
X_train.columns = col
X_test.columns = col

#1) LGBM
print('LGBM - Normal')

model = lgb.LGBMClassifier(random_state = 42)
model.fit(X_train, y_train, eval_metric='multi_logloss',eval_set=[(X_test, y_test)],early_stopping_rounds=50)

y_probas=model.predict_proba(X_test)
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()
saveResults[2][0] = roc
saveResults_Accuracy[2][0] = accuracy_score(y_test , model.predict(X_test))
lGboost_CM.append(["Unbalanced" , confusion_matrix(y_test, model.predict(X_test))])

##Test 2: UnderSampling

fatal_accidents = accidents_train[accidents_train['Accident_Severity']== 1 ]
serious_accidents = accidents_train[accidents_train['Accident_Severity']== 2 ].sample(len(fatal_accidents), random_state= 42)
slight_accidents = accidents_train[accidents_train['Accident_Severity']== 3 ].sample(len(fatal_accidents), random_state= 42)
ds_undersample = pd.concat([slight_accidents,fatal_accidents,serious_accidents],axis= 0 ).sample(frac=1, random_state= 42)

dataset_y = ds_undersample[ 'Accident_Severity' ]
dataset_x = ds_undersample.drop([ 'Accident_Severity' ], axis= 1 )
X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size= 0.20 , random_state= 42 )
col = np.array(X_train.columns , dtype = str)
col = np_f.replace(col, '[', '-')
col = np_f.replace(col, ']', '-')

X_train.columns = col
X_test.columns = col


model = lgb.LGBMClassifier(random_state = 42)
model.fit(X_train, y_train, eval_metric='multi_logloss',eval_set=[(X_test, y_test)],early_stopping_rounds=50)

y_probas=model.predict_proba(X_test)
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()
saveResults[2][1] = roc
saveResults_Accuracy[2][1] = accuracy_score(y_test , model.predict(X_test))
lGboost_CM.append(["UnderSampling" , confusion_matrix(y_test, model.predict(X_test))])

##Test 3: Normalization

dataset_y = accidents_train[ 'Accident_Severity' ]
dataset_x = accidents_train.drop([ 'Accident_Severity' ], axis= 1 )
X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size= 0.20 , random_state= 42 )
col = np.array(X_train.columns , dtype = str)
col = np_f.replace(col, ':', '=')
col = np_f.replace(col, '(', '')
col = np_f.replace(col, ')', '')
col = np_f.replace(col, '[', '')
col = np_f.replace(col, ']', '')
col = np_f.replace(col, '--', '')
col = np_f.replace(col, ',', '')
X_train.columns = col
X_test.columns = col

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_normalized , X_test_normalized = scaler.transform(X_train), scaler.transform(X_test)

model.fit(X_train_normalized, y_train, eval_metric='multi_logloss',eval_set=[(X_test_normalized, y_test)],early_stopping_rounds=50)

y_probas=model.predict_proba(X_test_normalized)
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()
saveResults[2][2] = roc
saveResults_Accuracy[2][2] = accuracy_score(y_test , model.predict(X_test))
lGboost_CM.append(["Normalization" , confusion_matrix(y_test, model.predict(X_test_normalized))])

##Test 4: Hypertunning of Parameters

param_grid = {
    'num_leaves': [500,1000],
    'max_depth': [20,30,40],
    'alpha': [0.01, 0.001,0.0001],
    'learning_rate': [0.5, 0.3, 0.1],
    'n_estimators': [100000,2000000],
    'reg_alpha': [0.5,0.6,0.7],
    'reg_lambda': [0.5,0.6,0.7]
}

model = lgb.LGBMClassifier(random_state = 42, 
                           boosting_type='gbdt',
                           n_jobs=8, 
                         silent=False,subsample=0.8, 
                         subsample_freq=10, 
                         colsample_bytree=0.6, 
                         objective=None, )
#grid_search = GridSearchCV(estimator = rfc, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2, scoring='roc_auc')
grid_search = GridSearchCV(estimator = model, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(X_train, y_train)
best_grid = grid_search.best_estimator_
grid_search.best_estimator_
print(best_grid)

model = lgb.LGBMClassifier(boosting_type='gbdt',
                         num_leaves=1000, 
                         max_depth=40, 
                         learning_rate=0.01, 
                         n_estimators=2000000, 
                         #subsample_for_bin=200000, 
                         objective=None, 
                         #min_split_gain=1, 
                         #min_child_weight=0.1, 
                         #min_child_samples=20, 
                         subsample=0.8, 
                         subsample_freq=10, 
                         colsample_bytree=0.6, 
                         reg_alpha=0.6, 
                         reg_lambda=0.6, 
                         n_jobs=8, 
                         silent=False,
                        )

#model = best_grid

model.fit(X_train, y_train, eval_metric='multi_logloss',eval_set=[(X_test, y_test)],early_stopping_rounds=100)
y_probas=model.predict_proba(X_test)
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()

saveResults[2][3] = roc
saveResults_Accuracy[2][3] = accuracy_score(y_test , model.predict(X_test))
lGboost_CM.append(["HyperTunning" , confusion_matrix(y_test, model.predict(X_test))])

##Test 5: Feature Importance / Curse Dimensionality
rank_attr_2 = pd.DataFrame()
rank_attr_2 = rank_attr.copy()
col = np.array(rank_attr_2['Features'] , dtype = str)
col = np_f.replace(col, ':', '=')
col = np_f.replace(col, '(', '')
col = np_f.replace(col, ')', '')
col = np_f.replace(col, '[', '')
col = np_f.replace(col, ']', '')
col = np_f.replace(col, '--', '')
col = np_f.replace(col, ',', '')
rank_attr_2['Features'] = np.array(col)

def getBestNumberOfFeatures(X_train, y_train, X_test, y_test, model, rank_attr):
    start = [30,50,100,150,200,250,271]
    max_roc = 0
    max_start = 0
    save_Best = pd.DataFrame([[1.000000]*2]*(len(start)),columns = ['dim','roc'])
    for i in range(len(start)):
        model.fit(X_train[rank_attr[0:start[i]]['Features']], y_train, eval_metric='multi_logloss',
                  eval_set=[(X_test[rank_attr[0:start[i]]['Features']], y_test)],early_stopping_rounds=50)
        y_probas=model.predict_proba(X_test[rank_attr[0:start[i]]['Features']])
        roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
        save_Best['dim'][i] = start[i]
        save_Best['roc'][i] = roc
        if roc>max_roc:
            max_roc = roc
            max_start = start[i]
    plt.plot(save_Best['dim'].astype(int), save_Best['roc'])
    return  max_start   

model = model

best_dim = getBestNumberOfFeatures(X_train, y_train, X_test, y_test, model, rank_attr_2)
new_dims = rank_attr_2[0:best_dim]['Features']
model.fit(X_train[new_dims], y_train,
           eval_set=[(X_test[new_dims], y_test)],early_stopping_rounds=50)

y_probas=model.predict_proba(X_test[new_dims])
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()

saveResults[2][4] = roc
saveResults_Accuracy[2][4] = accuracy_score(y_test , model.predict(X_test[new_dims]))
lGboost_CM.append(["Feature Selection" , confusion_matrix(y_test, model.predict(X_test[new_dims]))])


new_dims = rank_attr_2[0:100]['Features']
model.fit(X_train[new_dims], y_train,
           eval_set=[(X_test[new_dims], y_test)],early_stopping_rounds=50)

y_probas=model.predict_proba(X_test[new_dims])
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()

total_true = np.sum(yt)
yt=np.where(y_test==1 , 1 ,0)
yp=y_probas[:,0]
prob = pd.DataFrame([[yt[i],yp[i]] for i in range(len(yt))], columns = ['true','prob'])
prob['decil'] = pd.qcut(prob['prob'] , 10 , retbins = False).astype(str)
fig, axis1 = plt.subplots(figsize=(10,5))
bivar = prob.groupby('decil').agg({'decil':'size','true':'sum'})
bivar['por_fatal'] = bivar['true']/total_true*100
bivar.to_csv('results_veintil.csv')
axis_x = bivar.index
plt.title('decil')
plt.bar(axis_x , bivar['por_fatal'], label = '# of Accidents')     
plt.legend()
plt.xticks(rotation=90)       
#axis2.plot(axis_x , bivar['por_fatal'], color = 'red', linestyle='dashed', marker='o', label = 'Percentage Fatal Accidents')
plt.legend()
plt.show()

#######################################################
#### MLP
#######################################################

##Test 1: Normal


dataset_y = accidents_train[ 'Accident_Severity' ]
dataset_x = accidents_train.drop([ 'Accident_Severity' ], axis= 1 )
X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size= 0.20 , random_state= 42 )

#1) MLP
print('Multi Later Perceptron - Normal')

model = MLPClassifier()
model.fit(X_train, y_train)

y_probas=model.predict_proba(X_test)
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()
saveResults[3][0] = roc
saveResults_Accuracy[3][0] = accuracy_score(y_test , model.predict(X_test))
MLP_CM.append(["Unbalanced" , confusion_matrix(y_test, model.predict(X_test))])

##Test 2: UnderSampling

fatal_accidents = accidents_train[accidents_train['Accident_Severity']== 1 ]
serious_accidents = accidents_train[accidents_train['Accident_Severity']== 2 ].sample(len(fatal_accidents), random_state= 42)
slight_accidents = accidents_train[accidents_train['Accident_Severity']== 3 ].sample(len(fatal_accidents), random_state= 42)
ds_undersample = pd.concat([slight_accidents,fatal_accidents,serious_accidents],axis= 0 ).sample(frac=1, random_state= 42)

dataset_y = ds_undersample[ 'Accident_Severity' ]
dataset_x = ds_undersample.drop([ 'Accident_Severity' ], axis= 1 )
X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size= 0.20 , random_state= 42 )

model = MLPClassifier(random_state = 42)
model.fit(X_train, y_train)

y_probas=model.predict_proba(X_test)
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()
saveResults[3][1] = roc
saveResults_Accuracy[3][1] = accuracy_score(y_test , model.predict(X_test))
MLP_CM.append(["UnderSampling" , confusion_matrix(y_test, model.predict(X_test))])

##Test 3: Normalization
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_normalized , X_test_normalized = scaler.transform(X_train), scaler.transform(X_test)

model.fit(X_train_normalized, y_train)

y_probas=model.predict_proba(X_test_normalized)
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()
saveResults[3][2] = roc
saveResults_Accuracy[3][2] = accuracy_score(y_test , model.predict(X_test_normalized))
MLP_CM.append(["Normalization" , confusion_matrix(y_test, model.predict(X_test_normalized))])

##Test 4: Hypertunning of Parameters

param_grid = {
    'hidden_layer_sizes': [30,50,(30,30),(30,10)],
    'activation': ['logistic','tanh'],
    'alpha': [0.05,0.01, 0.005],
    'learning_rate': ['adaptive'],
    'momentum': [0.7,0.8, 0.9]
}

model = MLPClassifier(random_state = 42)
#grid_search = GridSearchCV(estimator = rfc, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2, scoring='roc_auc')

#best_grid = TestParameters(model, X_train_normalized, X_test_normalized , y_train , y_test, param_grid)

grid_search = GridSearchCV(estimator = model, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(X_train_normalized, y_train)
best_grid = grid_search.best_estimator_
grid_search.best_estimator_
print(best_grid)

model = MLPClassifier(activation='logistic', alpha=0.05, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(30, 30), learning_rate='adaptive',
              learning_rate_init=0.001, max_fun=15000, max_iter=200,
              momentum=0.7, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=42, shuffle=True, solver='adam',
              tol=0.0001, validation_fraction=0.1, verbose=False,
              warm_start=False)

model.fit(X_train_normalized, y_train)
y_probas=model.predict_proba(X_test_normalized)
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()

saveResults[3][3] = roc
saveResults_Accuracy[3][3] = accuracy_score(y_test , model.predict(X_test_normalized))
MLP_CM.append(["HyperTunning" , confusion_matrix(y_test, model.predict(X_test_normalized))])

##Test 5: Feature Importance / Curse Dimensionality

def getBestNumberOfFeatures(X_train, y_train, X_test, y_test, model, rank_attr):
    start = [30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,250,271]
    max_roc = 0
    max_start = 0
    save_Best = pd.DataFrame([[1.000000]*2]*(len(start)),columns = ['dim','roc'])
    for i in range(len(start)):
        model.fit(X_train[rank_attr[0:start[i]]['Features']], y_train)
        y_probas=model.predict_proba(X_test[rank_attr[0:start[i]]['Features']])
        roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
        save_Best['dim'][i] = start[i]
        save_Best['roc'][i] = roc
        if roc>max_roc:
            max_roc = roc
            max_start = start[i]
    plt.plot(save_Best['dim'].astype(int), save_Best['roc'])
    return  max_start   

model = model

X_train_normalized = pd.DataFrame(X_train_normalized, columns = [X_train.columns])
X_test_normalized = pd.DataFrame(X_test_normalized, columns = [X_train.columns])
best_dim = getBestNumberOfFeatures(X_train_normalized, y_train,
                                   X_test_normalized, 
                                   y_test, model, rank_attr)
new_dims = rank_attr[0:best_dim]['Features']
model.fit(X_train_normalized[new_dims], y_train)

y_probas=model.predict_proba(X_test_normalized[new_dims])
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()

saveResults[3][4] = roc
saveResults_Accuracy[3][4] = accuracy_score(y_test , model.predict(X_test_normalized[new_dims]))
MLP_CM.append(["Feature Selection" , confusion_matrix(y_test, model.predict(X_test_normalized[new_dims]))])



#######################################################
#### ADA BOOST
#######################################################

##Test 1: Normal

dataset_y = accidents_train[ 'Accident_Severity' ]
dataset_x = accidents_train.drop([ 'Accident_Severity' ], axis= 1 )
X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size= 0.20 , random_state= 42 )

#1) XGBoost
print('XGBoost - Normal')

model = AdaBoostClassifier(random_state = 42)
model.fit(X_train, y_train)

y_probas=model.predict_proba(X_test)
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()
saveResults[4][0] = roc
saveResults_Accuracy[4][0] = accuracy_score(y_test , model.predict(X_test))
Adaboost_CM.append(["Unbalanced" , confusion_matrix(y_test, model.predict(X_test))])

##Test 2: UnderSampling

fatal_accidents = accidents_train[accidents_train['Accident_Severity']== 1 ]
serious_accidents = accidents_train[accidents_train['Accident_Severity']== 2 ].sample(len(fatal_accidents), random_state= 42)
slight_accidents = accidents_train[accidents_train['Accident_Severity']== 3 ].sample(len(fatal_accidents), random_state= 42)
ds_undersample = pd.concat([slight_accidents,fatal_accidents,serious_accidents],axis= 0 ).sample(frac=1, random_state= 42)

dataset_y = ds_undersample[ 'Accident_Severity' ]
dataset_x = ds_undersample.drop([ 'Accident_Severity' ], axis= 1 )
X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size= 0.20 , random_state= 42 )


model = AdaBoostClassifier(random_state = 42)
model.fit(X_train, y_train)

y_probas=model.predict_proba(X_test)
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()
saveResults[4][1] = roc
saveResults_Accuracy[4][1] = accuracy_score(y_test , model.predict(X_test))
Adaboost_CM.append(["UnderSampling" , confusion_matrix(y_test, model.predict(X_test))])

##Test 3: Normalization

dataset_y = accidents_train[ 'Accident_Severity' ]
dataset_x = accidents_train.drop([ 'Accident_Severity' ], axis= 1 )
X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size= 0.20 , random_state= 42 )


scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_normalized , X_test_normalized = scaler.transform(X_train), scaler.transform(X_test)

model.fit(X_train_normalized, y_train)

y_probas=model.predict_proba(X_test_normalized)
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()
saveResults[4][2] = roc
saveResults_Accuracy[4][2] = accuracy_score(y_test , model.predict(X_test))
Adaboost_CM.append(["Normalization" , confusion_matrix(y_test, model.predict(X_test_normalized))])

##Test 4: Hypertunning of Parameters


                  
param_grid = {
    'base_estimator': [None] ,
    'algorithm': ['SAMME.R'],
    'n_estimators': [100,500,600,1000],
    'learning_rate': [1,0.5,0.1]
}

   
model = AdaBoostClassifier(random_state = 42)

best_grid = TestParameters(model, X_train, X_test , y_train , y_test, param_grid)

#grid_search = GridSearchCV(estimator = rfc, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2, scoring='roc_auc')
grid_search = GridSearchCV(estimator = model, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(X_train_normalized, y_train)
best_grid = grid_search.best_estimator_
grid_search.best_estimator_
print(best_grid)

model = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=0.5,
                   n_estimators=600, random_state=42)

model.fit(X_train, y_train)
y_probas=model.predict_proba(X_test)
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()

saveResults[4][3] = roc
saveResults_Accuracy[4][3] = accuracy_score(y_test , model.predict(X_test))
Adaboost_CM.append(["HyperTunning" , confusion_matrix(y_test, model.predict(X_test))])

##Test 5: Feature Importance / Curse Dimensionality

def getBestNumberOfFeatures(X_train, y_train, X_test, y_test, model, rank_attr):
    start = [60, 100,150,200,271]
    max_roc = 0
    max_start = 0
    save_Best = pd.DataFrame([[1.000000]*2]*(len(start)),columns = ['dim','roc'])
    for i in range(len(start)):
        model.fit(X_train[rank_attr[0:start[i]]['Features']], y_train)
        y_probas=model.predict_proba(X_test[rank_attr[0:start[i]]['Features']])
        roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
        save_Best['dim'][i] = start[i]
        save_Best['roc'][i] = roc
        if roc>max_roc:
            max_roc = roc
            max_start = start[i]
    plt.plot(save_Best['dim'].astype(int), save_Best['roc'])
    return  max_start   

model = model

best_dim = getBestNumberOfFeatures(X_train, y_train, X_test, y_test, model, rank_attr)
new_dims = rank_attr[0:best_dim]['Features']
model.fit(X_train[new_dims], y_train)

y_probas=model.predict_proba(X_test[new_dims])
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()

saveResults[4][4] = roc
saveResults_Accuracy[4][4] = accuracy_score(y_test , model.predict(X_test[new_dims]))
Adaboost_CM.append(["Feature Selection" , confusion_matrix(y_test, model.predict(X_test[new_dims]))])
















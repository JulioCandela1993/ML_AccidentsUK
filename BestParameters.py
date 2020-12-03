# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:36:17 2019

@author: juliochristian
"""
import numpy as np
import pandas as pd
import itertools as it

model = AdaBoostClassifier(random_state = 42)

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



from copy import deepcopy
parameters = param_grid

from sklearn.svm import SVC

svc=SVC(probability=True, kernel='linear')
rfc = RandomForestClassifier(n_estimators=300
                             )
param_grid = {
    'base_estimator': [svc , rfc,None] ,
    'algorithm': ['SAMME'],
    'n_estimators': [1000,3000, 5000],
    'learning_rate': [0.05,0.01,0.005]
}

   
best_grid = TestParameters(model, X_train, X_test , y_train , y_test, param_grid)
print(model)

model = best_grid

model.fit(X_train, y_train)
y_probas=model.predict_proba(X_test)
roc = roc_auc_score(np.where(y_test==1 , 1 ,0), y_probas[:,0])
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()

saveResults[4][3] = roc
Adaboost_CM.append(["HyperTunning" , confusion_matrix(y_test, model.predict(X_test))])
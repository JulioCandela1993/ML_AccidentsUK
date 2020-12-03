# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 20:47:40 2019

@author: juliochristian
"""

import pandas as pd
import numpy as np
import math
import os

data = accidents_train
varList = new_categorial_analysis
type0 = 'cat' 
target_id = 'isFatal'

def WOE(data, varList, type0='Con', target_id='y'):

        result = pd.DataFrame()
        for var in varList:
            print(var)
            try:
                if type0.upper() == "CON".upper():
                    df, retbins = pd.qcut(data[var], q=10, retbins=True, duplicates="drop")
                    tmp = pd.crosstab(df, data[target_id], margins=True)
                    tmp2 = pd.crosstab(df, data[target_id], margins=True).apply(lambda x: x / float(x[-1]), axis=1)
                else:
                    df = data[var].fillna("Missing Info")
                    tmp = pd.crosstab(data[var], data[target_id], margins=True)
                    tmp2 = pd.crosstab(data[var], data[target_id], margins=True).apply(lambda x: x / float(x['All']),
                                                                                       axis=1)
                res = tmp.merge(tmp2, how="left", left_index=True, right_index=True)
                res['ratio'] = res['All_x'] / res['All_x'][-1] * 100
                res['DB'] = (res['0_x']) / res['0_x'][-1]  # Adjusting Woe +0.5
                res['DG'] = (res['1_x']) / res['1_x'][-1]  # Adjusting Woe +0.5
                res['WOE'] = np.log(res['DG'] / res['DB'])
                res['DG-DB'] = res['DG'] - res['DB']
                res['IV'] = res['WOE'] * res['DG-DB']
                res['name'] = var
                res.index.name = ""
                #res = res.drop("All", axis = 1)
                res['IV_sum'] = res['IV'].sum()
                del res['0_y']
                del res['All_y']
                if type0.upper() == "CON".upper():
                    res['low'] = retbins[:-1]
                    res['high'] = retbins[1:]
                    res.index = range(len(retbins) - 1)
                else:
                    res['low'] = res.index
                    res['high'] = res.index
                    res.reset_index
                res = res[
                    ['name', 'All_x', 'ratio', 'low', 'high', '0_x', '1_x', '1_y', 'DB', 'DG', 'WOE',
                     'DG-DB',
                     'IV', 'IV_sum']]
                result = result.append(res)

            except Exception as e:
                print(e, var)
        return result
    
X_data = accidents_train
X_map = cat_woe
var_list = new_categorial_analysis    
    
def applyWOE(X_data, X_map, var_list, id_cols_list=None, flag_y=None):
       
        if flag_y:
            bin_df = X_data[[flag_y]]
        else:
            bin_df = pd.DataFrame(X_data.index)
        for var in var_list:
            x = X_data[var].fillna("Missing Info")
            bin_map = X_map[X_map['name'] == var]
            bin_df[var] = 0.0000
            for i in bin_map.index:
                upper = bin_map['high'][i]
                lower = bin_map['low'][i]
                if lower == upper:
                    woe = 0 if bin_map['WOE'][i] == float('-inf') or bin_map['WOE'][i] == float('inf') else bin_map['WOE'][i]
                    bin_df[var] = np.where(x == lower, woe, bin_df[var])
        return bin_df

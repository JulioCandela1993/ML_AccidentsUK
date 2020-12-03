import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import folium
from folium.plugins import HeatMap
from folium.vector_layers import CircleMarker
from sklearn.cluster import DBSCAN
from math import log
from folium.plugins import MarkerCluster
from folium.features import DivIcon

os.chdir("G:\Documentos\MasterDegree\BDMA\Classes\DataMining\FinalProject\Data")

trainFile = 'accidents_2012_to_2014.csv'

accidents_train = pd.read_csv(trainFile)
columnnames = [accidents_train.columns , accidents_train.dtypes]

# Geographic heatmap
def getHotSpotsAccidents(lat_lon, fileOutput, eps, minpts, fig):
    
    size = [1 for i in range(len(lat_lon))]
    samplesize = min(len(lat_lon),100000)
    lat_lon_sample = lat_lon.sample(samplesize, random_state = 900)
    
    clustering = DBSCAN(eps=eps, min_samples=minpts).fit(lat_lon_sample)
    lat_lon_sample['cluster'] = clustering.labels_
    centroids = lat_lon_sample.groupby('cluster').agg({'Latitude':'mean','Longitude':'mean','cluster':'size'})# mean()
    plt.figure(fig)
    plt.bar(centroids.index , centroids['cluster'])
    centroids = centroids[centroids.index!=-1]
    centroids['Rank'] = centroids['cluster'].rank(ascending = False)
    
    hmap = folium.Map([(lat_lon["Latitude"].max() + lat_lon["Latitude"].min())/2 ,
                       (lat_lon["Longitude"].max() + lat_lon["Longitude"].min())/2],
                         zoom_start=6,
                         #tiles = "Stamen Terrain"
                         )
    
    hm_accidents = HeatMap( list(zip(lat_lon["Latitude"], lat_lon["Longitude"], size)),
                       #min_opacity=0.2,
                       max_val=max(size),
                       radius=7,  
                       max_zoom=1, 
                       gradient={0.1: 'blue', 0.2: 'lime', 0.7: 'orange', 0.9: 'red'},
                     )
    
    hmap.add_child(hm_accidents)
    
    for index, row in centroids.iterrows():
        
        popup_text = "{}<br> Cluster: {:,}<br> Freq: {:,}"
        popup_text = popup_text.format(
                          row['Rank'],
                          index,
                          row['cluster']
                          )
        
        hm_concentration = CircleMarker(
                location = [row['Latitude'], row['Longitude']],
                radius=log(row['cluster']) +2,
                popup = popup_text,
                color="#FF0000",
                threshold_scale=[0,1,2,3],
                #fill_color=colordict[traffic_q],
                fill=True
                )
        
        hm_text = folium.map.Marker(
                    [row['Latitude']+0.12, row['Longitude']-0.21],
                    icon=DivIcon(
                        icon_size=(20,20),
                        icon_anchor=(0,0),
                        html='<div style="font-size: 10pt" align= "center"><b>%s</b></div>' % str(int(row['Rank'])),
                        )
                    )
        hmap.add_child(hm_concentration)       
        hmap.add_child(hm_text) 
    hmap.add_child(hm_concentration)
    hmap.save(fileOutput)
    
    return(clustering)

def dbscan_predict(model, X):
    
    nr_samples = X.shape[0]

    y_new = np.ones(shape=nr_samples, dtype=int) * -1

    X = np.array(X)

    for i in range(nr_samples):
        diff = model.components_ - X[i][:]  # NumPy broadcasting

        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance

        shortest_dist_idx = np.argmin(dist)

        if dist[shortest_dist_idx] < model.eps:
            y_new[i] = model.labels_[model.core_sample_indices_[shortest_dist_idx]]

    return y_new


lat_lon = accidents_train[["Latitude","Longitude"]].dropna()

cluster_accidents = getHotSpotsAccidents(lat_lon, 'accidents_hs.html',0.12,600,1)
pred_cluster_accidents = dbscan_predict(cluster_accidents,lat_lon)

fatal_accidents = accidents_train[accidents_train['Accident_Severity'] == 1]
fatal_lat_lon = fatal_accidents[["Latitude","Longitude"]].dropna()

cluster_fatal = getHotSpotsAccidents(fatal_lat_lon, 'fatal_accidents_hs.html',0.2,40,2)
pred_cluster_fatal = dbscan_predict(cluster_fatal,lat_lon)
np.unique(pred_cluster_fatal)


accidents_train['cluster_accidents'] = pred_cluster_accidents
accidents_train['fatal_accidents'] = pred_cluster_fatal
accidents_train['isFatal'] = np.where(accidents_train['Accident_Severity'] == 1 , 1 , 0)

accidents_train.to_csv('accidents_clusterized.csv')

accidents_des = accidents_train.groupby('cluster_accidents').agg({'Latitude':'mean','Longitude':'mean','cluster_accidents':'size','isFatal':'sum'})
accidents_des['por_fatal'] = np.round(accidents_des['isFatal']/accidents_des['cluster_accidents'],4)
rank = np.array(accidents_des[accidents_des.index!=-1]['cluster_accidents'].rank(ascending = False))
accidents_des['Rank'] = np.where(accidents_des.index==-1,-1 , rank[accidents_des.index])
accidents_des.to_csv('cluster_accidents.csv')

fatal_des = accidents_train[accidents_train['isFatal'] == 1].groupby('fatal_accidents').agg({'Latitude':'mean','Longitude':'mean','fatal_accidents':'size','isFatal':'sum'})
rank = np.array(fatal_des[fatal_des.index!=-1]['fatal_accidents'].rank(ascending = False))
fatal_des['Rank'] = np.where(fatal_des.index==-1,-1 , rank[fatal_des.index])
fatal_des.to_csv('fatal_accidents.csv')

table = plt.table(cellText = np.array(accidents_des), colLabels = np.array(accidents_des.columns), loc='center')
table.scale(2, 2)
plt.axis('off')
plt.show()
table = plt.table(cellText = np.array(fatal_des), colLabels = np.array(fatal_des.columns), loc='center')
table.scale(2, 2)
plt.axis('off')
plt.show()


##### Bivar Plots and Charts

#n, bins, _ = plt.hist(x=accidents_train['Latitude'], bins=10, color='#0504aa',
#                            alpha=0.7, rwidth=0.85)




#null_in_column = []
#for column in df.columns:
#    null_in_column.append((column, df[column].isnull().sum(),str(df[column].isnull().sum()*100/len(df))+"%"))
#pd.DataFrame(null_in_column, columns=['Column Name', 'Total Missing', 'Percentage Missing'])


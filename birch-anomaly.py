import requests
import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import Birch
from sklearn.metrics import confusion_matrix

HOST = 'http://localhost:9000'

def execute_query(query_sql: str) -> pd.DataFrame:
    """
    This function takes in a string query_sql representing an SQL query and returns the results as a dataframe
    """
    query_params = {'query': query_sql, 'fmt' : 'json'}
    try:
        response = requests.get(HOST + '/exec', params=query_params)
        json_response = json.loads(response.text)
        skab_df = pd.DataFrame(data=json_response['dataset'], columns=[i['name'] for i in json_response['columns']])
        
        print(skab_df.shape)
        return skab_df
    except KeyError:
        print(json_response['error'])
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

def process_data(df: pd.DataFrame) -> tuple:
    del df['changepoint']
    df = df.astype({'anomaly':'float'}).astype({'anomaly':'int'})
    df.set_index('datetime', inplace=True)
    anomaly = df['anomaly']
    del df['anomaly']
    return df, anomaly

def get_labels(labels: list) -> list:
    """
    This function returns the anomaly label of the datapoints.
    """
    l, c = np.unique(labels, return_counts=True)
    print(l,c)
    mini = l[np.argmin(c)]
    return np.array([i==mini for i in labels], dtype='int64')

def plot_anomaly_distribution(labels: list, df: pd.DataFrame) -> None:
    """
    This function plots the anomaly distribution of the input dataframe given the labels.
    """
    X= PCA(n_components=2).fit_transform(df)
    plt.scatter(X[:,0], X[:,1], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.title("Anomaly Distribution on SKAB Data using BIRCH")
    plt.show()

# create dataframe
skab_df = execute_query("SELECT * FROM alldata_skab.csv WHERE anomaly IN('0.0', '1.0');") #  
skab_df, anomaly = process_data(skab_df)

birch_model = Birch(n_clusters=2,threshold=1.5,)
birch_model.fit(skab_df)
labels = birch_model.labels_

pred = get_labels(labels)
cm = confusion_matrix(anomaly, pred)
print(f"Anomalies detected: {cm[1][1]/sum(cm[1])}")

#plot result
plot_anomaly_distribution(labels, skab_df)
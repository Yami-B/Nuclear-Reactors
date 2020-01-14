# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 12:07:37 2020

@author: Yasta
"""
import csv
with open('C:\\Users\\Yasta\\Desktop\\project2\\capacity factors.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    line_count = 0
    matrix=[]
    for row in csv_reader:
        matrix.append([float(row[0]),float(row[1]),float(row[2]),float(row[3]),float(row[4]),
                       float(row[5]),float(row[6]),float(row[7]),float(row[8]),float(row[9]),
                       float(row[10])])
import pandas as pd
cf = pd.read_csv('C:\\Users\\Yasta\\Desktop\\project2\\capacity factors.csv') 

t_student=[[0 for i in range(0,96)] for j in range(0,96)]
from math import sqrt
for i in range(0,len(t_student)):
    for j in range(0,len(t_student[0])):
        if i!=j:
            x=cf[str(i+1)]-cf[str(j+1)]
            t_student[i][j]+=x.mean()*sqrt(11)/x.std()

similar_time_series=[]
for i in range(0,len(t_student)):
    for j in range(0,len(t_student[0])):
        if i>j and abs(t_student[i][j])<=2.228:
            similar_time_series.append((i+1,j+1))

import networkx as nx
import matplotlib.pyplot as plt
G=nx.Graph()
G.add_edges_from(similar_time_series)
nx.draw(G)
plt.show() # display

import community
dendo = community.generate_dendrogram(G)

list_values = [v for v in dendo[0].values()]

with open('C:\\Users\\Yasta\\Desktop\\project2\\community.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(list_values)
     
######
#Automatically forecatsing for each time series would be time consuming since 
#some of them are non stationary and would therefore specific types of models (like ARIMA
#or random  walk) and others not. Dealing with very short time series is not helping either.
#What we suggest here is to fit one general model to all the time series as it would increase the number
#of data.
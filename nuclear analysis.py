import csv
import pandas as pd

cf = pd.read_csv('capacity factors.csv') 
import matplotlib.pyplot as plt
from scipy import stats
fig, ax = plt.subplots(figsize=(6, 3))

_ = stats.probplot(
    cf['65'],       
    sparams=(0.3, 2),  
    dist=stats.beta,   
    plot=ax            
)

#The normal probability plot shows the time series don't follow a normal distribution
#Therefore to compare the time series, we must  use non parametric tests such as
#Kruskal Wallis test (one-way ANOVA but with no assumption on the distribution of samples)

from scipy.stats import kendalltau
dependent=[]
for i in range(0,96):
    for j in range(0,96):
        if i!=j:
            tau, p_value = kendalltau(cf[str(i+1)],cf[str(j+1)])
            if p_value <=0.05 and abs(tau)>=0.75 and ((j+1,i+1) not in dependent):
                dependent.append((i+1,j+1))
#We calculated the correlation between the time series of each plant
#and stored the plants that might have non linear relationship with other plants in
#the "dependent" list                

import networkx as nx
import matplotlib.pyplot as plt
G=nx.Graph()
G.add_edges_from(dependent)
pos = nx.spring_layout(G,k=0.15,iterations=25)
nx.draw(G,pos,with_labels=True)
plt.show() 
#This displays the plants and their relationship with others. In what follows,
#we only want to keep independent plants

nodes_to_remove=[15,33,48,80,75,47,17,7,60,50,29,53,64,70,41,38,37,81]
independent=cf
for i in range(0,len(nodes_to_remove)):
     independent.drop([str(nodes_to_remove[i])],axis=1,inplace=True)
#"independent" is the list of independent plants

BECH=[2,3,16,23,25,32,34,36,46,52,61,62,63,65,67,68,83,84,87,88,90,91,97]
SW=[4,5,51,54,55,56,74,85,86]
SL=[6,13,14,20,30,31,35,44,45,71,72]
def transform_mean(A):
    for i in range(0,len(A)):
        A[i]=cf[str(A[i]-1)].mean()
    return(A)
BECH_mean=transform_mean(BECH)
SW_mean=transform_mean(SW)
SL_mean=transform_mean(SL)
#These lists contain the means of capacity factors of each plant from the constructors BECH, S&L and S&W


from scipy.stats import levene
W,p_value=levene(BECH_mean,SW_mean,SL_mean)
print(W)
print(p_value)
#Hypothesis that the variances are equal can't be rejected and each sample has more than 5 individuals
#(p_value=0.28>0.05 ==> H0 failed to be rejected)
#All assumptions for Kruskal are respected

from scipy.stats import kruskal
H,p_value=kruskal(BECH_mean,SW_mean,SL_mean)
print(H)   
print(p_value)  
#p_value=0.002, it means the samples have different medians

data_dict = {'BECH': BECH_mean, 'SW': SW_mean, 'SL': SL_mean}
df = pd.DataFrame({k:pd.Series(v) for k,v in data_dict.items()})
df=df.melt(var_name='groups', value_name='values')
from scikit_posthocs import posthoc_dscf
posthoc_dscf(df, val_col='values', group_col='groups')
#output shows  each sample is different from the others: 
#       BECH        SL        SW
#BECH -1.000  0.001000  0.001000
#SL    0.001 -1.000000  0.009617
#SW    0.001  0.009617 -1.000000

#paired p_values are all smaller than 0.05, so each sample is significantly different from the
#two others
import pylab
BoxName = ['BECH','SL','SW']
data = [BECH_mean,SL_mean,SW_mean]
pylab.xticks([1,2,3], BoxName)
plt.boxplot(data);
plt.savefig('MultipleBoxPlot.png')
plt.show()
#From this plot and knowing that plants constructed by BECH, S&L and S&W are significantly different,
#we can say from the boxplot that S&L plants better than the others but S&W's plants are more reliable 
#than the others (smaller variance in means of capacity factors)



######
#Automatically forecasting for each time series would be time consuming since 
#some of them are non stationary and would therefore specific types of models (like ARIMA
#or random  walk) and others not. Dealing with very short time series is not helping either.
#What we suggest here is to fit one general model to all the time series as it would increase the number
#of data and therefore the reliability of the forecast
     

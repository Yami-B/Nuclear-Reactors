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


ro = pd.read_csv('C:\\Users\\Yasta\\Desktop\\project2\\reactors-operating 1.csv') 
NRC1=[]
NRC2=[]
NRC3=[]
NRC4=[]
NRC_matrix=[NRC1,NRC2,NRC3,NRC4]
for i in range(0,ro.shape[0]):
    if i+1 not in nodes_to_remove: 
       NRC_matrix[ro.loc[i,'NRC Region']-1].append(cf[str(i+1)])


def Mean(A):
    B=[]
    for i in range(0,len(A)):
        B.append(A[i].mean())
    return(B)
NRC1_mean=Mean(NRC1)
NRC2_mean=Mean(NRC2)
NRC3_mean=Mean(NRC3)
NRC4_mean=Mean(NRC4)

from scipy.stats import levene
W,p_value=levene(NRC1_mean,NRC2_mean,NRC3_mean,NRC4_mean)
print(W)
print(p_value)
#Hypothesis that the variances are equal can't be rejected and each sample has more than 5 individuals
#(p_value=0.45>0.05 ==> H0 failed to be rejected)
#All assumptions for Kruskal are respected

from scipy.stats import kruskal
H,p_value=kruskal(NRC1_mean,NRC2_mean,NRC3_mean,NRC4_mean)
print(H)   
print(p_value)  
#p_value=0.16 > 0.05, it means the samples have supposedly the same medians
#The plants don't differ significantly depending of NRC regions

import pylab
BoxName = ['NRC1','NRC2','NRC3','NRC4']
data = [NRC1_mean,NRC2_mean,NRC3_mean,NRC4_mean]
pylab.xticks([1,2,3,4], BoxName)
plt.boxplot(data)
plt.show()




from statsmodels.tsa.stattools import grangercausalitytests
from pandas import Series

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
#difference time series in order to be able to apply VAR fitting on stationary time series

date_rng = pd.date_range(start='1/2008', end='1/2019', freq='Y')
df=cf.iloc[::-1]
df['Date'] = pd.to_datetime(date_rng)
df.set_index(df['Date'],inplace=True)
df.drop(['Date'],inplace=True,axis=1)
upsampled = df.resample('M')
interpolated = upsampled.interpolate(method='linear')
#Time indexed the dataframe containing the capacity factors following years and 
#interpolated the points in order to have longer time series for forecasting


from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
from math import sqrt

ts1=interpolated['7']
ts2=interpolated['9']
RMSE=[]
RMSE1=[]
x=[]
#for i in range(70,110):
result1 = kpss(ts1)[1]
result2 = kpss(ts2)[1]
result3 = adfuller(ts1)[1]
result4 = adfuller(ts2)[1]
while(result1 >= 0.05 and result2 >= 0.05 and result4 <=0.05 and result3 <=0.05):
    ts1=difference(ts1)
    ts2=difference(ts2)
    result1 = kpss(ts1)[1]
    result2 = kpss(ts2)[1]
    result3 = adfuller(ts1)[1]
    result4 = adfuller(ts2)[1]
    print('NOK')
data=pd.DataFrame()
data['ts1']=ts1
data['ts2']=ts2
training_data=data.iloc[0:95,:]
from statsmodels.tsa.ar_model import AR
model = VAR(training_data)
model2=AR(ts2.iloc[0:95])
results = model.fit(maxlags=5, ic='aic')
results2 = model2.fit()
predictions = results2.predict(start=95, end=120, dynamic=False)
#results.plot_forecast(26)
#results2.plot_forecast(26)
error = mean_squared_error(ts2.iloc[95:], predictions)
print(error)
y_predicted=results.forecast(data.values[-5:], 121-95)
y_actual=data.iloc[95:,:]
rms = sqrt(mean_squared_error(y_actual.values[:,0], y_predicted[:,0]))
rms1 = sqrt(mean_squared_error(y_actual.values[:,1], y_predicted[:,1]))
print(rms1)

RMSE.append(rms)
RMSE1.append(rms1)
x.append(i)
plt.plot(x[5:],RMSE[5:])
plt.plot(x,RMSE1)

#Take the dependent time series and fit a Vector Autoregressive Model
#in order to improve the forecasts (since some time series are dependent:
#lagged values of one time serie might give information on future values of another time serie)

#Next we'll fit ARIMA model to independent plants' capacity factors' evolution in time



# Nuclear-Reactors: What makes a nuclear reactors more reliable ? + Attempt in forecasting future performances.

Study of a dataset dealing with the nuclear plants located in the USA. I downloaded a dataset from the NRC including all kind of informations such as the location of the plant, the reactor type, the parent plant name, the capacity factor (very crucial figure for any electricity generating plant) during last years,..

Here I'm trying to figure out what makes a nuclear plant significantly more productive than the others. I cleaned the original dataset from the NRC (reactors-operating.xls) and cleaned it:

-website page columns deleted
-added STAR column (strategic teaming ressource)
-docket number removed
-added state and city for my Tableau map 
...

For the moment, I tried to identify patterns in my data doing MCA analysises and I noticed that plants were prone to choose the same architect and constructor (which is in fact quiet logical since you don't want to introduce too much variability in the process of constructing anything from plants to mechanical parts).

![The time series values don't follow the red line --> no normal distribution](https://github.com/Yami-B/Nuclear-Reactors/blob/master/normal%20plot.png)

# Detect dependency between the plants performances 

The figure above is Quantile-Quantile plot of one of the time series, to test whether it follows a normal distribution. If it was the case the blue dots would have followed the red line, which is clearly not the case. This means that we are dealing with non normal distributed time series and there fore the famous Student test could not be applied here to test similarity between time series. Therefore, I had to use a regularly used non parametric method to compare two samples called the Kruskal Wallis test. However the assumption of this test is that, the time series being tested are independent, so the first thing to do is separate dependent time series and independent time series.

For the purpose of detecting dependency we chose the Kandal Tau test (non parametric method) which calculates the correlation between the two variables (here the time series) based on the ranks of each value, and when this correlation is significant (in litterature, it often is when the correlation coefficient is greater than 0.70, but we took 0.75 to give us a marge), we created a graph representing the supposedly dependency links between plants (only their indexes are indicated, see figure below). 

![Dependency between plants](https://github.com/Yami-B/Nuclear-Reactors/blob/master/Dependency%20graph.png)

These plants are therefore not included in our study similarity between plants.

# Are plants constructed by BECH, S&L and S&L perform the same ?

Now we have got independent plants, we would like to know if the constructor of the plant has an influence on the performance of the plants. We only took the three constructors who took part in the construction of most the plants (>5), because otherwise the results could not be trusted. So, we separated the means of capacity factors by years in 3 groups, depending on which constructor BECH, S&L or S&W participated in establishing the plant. Now to use Kruskal Wallis, we have got to check the following assumptions:

-independent variables 
-same variance

We already have independent time series, to test the assumption of the same shape of the samples (more precisely same variance) we use the non parametric version of Levene's test. The test on the 3 groups showed that they indeed shared the same "shape", therefore the Kruskal Wallis can eventually be applied check if the performance of the 3 constructors significantly differ. As in in the python file,
the results of the test show that each constructor is significantly different from others, the following box plot represents the repartition of the capacity factors following the constructor:

![Mean performances folling the constructor](https://github.com/Yami-B/Nuclear-Reactors/blob/master/MultipleBoxPlot.png)

Interpretation: We can say that S&L plants perform better than the others but S&W's plants are more consistent 
than the others (less variation in means of capacity factors)

# Does the NRC (Nuclear Regulatory Commission) region have any kind of influence on the plants' performances ?

Following the same steps, Kruskal Wallis test showed that we could not reject the hypothesis that the perforancec of plants were significantly different dependeing NRC region. Therefore, we can't say that it really has an influence and we show the corresponding box plots.

![Mean performances folling the NRC region](https://github.com/Yami-B/Nuclear-Reactors/blob/master/MultipleBoxPlotNRC.png)

# Forecasting future capacity factors of each plant



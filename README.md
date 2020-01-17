# Nuclear-Reactors

Study of a dataset dealing with the nuclear plants located in the USA. I downloaded a dataset from the NRC including all kind of informations such as the location of the plant, the reactor type, the parent plant name, the capacity factor (very crucial figure for any electricity generating plant) during last years,..

Here I'm trying to figure out what makes a nuclear plant significantly more productive than the others. I cleaned the original dataset from the NRC (reactors-operating.xls) and cleaned it:

-website page columns deleted
-added STAR column (strategic teaming ressource)
-docket number removed
-added state and city for my Tableau map 
...

For the moment, I tried to identify patterns in my data doing MCA analysises and I noticed that plants were prone to choose the same architect and constructor (which is in fact quiet logical since you don't want to introduce too much variability in the process of constructing anything from plants to mechanical parts).

I'm also trying analysis approaches on the short time series I've got. I ploted normal probablity plot for each time series and I came to the conclusion that the capacity factors values didn't follow a normal distribution. Therefore, to test for example if time series are similar I have to use non-parametric test such as Kruskal Wallis test or Friedman test instead of regular T-test. The main issue will be to assess whether the the similarity of the time series is significant since my tiime series are really short (11 instances is considered short).

![The time series values don't follow the red line --> no normal distribution](https://github.com/Yami-B/Nuclear-Reactors/blob/master/normal%20plot.png)





# Machine Learning Engineer Nanodegree
## Capstone Project
Patrick Poon  
May 25, 2018

## I. Definition
<!--
_(approx. 1-2 pages)_
-->

### Project Overview

<!--
In this section, look to provide a high-level overview of the project in layman’s terms. Questions to ask yourself when writing this section:
- _Has an overview of the project been provided, such as the problem domain, project origin, and related datasets or input data?_
- _Has enough background information been given so that an uninformed reader would understand the problem domain and following problem statement?_
-->

> Most people...the interaction that they're going to have with a police officer is because [...] they're stopped for speeding.  Or, forgetting to turn their blinker off.[1]
> 
>-- Cheryl Phillips, Journalism Professor at Stanford University

On a typical day in the United States, police officers make more than 50,000 traffic stops.[2]  In recent years, there have been numerous incidents that have made national headlines that involved an officer shooting and, in some cases, killing the driver or an occupant.  Many cite racial biases against Blacks and Hispanics for the disproportionate number of such incidents.  Here are some relevant articles:

- Was the Sandra Bland traffic stop legal -- and fair? (https://www.cnn.com/2015/07/23/opinions/cevallos-sandra-bland-traffic-stop/index.html)
- Philando Castile shooting: Dashcam video shows rapid event (https://www.cnn.com/2017/06/20/us/philando-castile-shooting-dashcam/index.html)

This Capstone project will not attempt to prove or disprove this controversial topic, and will attempt to  avoid making any controversial or provocative statements on either side of the conversation.


### Problem Statement

<!--
In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section:
- _Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?_
- _Have you thoroughly discussed how you will attempt to solve the problem?_
- _Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?_
-->

Instead, this project aims to create a multi-class classifier that takes various discrete traffic stop situational values to predict the outcome of a traffic stop, specifically in the state of Connecticut (CT).  Given a driver's age, gender, race, traffic stop violation, and the county in which a traffic stop occurs, can we reliably predict whether the traffic stop will result in a verbal/written warning, a ticket, a summons to appear in court, or an arrest?

To accomplish this task, I will parse and process traffic stop data for the state of Connecticut, and feed it into a supervised learning algorithm that I will train and tune to predict these outcomes.  The data comes from the Stanford Open Policing Project (SOPP) at https://openpolicing.stanford.edu/data/. SOPP has collected data for 31 states, but the CT dataset was the cleanest and most consistent.

<!-- 
Direct file link:  https://stacks.stanford.edu/file/druid:py883nd2578/CT-clean.csv.gz
-->

### Metrics
<!--
In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:
- _Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?_
- _Have you provided reasonable justification for the metrics chosen based on the problem and solution?_
-->

For this project, I will use **accuracy classification score** as my evaluation metric.  According to the scikit-learn page for the **`accuracy_score`** function[3], in the context of multiclass classification, the function is equivalent to the **`jaccard_similarity_score`** function which calculates the Jaccard index[4], also known as "Intersection over Union," as illustrated in the following formula:

<center>
![Intersection over Union formula](./IoU.svg)
</center>



## II. Analysis
<!--
_(approx. 2-4 pages)_
-->

### Data Exploration & Exploratory Visualization
<!--
In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any Mabnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_
-->

<!--
### Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_
-->

In this section, I will break down and decompose the raw data into its basic elements.  In doing so, I will attempt to gain insights that may guide me at different points in my journey to develop an effective and accurate classifier.  I will start by presenting some sample records, then discuss why certain columns should be dropped, and finally explore characteristics of some columns that may provide predictive power for my classifier.



```python
from datetime import datetime
import itertools

import humanize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display, Math, Latex
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tabulate import tabulate

%matplotlib inline

```


```python
df = pd.read_csv('./data/CT-clean.csv', header=0)
```

    /home/pato/anaconda2/envs/py36/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2698: DtypeWarning: Columns (22) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)


The raw data is available at https://stacks.stanford.edu/file/druid:py883nd2578/CT-clean.csv.gz, and is comprised of 318,669 records with 24 feature columns, collected over a period of 1 year and 5 months from 2013 to 2015.  A few of those columns, namely **`driver_age`**, **`driver_race`**, and **`search_type`**, have overlapping information as these fields have two columns with the name format of "X" and "X_raw", where the "X" values are cleaned or adjusted "X_raw" values.  

Here are a few sample rows with the feature columns broken down into three sections (**Please note**: Different rows have been selected for each section to provide a sense of the complexity involved with the different columns in this dataset):


```python
df.shape
```




    (318669, 24)




```python
df.iloc[:,:8].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>state</th>
      <th>stop_date</th>
      <th>stop_time</th>
      <th>location_raw</th>
      <th>county_name</th>
      <th>county_fips</th>
      <th>fine_grained_location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CT-2013-00001</td>
      <td>CT</td>
      <td>2013-10-01</td>
      <td>00:01</td>
      <td>westport</td>
      <td>Fairfield County</td>
      <td>9001.0</td>
      <td>00000 N I 95 (WESTPORT, T158) X 18 LL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CT-2013-00002</td>
      <td>CT</td>
      <td>2013-10-01</td>
      <td>00:02</td>
      <td>mansfield</td>
      <td>Tolland County</td>
      <td>9013.0</td>
      <td>rte 195 storrs</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CT-2013-00003</td>
      <td>CT</td>
      <td>2013-10-01</td>
      <td>00:07</td>
      <td>franklin</td>
      <td>New London County</td>
      <td>9011.0</td>
      <td>Rt 32/whippoorwill</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CT-2013-00004</td>
      <td>CT</td>
      <td>2013-10-01</td>
      <td>00:10</td>
      <td>danbury</td>
      <td>Fairfield County</td>
      <td>9001.0</td>
      <td>I-84</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CT-2013-00005</td>
      <td>CT</td>
      <td>2013-10-01</td>
      <td>00:10</td>
      <td>east hartford</td>
      <td>Hartford County</td>
      <td>9003.0</td>
      <td>00000 W I 84 (EAST HARTFORD, T043)E.OF XT.56</td>
    </tr>
  </tbody>
</table>
</div>



<br>


```python
df.iloc[24500:24506,8:16].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>police_department</th>
      <th>driver_gender</th>
      <th>driver_age_raw</th>
      <th>driver_age</th>
      <th>driver_race_raw</th>
      <th>driver_race</th>
      <th>violation_raw</th>
      <th>violation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>24500</th>
      <td>State Police</td>
      <td>M</td>
      <td>39</td>
      <td>39.0</td>
      <td>White</td>
      <td>White</td>
      <td>Speed Related</td>
      <td>Speeding</td>
    </tr>
    <tr>
      <th>24501</th>
      <td>State Police</td>
      <td>M</td>
      <td>62</td>
      <td>62.0</td>
      <td>White</td>
      <td>White</td>
      <td>Cell Phone,Other</td>
      <td>Cell phone,Other</td>
    </tr>
    <tr>
      <th>24502</th>
      <td>State Police</td>
      <td>F</td>
      <td>31</td>
      <td>31.0</td>
      <td>White</td>
      <td>White</td>
      <td>Registration</td>
      <td>Registration/plates</td>
    </tr>
    <tr>
      <th>24503</th>
      <td>State Police</td>
      <td>F</td>
      <td>50</td>
      <td>50.0</td>
      <td>Hispanic</td>
      <td>Hispanic</td>
      <td>Other</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>24504</th>
      <td>State Police</td>
      <td>M</td>
      <td>28</td>
      <td>28.0</td>
      <td>White</td>
      <td>White</td>
      <td>Registration</td>
      <td>Registration/plates</td>
    </tr>
  </tbody>
</table>
</div>



<br>


```python
df.iloc[242:248,16:24].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>search_conducted</th>
      <th>search_type_raw</th>
      <th>search_type</th>
      <th>contraband_found</th>
      <th>stop_outcome</th>
      <th>is_arrested</th>
      <th>officer_id</th>
      <th>stop_duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>242</th>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>Verbal Warning</td>
      <td>False</td>
      <td>1000002364</td>
      <td>1-15 min</td>
    </tr>
    <tr>
      <th>243</th>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>Ticket</td>
      <td>False</td>
      <td>1000001904</td>
      <td>16-30 min</td>
    </tr>
    <tr>
      <th>244</th>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>Summons</td>
      <td>False</td>
      <td>41354688</td>
      <td>1-15 min</td>
    </tr>
    <tr>
      <th>245</th>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>Written Warning</td>
      <td>False</td>
      <td>348145142</td>
      <td>1-15 min</td>
    </tr>
    <tr>
      <th>246</th>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>Ticket</td>
      <td>False</td>
      <td>1000001914</td>
      <td>1-15 min</td>
    </tr>
  </tbody>
</table>
</div>




```python
#df.info()
```

The dataset is primarily comprised of discrete categorical values with only three columns that contain numerical data, namely **`driver_age`**, **`driver_age_raw`**, and **`county_fips`**.  However, **`county_fips`** is unlikely to yield any predictive benefit numerically as the values are simple label identifiers for values in the **`county_name`** column.  The **`county_fips`** column can be dropped, and the **`county_name`** column will be one-hot encoded.  The **`driver_age`** column can also be dropped as it duplicates information in the **`driver_age_raw`** column.  **`driver_age`** also has missing values, as the following table shows:


```python
df.isnull().sum().to_frame('null values count')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>null values count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>0</td>
    </tr>
    <tr>
      <th>state</th>
      <td>0</td>
    </tr>
    <tr>
      <th>stop_date</th>
      <td>0</td>
    </tr>
    <tr>
      <th>stop_time</th>
      <td>222</td>
    </tr>
    <tr>
      <th>location_raw</th>
      <td>41</td>
    </tr>
    <tr>
      <th>county_name</th>
      <td>42</td>
    </tr>
    <tr>
      <th>county_fips</th>
      <td>42</td>
    </tr>
    <tr>
      <th>fine_grained_location</th>
      <td>1663</td>
    </tr>
    <tr>
      <th>police_department</th>
      <td>0</td>
    </tr>
    <tr>
      <th>driver_gender</th>
      <td>0</td>
    </tr>
    <tr>
      <th>driver_age_raw</th>
      <td>0</td>
    </tr>
    <tr>
      <th>driver_age</th>
      <td>274</td>
    </tr>
    <tr>
      <th>driver_race_raw</th>
      <td>0</td>
    </tr>
    <tr>
      <th>driver_race</th>
      <td>0</td>
    </tr>
    <tr>
      <th>violation_raw</th>
      <td>0</td>
    </tr>
    <tr>
      <th>violation</th>
      <td>0</td>
    </tr>
    <tr>
      <th>search_conducted</th>
      <td>0</td>
    </tr>
    <tr>
      <th>search_type_raw</th>
      <td>313823</td>
    </tr>
    <tr>
      <th>search_type</th>
      <td>313823</td>
    </tr>
    <tr>
      <th>contraband_found</th>
      <td>0</td>
    </tr>
    <tr>
      <th>stop_outcome</th>
      <td>5356</td>
    </tr>
    <tr>
      <th>is_arrested</th>
      <td>5356</td>
    </tr>
    <tr>
      <th>officer_id</th>
      <td>0</td>
    </tr>
    <tr>
      <th>stop_duration</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



One glaring observation with this table is that the **`search_type_raw`** and **`search_type`** columns mostly contain null values and should be dropped.  These fields provide supplementary information when a search is conducted, with both columns containing one of the following values:  "Consent", "Other", "Inventory", or `nan` (not a number).  The **`search_conducted`** boolean column, by itself, should provide an adequate signal about a probable outcome when a car search is involved.

The next two columns that have the highest number of null values are **`stop_outcome`** and **`is_arrested`**.  The **`is_arrested`** column should be dropped, because "Arrest" is one of the outcome values, and keeping this column would defeat the purpose of creating this classifier.  It would also be cheating in a sense.  Next, the rows that contain null values for **`stop_outcome`** should be dropped, since the main objective of this project is to predict the outcome of a traffic stop.  It would not make sense to replace the null values for this field with a median or average value.

There are a few other columns that make sense to drop as well:
- **`id`** column values, like "CT-2013-00001", provide no predictive value.
- **`state`** and **`police_department`** columns only have one value each, "CT" and "State Police" respectively. 
- **`location_raw`** contains the specific city in which a traffic stop occurred.  But, this data may be too granular, and better insight might be gained by using the **`county_name`** instead.
- **`fine_grained_location`** values are inconsistent and non-standardized, as the values appear to be simple notes that the officer took about the spot where the traffic stop was conducted.
- **`driver_race_raw`** column duplicates data in the **`driver_race`** column.
- **`officer_id`** has 2,105 unique values and might be too granular, making overfitting likely, so it will be dropped.

One could make a case that **`location_raw`** and/or **`officer_id`** should be kept.  Even though these fields contain granular data that may be too specific to the point that it may contribute to overfitting, they may provide signals for certain biases that lead to certain outcomes.  For example, certain officers may have a propensity to issue a **`Verbal Warning`** to certain demographics instead of issuing a **`Ticket`**.  If my classifier's prediction accuracy seems to reach a ceiling during model development and exhaustive hyperparameter tuning, I will consider adding one or both of these columns back into the training data.

In the following sub-sections I will discuss the columns that comprise the input features that will be used to train my classifier.


```python
# Drop columns that clearly should be dropped

drop_cols = [
    'county_fips',
    'driver_age',
    'driver_race_raw',
    'fine_grained_location',
    'id',
    'is_arrested',
    'officer_id',
    'police_department',
    'search_type_raw',
    'search_type',
    'state',
]

df.drop(drop_cols, axis=1, inplace=True)
```


```python
#df.shape
```


```python
#df.isnull().sum().to_frame('null value count')
```


```python
# Drop empty stop_outcome and county_name/county_fips rows
df.dropna(subset=['stop_outcome', 'county_name'], axis=0, inplace=True)
```


```python
df.shape
```


```python
df.isnull().sum()
```

#### Traffic Stop Outcome Breakdown

The values from the **`stop_outcome`** column will serve as the output labels for my classifier.  Graphing the value distribution for this column makes it clear that the data set is **highly imbalanced**.


```python
outcome_breakdown = df['stop_outcome'].value_counts(normalize=True).mul(100).plot.bar(figsize=(15, 3), table=True, fontsize=14, title="Traffic Stop Outcome Breakdown")
outcome_breakdown.axes.get_xaxis().set_visible(False)
outcome_breakdown.axes.set_ylabel('%', fontsize=14)
outcome_breakdown.tables[0].auto_set_font_size(False)
outcome_breakdown.tables[0].set_fontsize(14)
outcome_breakdown.tables[0].scale(1, 2)
```


![png](output_27_0.png)


A vast majority of traffic stops result in the officer issuing a "`Ticket`" in 69.89% of the cases.  "`Arrest`"s comprise only 2.33% of traffic stops.  Some lucky drivers are issued warnings, verbal or written, 23.9% of the time, while a few unfortunate drivers receive a "Summons" to appear in court in 3.9% of traffic stops.  There is a high risk that a trained model using this dataset unaltered will have a strong bias towards predicting "Ticket" as the outcome if not handled properly.

#### Proportion of Searches Conducted Relative to All Stops

One interesting data point that CT officers collect is whether a search was conducted during the traffic stop, as captured in the **`search_conducted`** column.  When True, these traffic stops should correlate with a higher number of outcomes resulting in an "Arrest".


```python
searches = df['search_conducted'].value_counts(normalize=True).mul(100).plot.bar(figsize=(8, 2), table=True, title="Proportion of Searches Conducted Relative to All Stops")
searches.axes.get_xaxis().set_visible(False)
searches.axes.set_ylabel('%')
searches.tables[0].set_fontsize(14)
searches.tables[0].scale(1, 2)
```


![png](output_30_0.png)


#### Outcomes When Vehicle Searched

Indeed, the proportion of Arrests rises to 27.69% of traffic stops when a search is conducted.


```python
outcome_when_searched = df[df['search_conducted'] == True]['stop_outcome'].value_counts(normalize=True).mul(100).plot.bar(figsize=(15, 5), table=True, title="Outcomes When Vehicle Searched")
outcome_when_searched.axes.get_xaxis().set_visible(False)
outcome_when_searched.axes.set_ylabel('%')
outcome_when_searched.tables[0].auto_set_font_size(False)
outcome_when_searched.tables[0].set_fontsize(14)
outcome_when_searched.tables[0].scale(1, 2)

```


![png](output_32_0.png)


Still, searches only comprise 1.7% of all traffic stops, so the fact remains that the data set is highly imbalanced.

#### Outcome by `stop_duration`

Another interesting data point is the duration of the traffic stop in the **`stop_duration`** column, which may provide another predictive signal about the likely outcome.  Logically speaking, the longer the duration of a traffic stop, the more likely the outcome will be an "Arrest", as the officer may need to ask more questions, search the vehicle, conduct a sobriety test, and perform other duties that take time and extend the duration of the traffic stop.  


```python
# duration_outcomes = df.groupby(['stop_duration', 'stop_outcome'])
# duration_outcomes.agg({'id': 'count'}).groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).rename(columns={'id': '%'})
```


```python
duration_dummies = pd.get_dummies(df.stop_duration)
outcomes_by_duration = pd.concat([df.stop_outcome, duration_dummies], axis=1)
outcomes_by_duration_grouped = outcomes_by_duration.groupby(['stop_outcome']).agg({x: 'sum' for x in duration_dummies.columns.values}).apply(lambda x: 100 * x / float(x.sum())).T
ax = outcomes_by_duration_grouped.plot.barh(figsize=(15, 5), width=0.9, fontsize=14, title="Outcome by stop_duration")
ax.set_xlabel('%', fontsize=14)
ax.set_ylabel('Stop Duration', fontsize=14)
```




    Text(0,0.5,'Stop Duration')




![png](output_36_1.png)


As suspected, the chances of a traffic stop resulting in an "Arrest", as illustrated by the blue bars, is much higher when the stop lasts longer than 30 minutes at 35.64% than 1.01% when the stop lasts only 15 minutes or less.

#### Race Breakdown

Another potential signal for the outcome of a traffic stop is race.




```python
stops_by_race = df['driver_race'].value_counts(normalize=True).mul(100).plot.bar(figsize=(15, 4), table=True, title="Race Breakdown", fontsize=14)
stops_by_race.axes.get_xaxis().set_visible(False)
stops_by_race.axes.set_ylabel('%', fontsize=14)
stops_by_race.tables[0].set_fontsize(14)
stops_by_race.tables[0].scale(1, 2)

```


![png](output_39_0.png)


The barchart shows that the majority (76%) of traffic stops involved "White"s, with "Black"s at 11.75%, "Hispanic"s at 9.78%, "Asian"s at 1.87%, and a catch-all value of "Other" at 0.55%.  This approximately matches the 2010 census figures for Connecticut[5], where the racial composition is 77.57% "White", 10.14% "Black", 13.4% "Hispanic", 3.79% "Asian", and roughly 5.55% "Other".  Hispanics are separated out by Ethnicity, so the numbers I provided have some overlap and sum to more than 100%.

![CT 2010 Census Data](./images/CT-2010-census-data.PNG)

One interesting note about the "Other" value is that in the **`driver_race_raw`** column of the dataset, this value was denoted as "Native American" and was subsequently replaced with "Other" in the **`driver_race`** column.

#### Arrests by County by Race

Looking for further insights, I thought it might be interesting to analyze racial breakdown of traffic stops that resulted in an "Arrest" by county.  Please note that the values are in percentages.


```python
# df.loc[df['stop_outcome'] == 'Arrest'].groupby(['county_name', 'driver_race']).agg({'id': 'count'}).groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).rename(columns={'id': '%'})
```


```python
arrested = df.loc[df['stop_outcome'] == 'Arrest']
arrested_dummies = pd.get_dummies(arrested.county_name)
arrested_races_by_county = pd.concat([arrested.driver_race, arrested_dummies], axis=1)
county_races = arrested_races_by_county.groupby(['driver_race']).agg({x: 'sum' for x in arrested_dummies.columns.values}).apply(lambda x: 100 * x /float(x.sum())).T
county_races
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>driver_race</th>
      <th>Asian</th>
      <th>Black</th>
      <th>Hispanic</th>
      <th>Other</th>
      <th>White</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Fairfield County</th>
      <td>1.007049</td>
      <td>16.918429</td>
      <td>28.700906</td>
      <td>0.604230</td>
      <td>52.769386</td>
    </tr>
    <tr>
      <th>Hartford County</th>
      <td>1.319797</td>
      <td>23.553299</td>
      <td>19.289340</td>
      <td>0.507614</td>
      <td>55.329949</td>
    </tr>
    <tr>
      <th>Litchfield County</th>
      <td>1.140065</td>
      <td>3.094463</td>
      <td>10.097720</td>
      <td>0.325733</td>
      <td>85.342020</td>
    </tr>
    <tr>
      <th>Middlesex County</th>
      <td>1.206897</td>
      <td>12.586207</td>
      <td>13.103448</td>
      <td>0.000000</td>
      <td>73.103448</td>
    </tr>
    <tr>
      <th>New Haven County</th>
      <td>0.922819</td>
      <td>17.533557</td>
      <td>22.315436</td>
      <td>0.251678</td>
      <td>58.976510</td>
    </tr>
    <tr>
      <th>New London County</th>
      <td>1.277235</td>
      <td>8.715252</td>
      <td>9.541698</td>
      <td>0.450789</td>
      <td>80.015026</td>
    </tr>
    <tr>
      <th>Tolland County</th>
      <td>1.121076</td>
      <td>7.735426</td>
      <td>9.865471</td>
      <td>0.112108</td>
      <td>81.165919</td>
    </tr>
    <tr>
      <th>Windham County</th>
      <td>0.698324</td>
      <td>4.748603</td>
      <td>17.318436</td>
      <td>0.418994</td>
      <td>76.815642</td>
    </tr>
  </tbody>
</table>
</div>




```python
# races_by_county
county_graph = county_races.plot.barh(figsize=(15, 8), width=0.9, fontsize=14)
county_graph.axes.set_xlabel('%', fontsize=14)

```




    Text(0.5,0,'%')




![png](output_46_1.png)


While the racial breakdown of traffic stops for the state overall are, for the most part, consistent with 2010 census data, these figures suggest that there are certain counties where Blacks and Hispanics are pulled over disproportionately higher than their composition in the 2010 census, namely in the Fairfield, Hartford, and New Haven counties.  This is not to suggest that they are being pulled over because of racial bias but is simply stated as an observation.

#### Traffic Stops by County

The distribution of traffic stops by county is moderately distributed.  Almost half of traffic stops occurred in New Haven, New London, and Tolland counties.


```python
outcomes_by_county = df['county_name'].value_counts(normalize=True).mul(100).apply(lambda x: float('{:.6f}'.format(x))).plot.bar(figsize=(15, 5), table=True, fontsize=12, title="Traffic Stops by County")
outcomes_by_county.axes.get_xaxis().set_visible(False)
outcomes_by_county.axes.set_ylabel('%', fontsize=18)
outcomes_by_county.tables[0].auto_set_font_size(False)
outcomes_by_county.tables[0].set_fontsize(10)
outcomes_by_county.tables[0].scale(1, 3)
```


![png](output_50_0.png)


#### Outcomes by Violations

Two columns, **`violation`** and **`violations_raw`**, provide information about the related violation(s) involved in a traffic stop outcome.  Unfortunately, these columns suffer from repetitive values and inconsistent data entry issues.  For example, some values are phrased differently yet have the same meaning.  I will need to settle on a standard value for these duplicate values and perform one-hot encoding for each class value, since multiple violations can be associated with a single traffic stop.


```python
def normalize_violation(violation):
    """Normalize violation values
    """
    if violation == 'defective lights':
        return 'lights'
    elif violation == 'equipment violation':
        return 'equipment'
    elif violation == 'other/error':
        return 'other'
    elif violation == 'registration/plates':
        return 'registration'
    elif violation == 'seat belt':
        return 'seatbelt'
    elif violation == 'speed related':
        return 'speeding'
    elif violation == 'stop sign/light' or violation == 'stop sign':
        return 'bad_stop'
    return violation.replace(' ', '_')


def merge_violations(violations):
    """Merge violation and violation_raw columns
    """
    merged = []
    tokens = violations.lower().split(',')
    return list(set([normalize_violation(violation) for violation in tokens]))
    

def onehot_encode_violations(arr_violations):
    row = np.zeros(len(violations))
    for v in arr_violations:
        row[violations.index(v)] = 1
    return row


violations = []

for violation in list(df.violation.unique()) + list(df.violation_raw.unique()):
    tokens = violation.lower().split(',')
    violations.extend([normalize_violation(token) for token in tokens])

violations = sorted(set(violations))

merged = df[['violation_raw', 'violation']].apply(lambda x: ','.join(x), axis=1).apply(merge_violations)

violation_col_headers = ['violation_{}'.format(violation.replace(' ', '_')) for violation in violations]

df_violations = merged.apply(onehot_encode_violations).apply(lambda x: pd.Series(x, dtype=int))
df_violations.columns = violation_col_headers
```

The distribution of violation values shows that a majority of traffic stops involve speeding.  Unfortunately, "Other" represents a large portion of violations, which is not descriptive and may add noise to the training set.


```python
violations = df_violations.copy(deep=True)
violations.columns = [x.replace('violation_', '') for x in violations.columns.values]
violation_breakdown = violations.sum().sort_values(ascending=False).plot.bar(figsize=(17, 5), fontsize=14)
violation_breakdown.axes.set_ylabel('Number of traffic stops', fontsize=14)
```




    <matplotlib.text.Text at 0x22817a8d8d0>




![png](output_54_1.png)


The bar chart below shows the **outcome percentages** by violation type.


```python
violation_outcomes = pd.concat([df.stop_outcome, df_violations], axis=1)

agg_dict = {}
outcomes_counts_by_violation = violation_outcomes.groupby(['stop_outcome']).agg({x: 'sum' for x in violation_col_headers}).apply(lambda x: 100 * x /float(x.sum())).rename(columns={x: x.replace('violation_','') for x in violation_col_headers}).T

outcomes_counts_by_violation_chart = outcomes_counts_by_violation.plot.barh(figsize=(15,15), width=0.9, fontsize=14)
outcomes_counts_by_violation_chart.axes.set_xlabel('%', fontsize=18)
```




    <matplotlib.text.Text at 0x227d9fa3860>




![png](output_56_1.png)


Not surprisingly, most violations resulted in a "Ticket", but there are a few exceptions:
- Expired and suspended driver's licenses were more likely to result in a "Summons" to appear in court, as shown by the two elongated orange-colored bars.
- Improper display of license plates and non-functional lights were likely to result in a "Verbal Warning."

#### Date and Time

The **`stop_date`** and **`stop_time`** columns specify when a traffic stop occurred.  In the hopes of detecting potential time patterns, I will transform these into four columns: **`month`**, **`day`**, **`hour`**, and **`minute`**.


```python
# [TODO: REMOVEME?]
# import pandas as pd
# %matplotlib inline
# df = pd.read_csv('./data/CT-clean.csv', header=0)
# day_freq = df['stop_date'].apply(lambda x: x.split('-')[2])
# month_freq = df['stop_date'].apply(lambda x: x.split('-')[1])
# dt_outcomes = pd.get_dummies(df['stop_outcome'])
# dt = pd.concat([month_freq, day_freq, df['stop_outcome']], axis=1)
# cols = ['month', 'day', 'outcome']
# # cols.extend(list(dt_outcomes.columns.values))
# dt.columns=cols
# grouped = dt.groupby(['day'])['outcome'].count().plot.barh(figsize=(15, 15), fontsize=14)
# grouped.axes.set_xlabel('outcome count', fontsize=14)
# grouped.axes.set_ylabel('day', fontsize=14)

```

#### Gender Breakdown

Twice as many men got pulled over compared to women, as men comprised almost exactly two-thirds (66.5%) of this dataset.


```python
# gender_vbar = df['driver_gender'].value_counts(normalize=True).mul(100).plot.bar(figsize=(8, 4), table=True)
# xaxis = gender_vbar.axes.get_xaxis()
# xaxis.set_visible(False)
# gender_vbar.axes.set_ylabel('%')
# table = gender_vbar.tables[0]
# table.scale(1, 2)
```


```python
gender_vbar = df['driver_gender'].value_counts(normalize=True).mul(100).plot.pie(figsize=(4, 4), fontsize=16)
xaxis = gender_vbar.axes.get_xaxis()
xaxis.set_visible(False)
gender_vbar.axes.set_ylabel('')
# gender_vbar.axes.set_ylabel('%')
# table = gender_vbar.tables[0]
# table.scale(1, 2)
```




    <matplotlib.text.Text at 0x227ee600a90>




![png](output_62_1.png)


#### Age Breakdown

Those in their 20's have the highest percentage of traffic stops at 30.66%, with those in their 30's and 40's following suit, at 20.9% and 18.9% respectively.


```python
# df['driver_age_raw'].describe()

# [TODO:  Remove or make a pretty table]
```




    count    313274.000000
    mean         38.066325
    std          14.428419
    min           0.000000
    25%          26.000000
    50%          35.000000
    75%          49.000000
    max          99.000000
    Name: driver_age_raw, dtype: float64




```python
age_bins = pd.cut(df['driver_age_raw'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], include_lowest=True)
ax = age_bins.value_counts(sort=False, normalize=True).mul(100).plot.bar(rot=0, color="b", figsize=(16,4), title="Age Breakdown (10-year Bins)", fontsize=14)
ax.axes.set_ylabel('%', fontsize=14), 
```




    (Text(0,0.5,'%'),)




![png](output_65_1.png)


274 records specify ages that are less than 15 years old, which appear to be typos and will be removed in the pre-processing stage.


```python
weird_ages_rows = df[df["driver_age_raw"] < 15]['driver_age_raw']
weird_ages = weird_ages_rows.value_counts(sort=False).plot.bar(figsize=(13, 4), table=True, title='Age Values Count')
xaxis = weird_ages.axes.get_xaxis()
xaxis.set_visible(False)
weird_ages.axes.set_ylabel('Count')
table = weird_ages.tables[0]
table.set_fontsize(14)
table.scale(1, 2)
```


![png](output_67_0.png)


### Algorithms and Techniques
<!--
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_
-->

<!--
To solve this problem, I will experiment with three algorithms:
1. The VotingClassifier from the scikit-learn library
1. GradientBoostingClassifier also  from the scikit-learn library
1. XGBoost from the Distributed(Deep) Machine Learning Community (DMLC) at https://github.com/dmlc/xgboost.
-->

Since this is a multi-class classification problem, I will experiment with the following  algorithms that scikit-learn lists as multi-class at http://scikit-learn.org/stable/modules/multiclass.html:

<table width=80%>
<thead>
<tr>
    <th style="text-align: left; width: 25%;">Classifier</th>
    <th style="text-align: left;">Description</th>
</tr>
</thead>
<tbody>
<tr>
    <td style="vertical-align:top; text-align: left; width: 25%;">**GaussianNB**</td>
    <td style="vertical-align:top; text-align: left;">
        <ul>
            <li>Implements the Gaussian Naive Bayes algorithm.</li>
            <li>Known to work well with large datasets such as this one.</li>
            <li>assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature</li>
        </ul>
    </td>
</tr>
<tr>
    <td style="vertical-align:top; text-align: left; width: 25%;">**RandomForestClassifier**</td>
    <td style="vertical-align:top; text-align: left;">
        <ul>
            <li>Meta estimator that fits a number of decision tree classifiers on dataset sub-samples, using averages to improve predictive accuracy and minimize over-fitting.</li>
            <li>Intrinsically suited for multiclass problems.</li>
            <li>Works well with a mixture of numerical and categorical features</li>
        </ul>
    </td>
</tr>
<tr>
    <td style="vertical-align:top; text-align: left; width: 25%;">**DecisionTreeClassifier**</td>
    <td style="vertical-align:top; text-align: left;">
        <ul>
            <li>Non-parametric learning method that predicts the value of a target variable by learning simple decision rules inferred from data features.</li>
            <li>Works well with regression and classification problems.</li>
            <li>Some regard as "set it and forget it" due to the minimal optimization needed</li>
        </ul>
    </td>
</tr>
<tr>
    <td style="vertical-align:top; text-align: left; width: 25%;">**GradientBoostingClassifier**</td>
    <td style="vertical-align:top; text-align: left;">
        <ul>
            <li>Builds an additive model in a forward stage-wise fashion, allowing for the optimization of random differentiable loss functions.</li>
            <li>Generally performs better than `random forest`.</li>
            <li>Provides a plethora of tuning parameters.</li>
        </ul>
    </td>
</tr>
</tbody>
</table>


For the most part, I used the default parameters for each classifier, with the exception of the **`n_jobs`**, **`verbose`**, and **`random_state`** parameters when available, which are specified as follows:

<table>
    <thead>
        <tr>
            <th></th>
            <th colspan=4 style="text-align: center;">Classifier</th>
        </tr>
        <tr>
            <th style="text-align: center;">Parameter</th>
            <th style="text-align: center;">GaussianNB</th>
            <th style="text-align: center;">RandomForestClassifier</th>
            <th style="text-align: center;">DecisionTreeClassifier</th>
            <th class="s2 softmerge">GradientBoostingClassifier</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="text-align: left; font-weight: bold;">bootstrap</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">True</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;"></td>
        </tr>
        <tr>
            <td style="text-align: left; font-weight: bold;">class_weight</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">None</td>
            <td style="text-align: center;">None</td>
            <td style="text-align: center;"></td>
        </tr>
        <tr>
            <td style="text-align: left; font-weight: bold;">criterion</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">gini</td>
            <td style="text-align: center;">gini</td>
            <td style="text-align: center;">friedman_mse</td>
        </tr>
        <tr>
            <td style="text-align: left; font-weight: bold;">init</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">None</td>
        </tr>
        <tr>
            <td style="text-align: left; font-weight: bold;">learning_rate</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">0.1</td>
        </tr>
        <tr>
            <td style="text-align: left; font-weight: bold;">loss</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">deviance</td>
        </tr>
        <tr>
            <td style="text-align: left; font-weight: bold;">max_depth</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">None</td>
            <td style="text-align: center;">None</td>
            <td style="text-align: center;">3</td>
        </tr>
        <tr>
            <td style="text-align: left; font-weight: bold;">max_features</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">auto</td>
            <td style="text-align: center;">None</td>
            <td style="text-align: center;">None</td>
        </tr>
        <tr>
            <td style="text-align: left; font-weight: bold;">max_leaf_nodes</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">None</td>
            <td style="text-align: center;">None</td>
            <td style="text-align: center;">None</td>
        </tr>
        <tr>
            <td style="text-align: left; font-weight: bold;">min_impurity_split</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">0.0000001</td>
            <td style="text-align: center;">0.0000001</td>
            <td style="text-align: center;">0.0000001</td>
        </tr>
        <tr>
            <td style="text-align: left; font-weight: bold;">min_samples_leaf</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">1</td>
            <td style="text-align: center;">1</td>
            <td style="text-align: center;">1</td>
        </tr>
        <tr>
            <td style="text-align: left; font-weight: bold;">min_samples_split</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">2</td>
            <td style="text-align: center;">2</td>
            <td style="text-align: center;">2</td>
        </tr>
        <tr>
            <td style="text-align: left; font-weight: bold;">min_weight_fraction_leaf</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">0</td>
            <td style="text-align: center;">0</td>
            <td style="text-align: center;">0</td>
        </tr>
        <tr>
            <td style="text-align: left; font-weight: bold;">n_estimators</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">10</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">100</td>
        </tr>
        <tr>
            <td style="text-align: left; font-weight: bold;">n_jobs</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center; font-weight: bold;">8</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;"></td>
        </tr>
        <tr>
            <td style="text-align: left; font-weight: bold;">oob_score</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">False</td>
            <td style="text-align: center;">best</td>
            <td style="text-align: center;"></td>
        </tr>
        <tr>
            <td style="text-align: left; font-weight: bold;">presort</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">False</td>
            <td style="text-align: center;">auto</td>
        </tr>
        <tr>
            <td style="text-align: left; font-weight: bold;">priors</td>
            <td style="text-align: center;">None</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;"></td>
        </tr>
        <tr>
            <td style="text-align: left; font-weight: bold;">random_state</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center; font-weight: bold;">0</td>
            <td style="text-align: center; font-weight: bold;">0</td>
            <td style="text-align: center; font-weight: bold;">0</td>
        </tr>
        <tr>
            <td style="text-align: left; font-weight: bold;">splitter</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">best</td>
            <td style="text-align: center;"></td>
        </tr>
        <tr>
            <td style="text-align: left; font-weight: bold;">subsample</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">1</td>
        </tr>
        <tr>
            <td style="text-align: left; font-weight: bold;">verbose</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center; font-weight: bold;">3</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center; font-weight: bold;">3</td>
        </tr>
        <tr>
            <td style="text-align: left; font-weight: bold;">warm_start</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">False</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">False</td>
        </tr>
    </tbody>
</table>

After experimenting with different configurations of the dataset, I will move forward with the best performing model and tune its hyperparameters.

<!--
I may need to manipulate my dataset until a desired accuracy score is reached.  At that point, I will select the most accurate classifier and conduct a grid search to tune its hyperparameters.

Finally, I hope to take advantage of my NVIDIA<sup>&reg;</sup> GeForce<sup>&reg;</sup> GTX 1080Ti GPU by using the **`XGBoost`** classifier.  Short for "e**X**treme **G**radient **Boost**ing," XGBoost is an optimized distributed gradient boosting library that implements machine learning algorithms under the Gradient Boosting framework and supports NVIDIA<sup>&reg;</sup> GPU's.  It has been used in many winning Kaggle competition solutions ([TODO: Add reference to https://github.com/dmlc/xgboost/tree/master/demo#machine-learning-challenge-winning-solutions).
-->


### Benchmark
<!--
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_
-->

As far as I know, there is no external benchmark to assess the accuracy of a traffic stop outcome prediction.  From my analysis of the data, generating a naive predictor that predicts the outcome to be most common value, "Ticket," should suffice as a benchmark.  "Ticket" comprises ~70% of all outcomes as described in the following table:

| Outcome         | Count   |    %   |
|:----------------|--------:|-------:|
|Arrest           |   7,312 |  2.33% |
|Summons          |  12,205 |  3.90% |
|**Ticket**       | **218,973** | **69.89%** |
|Verbal Warning	  |  47,753 | 15.24% |
|Written Warning  |  27,070 |  8.64% |
|                 | 313,313 |        |

To ensure that I am doing a fair comparison, I will run this model against the same test set that will be used as input to the **`score()`** functions of the classifiers specified above.



## III. Methodology
<!--
_(approx. 3-5 pages)_
-->

### Data Preprocessing

<!--
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_
-->

I performed the following preprocessing steps on the raw dataset to generate training and testing sets for classifier development:

1. Dropped the data columns that I believed were unnecessary, namely:
```
    county_fips
    driver_age
    driver_race_raw
    fine_grained_location
    id
    is_arrested
    location_raw
    officer_id
    police_department
    search_type
    search_type_raw
    state
```
1. Removed rows where **`stop_outcome`** and **`county_name`** had empty values, as well as rows which had **`driver_age_raw`** values below 15.
1. Calculated the median traffic stop time and used it to fill in the null values for the **`stop_time`** field.
1. Split **`stop_date`** and **`stop_time`** string values into **`month`**, **`day`**, **`hour`**, and **`min`** numerical value columns then dropped the **`stop_date`** and **`stop_time`** columns.
1. Normalized the violations data by combining values from the **`violation`** and **`violation_raw`** columns, merging similar values into a common value, then manually one-hot encoded the values into their own binary columns, which were then appended to the main dataframe as additional features.  I subsequently dropped the **`violation`** and **`violation_raw`** columns.
1. Converted **`search_conducted`** and **`contraband_found`** columns to binary values in-place.
1. Normalized **`driver_age`** values to values between 0.0 and 1.0 using **`sklearn.preprocessing.MinMaxScaler`** in-place.
1. Performed one-hot encoding on the following categorical value fields:  **`county_name`**, **`driver_gender`**, **`driver_race`**, **`stop_duration`**.




```python
# Append one-hot encoded violations
df = pd.concat([df, df_violations], axis=1)

# Remove records with age less than 15
df.drop(index=weird_ages_rows.index, inplace=True)

# Fill in empty **`stop_time`** with median value
populated = df[df.stop_time.notnull()]['stop_time'].sort_values()
median_stop_time = populated.iloc[populated.shape[0] // 2]
df['stop_time'].fillna(median_stop_time, inplace=True)
```


```python
# df.shape
```




    (313129, 28)




```python
# Categorize stop_time into time-of-day: "morning, afternoon, evening, small hours"
def day_period(time_str):
    hour = time_str.hour
    if hour >= 0 and hour < 6:
        return 'Small Hours'
    elif hour >= 6 and hour < 12:
        return 'Morning'
    elif hour >= 12 and hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'

df['day_period'] = pd.to_datetime(df['stop_time']).apply(day_period)

```


```python
# df.shape
```




    (313129, 29)




```python
# Categorize stop_date by season
def season(stop_date):
    month = datetime.strptime(stop_date, '%Y-%m-%d').month
    if month >= 3 and month < 6:
        return 'Spring'
    elif month >= 6 and month < 9:
        return 'Summer'
    elif month >= 9 and month < 12:
        return 'Fall'
    return 'Winter'

df['season'] = df['stop_date'].apply(season)
```


```python
# Transform driver_gender to binary
df['is_male'] = df['driver_gender'].apply(lambda x: 1 if x == 'M' else 0)
```


```python
# Experiment: See whether labelencoding location_raw improves performance
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['location_raw'] = le.fit_transform(df['location_raw'])
```


```python
# df.shape
```




    (313129, 31)




```python
# Drop columns no longer needed due to normalization
drop_cols = [
    'driver_gender',
#     'county_name',
#     'location_raw',
#     'officer_id',
    'stop_date',
    'stop_time',
    'violation_raw',
    'violation',
]

df.drop(drop_cols, axis=1, inplace=True)
```


```python
# df.shape
```




    (313129, 26)




```python
# Convert booleans to 0 and 1
df['search_conducted'] = df['search_conducted'].apply(lambda x: int(x))
df['contraband_found'] = df['contraband_found'].apply(lambda x: int(x))
```


```python
# Normalize driver_age
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() # default=(0, 1)
features_transformed = pd.DataFrame(data=df)
features_transformed['driver_age_raw'] = scaler.fit_transform(features_transformed['driver_age_raw'].reshape(-1, 1))
# features_transformed
```


```python
features_transformed.shape
```




    (313129, 26)




```python
features_transformed.drop_duplicates(inplace=True)
```


```python
# features_transformed.shape
```




    (277733, 26)




```python
# Prefix officer id numbers as dtype is object and get_dummies() creates some duplicate columns
# features_transformed['officer_id'] = le.fit_transform(features_transformed['officer_id'].apply(lambda x: 'no_{}'.format(x)))

```


```python
# One-hot encode categorical variables
cols_to_encode = [
#     'location_raw',
    'county_name',
    'driver_race',
#     'officer_id',
    'stop_duration',
    'day_period',
    'season',
]
final_features = pd.get_dummies(features_transformed, columns=cols_to_encode)
# final_features.columns.values
```




    array(['location_raw', 'driver_age_raw', 'search_conducted',
           'contraband_found', 'stop_outcome', 'violation_bad_stop',
           'violation_cell_phone', 'violation_display_of_plates',
           'violation_equipment', 'violation_license', 'violation_lights',
           'violation_moving_violation', 'violation_other',
           'violation_registration', 'violation_safe_movement',
           'violation_seatbelt', 'violation_speeding',
           'violation_suspended_license', 'violation_traffic_control_signal',
           'violation_window_tint', 'is_male', 'county_name_Fairfield County',
           'county_name_Hartford County', 'county_name_Litchfield County',
           'county_name_Middlesex County', 'county_name_New Haven County',
           'county_name_New London County', 'county_name_Tolland County',
           'county_name_Windham County', 'driver_race_Asian',
           'driver_race_Black', 'driver_race_Hispanic', 'driver_race_Other',
           'driver_race_White', 'stop_duration_1-15 min',
           'stop_duration_16-30 min', 'stop_duration_30+ min',
           'day_period_Afternoon', 'day_period_Evening', 'day_period_Morning',
           'day_period_Small Hours', 'season_Fall', 'season_Spring',
           'season_Summer', 'season_Winter'], dtype=object)




```python
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

non_oversampled = final_features.copy(deep=True)
non_oversampled_outcomes = non_oversampled.pop('stop_outcome')

# non_oversampled.to_pickle('./final_features-{}-non_oversampled.pkl'.format(timestamp))
# non_oversampled_outcomes.to_pickle('./labels-{}-non_oversampled.pkl'.format(timestamp))
# print('timestamp = {}'.format(timestamp))
```

### Implementation
<!--
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_
-->

<!--
#### _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
-->

<!--
For the **`sklearn.linear_model.SGDClassifier`**, untuned, this classifier produced an abysmal score of **`0.3173951596077613`** or **31.74% accuracy**.

In my preliminary testing with the following classifiers, I kept hitting a ceiling of 82% prediction accuracy no matter how many runs I set or parameters I tweaked.  So, I decided to add the **`location_raw`** data back into the training data, and the accuracy improved significantly.  The results that follow reflect the addition of this data column.

The **VotingClassifier** resulted in a score of **`0.9195076152722721`**, considerably more accurate than the SGDClassifier.  I was pleaantly surprised by this result, not expecting to achieve this level of accuracy with no parameter tuning.  Curious, I fitted each classifier independently, again with just the default parameters, and generated the following results:

| Classifier                  | Accuracy Score | Training Time  |
|:----------------------------|:--------------:|:--------------:|
| RandomForestClassifier      |      0.9140    |      9.55 secs |
| GaussianNB                  |      0.4560    |     0.959 secs |
| DecisionTreeClassifier      |      0.9102    |      4.43 secs |
| GradientBoostingClassifier  |      0.5133    |  7 min 13 secs |

Next, with minimal parameters set, **XGBoost** achieved **0.114328 error**, equating to **88.57% accuracy**. This result was achieved after 9,006 runs where the **`early_stopping_rounds`** parameter was set to 50.  To clarify, I set the number of rounds to 10,000, but the model would stop training once the current run generated an error that was no lower than the lowest eror over the previous 50 runs.  Please note that I am using [training] run and round synonymously.  An interesting issue I encounted was that the browser I was using for the Jupyter notebook to code this run kept crashing once the training session got past 3,300+ runs, so I had to move the code to a Python script and execute it from the command line.  Based on my recollection, the training time took around an hour and a half to achieve this score.
-->

As part of my implementation, I performed an 80/20 split of the preprocessed data into training and test sets, respectively.  Feeding the training set into the selected classifiers' **fit()** functions and subsequently calling the resulting fitted models' **score()** functions with the test set achieved the following accuracy scores:

| Algorithm | Accuracy Score |
|:----------|:--------------:|
| (**Benchmark**) | **0.6912** |
| GaussianNB | 0.643051 |
| DecisionTreeClassifier | 0.537262 |
| RandomForestClassifier | 0.685543 |
| **GradientBoostingClassifier** | **0.718311** |

Only the **GradientBoostingClassifier** performed better than the benchmark, and only by .027111 or 2.71%.  In an effort to improve performance, I tested different modifications to the data set.  To make this task easier to understand and track, I created the following flags and then tested different combinations between them in different stages:

<!--
| Flag | Description |
|:-----|:------------|
| include_location_raw | Add **`location_raw`** column to the data set |
| include_driver_race | Add **`driver_race`** column to the data set |
| label_encode_categoricals | Use sklearn.preprocessing.LabelEncoder to encode categorical column values. If False, use pandas.get_dummies() to one-hot encode each categorical column value into its own binary value column |
| oversample | Oversample the data to address data set imbalance |
| undersample | Undersample the data to address data set imbalance |

-->
<style>

</style>
<table width=80%>
<thead>
<tr>
    <th style="text-align: left; width: 25%;">Flag</th>
    <th style="text-align: left;">Data Transformation Description</th>
</tr>
</thead>
<tbody>
    <tr>
        <td style="text-align: left; width: 25%;">include_location_raw</td>
        <td style="text-align: left;">Add **`location_raw`** column to the data set</td>
    </tr>
    <tr>
        <td style="text-align: left; width: 25%;">include_driver_race</td>
        <td style="text-align: left;">Add **`driver_race`** column to the data set</td>
    </tr>
    <tr>
        <td style="text-align: left; width: 25%; vertical-align: top;">label_encode_categoricals</td>
        <td style="text-align: left;">Use sklearn.preprocessing.LabelEncoder to encode categorical column values. If False, use pandas.get_dummies() to one-hot encode each categorical column value into its own binary value column.</td>
    </tr>
    <tr>
        <td style="text-align: left; width: 25%;">oversample</td>
        <td style="text-align: left;">Oversample the data to address data set imbalance</td>
    </tr>
    <tr>
        <td style="text-align: left; width: 25%;">undersample</td>
        <td style="text-align: left;">Undersample the data to address data set imbalance</td>
    </tr>
</tbody>
</table>

The process of **oversampling** involved calculating a multiplier for each outcome value by dividing the number of rows of the outcome with the highest row count by each of the other outcome row counts then rounding down.  The non-largest outcome value rows were then replicated by their multipliers and appended to the main dataset, with the resulting dataset being shuffled prior to splitting into training and testing sets.

**Undersampling** involved removing a percentage of rows that had a **`stop_outcome`** value of "Ticket."  I experimented with different values between 1-50%, but the accuracy always decreased.  Stage 5 below with the `undersampling` flag checked reflects a 1% removal of "Ticket" rows.

The accuracy score results are shown below, where the scores reflect the increase or decrease in accuracy by changing one flag (Please note that Stage 1 reflects the initial implementation results):

<table border="0" cellspacing="0" cellpadding="0" class="ta1">
    <tr class="ro1">
        <td style="text-align:left;" class="ce1"> </td>
        <td colspan=7 style="text-align:center;font-weight: bold;" class="ce11">STAGES</td>
    </tr>
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; font-weight: bold;" class="ce2">
            <p>FLAGS</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce12">
            <p>1</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce12">
            <p>2</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce12">
            <p>3</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce12">
            <p>4</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce12">
            <p>5</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce12">
            <p>6</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce12">
            <p>7</p>
        </td>
    </tr>
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; " class="ce1">
            <p>include_location_raw</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
    </tr>
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; " class="ce3">
            <p>include_driver_race</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
    </tr>
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; " class="ce1">
            <p>label_encode_categoricals</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
    </tr>
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; " class="ce1">
            <p>oversample</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
    </tr>
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; " class="ce1">
            <p>undersample</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
    </tr>
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; " class="ce2">
            <p><b>CLASSIFIER SCORES</b></p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
    </tr>
    <!--
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; " class="ce2">
            <p>SGDClassifier</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.694371</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.707311</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.370454</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.704465</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.694838</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.703134</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.69509</p>
        </td>
    </tr>
    -->
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; " class="ce1">
            <p>GaussianNB</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.643051</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.202978</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.686507</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.685147</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.686523</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.207804</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.685847</p>
        </td>
    </tr>
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; " class="ce1">
            <p>DecisionTreeClassifier</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.537262</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.571481</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.559568</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.567211</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.553123</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.568991</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.556581</p>
        </td>
    </tr>
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; " class="ce1">
            <p>RandomForestClassifier</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.685543</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce14">
            <p>0.70189</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.696357</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.681789</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.693015</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.702271</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.69653</p>
        </td>
    </tr>
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; " class="ce1">
            <p>GradientBoostingClassifier</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce14">
            <p>0.718311</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce14">
            <p>0.724262</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.721336</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.711789</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.715451</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>**0.724468**</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.721238</p>
        </td>
    </tr>
    <!--
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; " class="ce1">
            <p>GradientBoostingClassifier (Tuned)</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce14">
            <p>0.717432</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce14">
            <p>0.7285</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.728404</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.706416</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.70536</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.72885</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.728466</p>
        </td>
    </tr>
    -->
 </table>


A few notable observations stand out:

1. The **GradientBoostingClassifier** consistently outperforms the other classifiers for this dataset.
1. Adding the **`1ocation_raw`** field back into the dataset improves performance.
1. One-hot encoding is preferable over using the `LabelEncoder()` for this use case.
1. Over/under-sampling decreases accuracy performance for this problem.  Not shown here are results from testing several values for outcome multipliers to balance the dataset.
1. Dropping **`driver_race`** appears to increase performance slightly, so it may be an unnecessary data column for this classifier.
1. The Stage 6 combination of flags and dataset had the best performance and should be used for hyperparameter tuning.


<!--
#### _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
-->

<!--
- XGBoost's API enables you to create and train a model through a method or a class:  xgb.train or XGBClassifier.  The latter can be used with **`sklearn.model_selection.GridSearchCV`** to perform a grid search.  Unfortunately, I was not able to get it to work with my GPU, so I implemented a manual grid search in which I experimented with various values for 3 parameter variables [TODO: list the variables] until minimum error was achieved using sample sets of 10,000 rows.
-->

<!--
#### Was there any part of the coding process (e.g., writing complicated functions) that should be documented?
-->

### Refinement
<!--
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_
-->

<!--
#### _Has an initial solution been found and clearly reported?_
-->

The results reveal the **GradientBoostingClassifier** to be the best classifier for this project, using the dataset from the stage 6 configuration.  To optimize this model, I used **`sklearn.model_selection.RandomizedSearchCV`** to perform a randomized search of select parameters for the GradientBoostingClassifier class.  RandomizedSearchCV is similar to GridSearchCV which performs an exhaustive search over specified parameter values for an estimator, using cross-validation.  RandomizedSearchCV differs slightly from GridSearchCV in that, rather than searching over all specified parameter values, it performs a randomized search where each setting is sampled from a distribution over possible parameter values.  As a result, RandomizedSearchCV is able to complete a search for optimal parameters values in less time than GridSearchCV with slightly less accuracy.

I chose the following list of values to search for the GradientBoostingClassifier class:

* criterion = [ 'friedman_mse', 'mse', 'mae' ]
* learning_rate = [ 0.09, 0.1 ]
* max_depth = [ 5, 6, 7 ]
* max_features = [ None, 219<feature count> ]
* subsample = [ 0.85, 0.9, 0.85 ]

The RandomizedSearchCV class was instantiated with a dictionary of these parameters as the **params_distribution** input parameter along with the following input parameters:

* scoring='accuracy'
* cv=5
* verbose=3
* n_iter=1

**`cv`** is the number of cross-folds to use for validation, and `n_iter` specifies the number of iterations.   The model instance was then called with the `.fit()` method with the training dataset as its input.  It is important to note that `n_iter` provides a mechanism to limit the duration for a search.  The higher the number the more samplings RandomizedSearchCV performs, but the longer it will take to finish.  In my case, one iteration took 38 minutes to complete.  

<!--
#### _Is the process of improvement clearly documented, such as what techniques were used?_
-->
Once RandomizedSearchCV completed its search, the optimal parameter values that were returned for the GradientBoostingClassifier class were the following:

* criterion='mse'
* learning_rate=0.1
* max_depth=5
* max_features=219
* subsample=0.85

<!--
As mentioned earlier, I added the **`location_raw`** column back into the training data after my preliminary training runs reached an accuracy limit.  To add this data column back, I tried one-hot encoding it, but doing so added 184 feature columns, one for each unique value.  This resulted in out-of-memory errors while training a model and was not a feasible option.  To overcome this, I used **`sklearn.preprocessing.LabelEncoder`** which transforms each unique value into an integer value which could be transformed back into its string label value later on.  The added benefit is that only one feature column slot is used.

Next, I focused on tuning XGBoost's parameters to leverage my GPU and perform more experiments with different parameters in a shorter period of time than with just a CPU.  I attempted to implement a grid search using **`sklearn.model_selection.GridSearchCV`**.  Unfortunately, I was not able to leverage my GPU to do this using XGBoost's **`XGBClassifier`** class.  Even though there were no errors while fitting this model, I monitored GPU activity with NVIDIA's **`nvidia-smi`** utility and saw that the GPU was not being used.

Instead, to leverage the GPU, I mimicked a grid search by using the **`xgb.train()`** function with the **`tree_method`** parameter set to "gpu_hist", iterating through different values for various parameters through nested Python for-loops and a data sample of 10,000 rows.  The optimal set of parameters I found for XGBoost were the following:

| Parameter | Value |
|-----------|:-----:|
| objective | multi:softmax |
| learning_rate | 0.0983 |
| max_depth | 19 |
| silent | 0 |
| nthread | 8 |
| tree_method | gpu_hist |
| subsample | 0.9 |
| num_class | 44 |

-->
<!--
#### _Are intermediate and final solutions clearly reported as the process is improved?_
-->

<!--
I set the **`num_boost_rounds`** to "10000" and **`early_stopping_rounds`** to "50", so that the classifier would attempt no more than 10,000 runs and would stop fitting once the error was higher than the lowest error over the previous 50 runs.  With these parameters, XGBoost achieved an error of **0.061229918631337366** or **93.88% accuracy** over 597 runs in 1 hour and 3 minutes.  The result is somewhat disappointing as it only represents a **2.48% improvement** over the best CPU-bound classifier, RandomForestClassifier, which achieved 91.4% accuracy in 9.55 seconds.


Results after adding back **`location_raw`**:

| Classifier                  | Accuracy Score | Training Time  |
|:----------------------------|:--------------:|:--------------:|
| RandomForestClassifier      |      0.9264    |     5.10 secs  |
| GaussianNB                  |      0.3754    |     1.54 secs  |
| DecisionTreeClassifier      |      0.9190    |     1.13 secs  |
| GradientBoostingClassifier  |      0.    |  7 min 13 secs |


Using RandomizedSearchCV with RandomForestClassifier and the following parameters:
    'n_estimators': [10, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 19],
    'max_features': [None, 'sqrt', 'log2'],
    'n_jobs': [8],
    'random_state': [0],
    'verbose': [3],

...determined that optimal params are:
verbose=3
random_state=0
n_jobs=8
n_estimators=<n_estimators>
max_features="sqrt"
max_depth=None
criterion="entropy"

Did further tuning with n_estimators and got following performance:

    n_estimators=50   # 0.9301481326935114   18.5 s
    n_estimators=100  # 0.9306175672856248   35.7 s
    n_estimators=200  # 0.930732317963697
    n_estimators=450  # 0.9310922178176507
    n_estimators=675  # 0.9312121844356352  # 3 mins

    n_estimators=1 0   # 0.9288092760517318     4.61s
    n_estimators=100   # 0.9321787820226947       39s
    n_estimators=450   # 0.9322828402953273    2m 26s
    n_estimators=675   # 0.9322283335810911    8m  2s
-->


## IV. Results
<!--
_(approx. 2-3 pages)_
-->

### Model Evaluation and Validation

<!--
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_
-->


```python
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


X_train = pickle.load(open('./data/stage6-train.pkl', 'rb'))
y_train = X_train.pop('stop_outcome')
X_test = pickle.load(open('./data/stage6-test.pkl', 'rb'))
y_test = X_test.pop('stop_outcome')

gbc = GradientBoostingClassifier(criterion='mse', learning_rate=0.1, max_depth=5, max_features=219,
                                 subsample=0.85, verbose=3, random_state=0)
gbc.fit(X_train, y_train)
print('{}'.format(gbc.score(X_test, y_test)))
```

For the final model, I instantiated GradientBoostingClassifier with these parameters, fitted the model, and got an accuracy score of **0.7288** for the stage 6 test set -- an improvement of 0.0044 or 0.44% over the non-tuned GradientBoostingClassifier instance.  To compare the results with the other stages, I retrained and scored the GradientBoostingClassifier class with the optimal parameters using the dataset configurations for each of the other stages, and the results are as follows:

<table border="0" cellspacing="0" cellpadding="0" class="ta1">
    <tr class="ro1">
        <td style="text-align:left;" class="ce1"> </td>
        <td colspan=7 style="text-align:center;font-weight: bold;" class="ce11">STAGES</td>
    </tr>
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; font-weight: bold;" class="ce2">
            <p>FLAGS</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce12">
            <p>1</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce12">
            <p>2</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce12">
            <p>3</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce12">
            <p>4</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce12">
            <p>5</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce12">
            <p>6</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce12">
            <p>7</p>
        </td>
    </tr>
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; " class="ce1">
            <p>include_location_raw</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
    </tr>
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; " class="ce3">
            <p>include_driver_race</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
    </tr>
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; " class="ce1">
            <p>label_encode_categoricals</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
    </tr>
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; " class="ce1">
            <p>oversample</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
    </tr>
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; " class="ce1">
            <p>undersample</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13">
            <p>&#10004;</p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
    </tr>
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; " class="ce2">
            <p><b>CLASSIFIER SCORES</b></p>
        </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
        <td style="text-align:left;width:73.76pt; " class="ce13"> </td>
    </tr>
    <!--
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; " class="ce2">
            <p>SGDClassifier</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.694371</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.707311</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.370454</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.704465</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.694838</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.703134</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.69509</p>
        </td>
    </tr>
    -->
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; " class="ce1">
            <p>GaussianNB</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.643051</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.202978</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.686507</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.685147</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.686523</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.207804</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.685847</p>
        </td>
    </tr>
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; " class="ce1">
            <p>DecisionTreeClassifier</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.537262</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.571481</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.559568</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.567211</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.553123</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.568991</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.556581</p>
        </td>
    </tr>
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; " class="ce1">
            <p>RandomForestClassifier</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.685543</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce14">
            <p>0.70189</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.696357</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.681789</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.693015</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.702271</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.69653</p>
        </td>
    </tr>
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; " class="ce1">
            <p>GradientBoostingClassifier</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce14">
            <p>0.718311</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce14">
            <p>0.724262</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.721336</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.711789</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.715451</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.724468</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.721238</p>
        </td>
    </tr>
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; font-weight: bold;" class="ce1">
            <p>GradientBoostingClassifier (Tuned)</p>
        </td>
        <td style="text-align:right; width:73.76pt; font-weight: bold;" class="ce14">
            <p>0.71882</p>
        </td>
        <td style="text-align:right; width:73.76pt; font-weight: bold;" class="ce14">
            <p>0.7280679</p>
        </td>
        <td style="text-align:right; width:73.76pt; font-weight: bold;" class="ce13">
            <p>0.72805</p>
        </td>
        <td style="text-align:right; width:73.76pt; font-weight: bold;" class="ce13">
            <p>0.709054</p>
        </td>
        <td style="text-align:right; width:73.76pt; font-weight: bold;" class="ce13">
            <p>0.70456</p>
        </td>
        <td style="text-align:right; width:73.76pt; font-weight: bold;" class="ce13">
            <p>0.728882</p>
        </td>
        <td style="text-align:right; width:73.76pt; font-weight: bold;" class="ce13">
            <p>0.72781</p>
        </td>
    </tr>
    <!--
    <tr class="ro1">
        <td style="text-align:left;width:213.45pt; " class="ce1">
            <p>GradientBoostingClassifier (Tuned)</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce14">
            <p>0.717432</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce14">
            <p>0.7285</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.728404</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.706416</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.70536</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.72885</p>
        </td>
        <td style="text-align:right; width:73.76pt; " class="ce13">
            <p>0.728466</p>
        </td>
    </tr>
    -->
 </table>


<!--
#### _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
-->

As the table above shows, the optimal model has the following traits:

* An instance of the GradientBoostingClassifier class with the following parameters:
  1. criterion='mse'
  1. learning_rate=0.1
  1. max_depth=5
  1. max_features=219
  1. subsample=0.85
* Trained with data that went through the preprocessing steps detailed above.
* Trained with the **`location_raw`** data column.
* Trained without the **`driver_age`** data column.
* Categorical data column values were one-hot encoded.

<!--
#### _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
-->

<!--
The final model I chose was the XGBoost classifier with the parameters specified in the previous section.  Even though it took **`RandomForestClassifier`** and **`DecisionTreeClassifier`** mere seconds to achieve accuracy scores there were only ~2.5% lower, XGBoost still had the highest accuracy.

To ensure that I could be confident of my classifier, I sliced 5% of the dataset, before feeding the dataset through **`sklearn.model_selection.train_test_split`**, for the purpose of adding a second testing set that was not involved in the training process at all.  Testing model predictions with this dataset was consistent, usually within 1/10000th of the error from the test set generated by **`train_test_split`**.
-->

<!--
#### _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_

#### _Can results found from the model be trusted?_
-->


```python
# Plot training and testing results

# logfile = 'runs/201805111748-oversampled-647-run.log'

# error_results = {
#     'train': [],
#     'test': [],
# }

# with open(logfile, 'r') as f:
#     for line in f:
#         tokens = line.strip().split('\t')
#         error_results['train'].append(float(tokens[1].split(':')[1]))
#         error_results['test'].append(float(tokens[2].split(':')[1]))

# df_errors = pd.DataFrame(error_results)
# df_accuracy = df_errors.copy()
# df_accuracy_processed = df_accuracy.mul(-1).add(1)

# learning_curve = df_accuracy_processed.plot.line(title='Learning curve', figsize=(12,7))
# learning_curve.set_xlabel('Number of runs')
# learning_curve.set_ylabel('Accuracy')

```

### Justification

<!--
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_
-->

<!--
#### _Are the final results found stronger than the benchmark result reported earlier?_
-->

The final model predicts traffic stop outcomes better than the benchmark model, but not by much.  Of the actual outcomes for traffic stops in the test set, it accurately predicts their outcomes 72.89% of the time, just 3.77% better than the benchmark model which can do so 69.12% of the time.

This is encapsulated in the formula to calculate recall:

$$
\frac{T_p}{T_p+F_n}
$$

where ${T_p}$ = True Positives and ${F_n}$ = False Negatives

A confusion matrix with recall scores is provided in the next section.

<!--
#### _Have you thoroughly analyzed and discussed the final solution?_

#### _Is the final solution significant enough to have solved the problem?_

I had hoped to achieve accuracy greater than 90%, but after extensive testing and experimentation on the dataset and with different algorithms (including some not detailed in this report), I do not believe it would be achievable with the features available in this dataset.
-->


## V. Conclusion
<!--
_(approx. 1-2 pages)_
-->

### Free-Form Visualization
<!--
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_
-->

<!--
#### Have you visualized a relevant or important quality about the problem, dataset, input data, or results?
-->

Here is a horizontal barchart of the sorted feature importances from the tuned **`GradientBoostingClassifier`**:


```python
# TODO: Confirm F-score is the proper y_label

gbc_tuned_df = pd.DataFrame(data=gbc.feature_importances_[:15], index=X_train.columns.values[:15])
gbc_tuned_plot = gbc_tuned_df.sort_values(by=0).plot.barh(figsize=(15, 10), fontsize=14)
gbc_tuned_plot.axes.legend().set_visible(False)
gbc_tuned_plot.set_title('Feature Importance Ranked', fontsize=14)
gbc_tuned_plot.axes.set_xlabel('Relative ranking score that sums to 1.0', fontsize=14)
gbc_tuned_plot.axes.set_ylabel('Features', fontsize=14)
```




    Text(0,0.5,'Features')




![png](output_112_1.png)


The chart indicates that the **hour** of the day and the **age** of the driver are the most important features used in predicting the outcome of a traffic stop in our dataset.  However, this is likely to be an inaccurate reflection of feature importance, as the top 5 ranked features are the only numerical data columns in the training set, and the remaining columns are binary value columns from features that were one-hot encoded from categorical data columns.  In the stages that used **`LabelEncoder()`** to enumerate categorical values in-place, the top ranked feature was **`location_raw`** which suggests that the **city** in which a driver is stopped has a strong influence on the outcome.

Perhaps, we could gain better insight by analyzing the confusion matrix for our test set.  The following plot illustrates the accuracy (recall) scores of this classifier, with darker cells signifying higher recall:


```python
classes = ['Arrest', 'Summons', 'Ticket', 'Verbal Warning', 'Written Warning']
# cm = pd.DataFrame(data=confusion_matrix(y_test, gbc.predict(X_test)), columns=classes, index=classes)
cm = confusion_matrix(y_test, gbc.predict(X_test))
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
np.set_printoptions(precision=4)
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Accuracy Scores)', fontsize=14)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45, fontsize=14)
plt.yticks(tick_marks, classes, fontsize=14)

fmt = '.4f'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             fontsize=14,
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('Actual Outcome', fontsize=14)
plt.xlabel('Predicted Outcome', fontsize=14)
plt.show()
```


![png](output_115_0.png)



```python
#precision_recall_fscore_support(y_test, gbc.predict(X_test), average=None)

```

The only outcome that this classifier is able to predict accurately is "Ticket" with a recall score of 0.9558, which means that out of all of the traffic stops that actually resulted in a "Ticket," this classifier correctly predicted an outcome to be a "Ticket" 95.58% of the time.  The next most accurate outcome is "Arrest" at 35.80% recall.  The least accurate is "Written Warning" at 12.25%.  Intuitively, this may make sense, as the criteria as to what leads to a "Ticket" or a "Written Warning" is likely to be the same, with the probable differing determinant being the officer's discretion.


<!--
#### Is the visualization thoroughly analyzed and discussed?

#### If a plot is provided, are the axes, title, and datum clearly defined?
-->

### Reflection
<!--
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_
-->

<!--
#### _Have you thoroughly summarized the entire process you used for this project?_
-->

For this project, I took traffic stop data compiled from the Stanford Open Policing Project(SOPP) and created a model to predict their outcomes.  After transforming the data into a form suitable to use as input to a slate of different classification algorithms, I trained the models and compared their predictive capabilities, picked the most accurate, and tuned its hyperparameters to optimize performance.  The optimized model outperforms the benchmark model by only a modest margin.  

<!--
#### _Were there any interesting aspects of the project?_
-->

I had hoped to achieve accuracy greater than 90%, but after extensive testing and experimentation on the dataset and different algorithms (including some not detailed in this report), I do not believe it is achievable with the given dataset.  The number and quality of features may be insufficient.  It may be possible that data for other states might provide better feature signals, but the nature of the problem may be a problem itself.  The factors that influence the outcome of a traffic stop may depend on more than the features that were available in this dataset.


<!--
There were many interesting aspects of the project.  I experienced many moments of discovery as I experimented with data structures and tuned various hyperparameters.  For me, the most interesting aspect was seeing how big of an impact adding the **`location_raw`** field had on accuracy performance.  Prior to that, I kept hitting a ceiling of 82% accuracy no matter which algorithm I used or which hyperparameters I tuned.

**`location_raw`** had the second highest feature importance.  The field with the highest importance was **`driver_age`**.  It makes sense as it is the only truly numerical field in the dataset, even if it is more characteristic of a discrete label.
-->

<!--
#### _Were there any difficult aspects of the project?_
-->

<!--
Because XGBoost has been used in so many winning Kaggle competition solutions, I expected there to be more information available in forums and blog posts to resolve issues.  For some of the problems I encountered, the information was surprisingly sparse.

Of course, training time is always an issue.  I used my own hardware rig for training, and even with an Intel i7-7700K CPU (quad-core with 8 threads) with 16 GB RAM and an NVIDIA GTX 1080Ti GPU, it felt long and time-consuming for many experiments.  I am sure it would have felt even more tedious and prolonged had I only been able to use a CPU-only system.
-->

I found a few aspects of this project to be challenging:

1. Insufficient memory made it impossible to include the **`officer_id`** field.  With 2,105 unique values, this field could not be one-hot encoded nor encoded with the **`sklearn.preprocessing.LabelEncoder`**.
1. At one point, I made the mistake of extracting the test set after oversampling, which set me down an incorrect path for longer than it should have.  Doing so gave extremely high accuracy which was misleading.
1. I had to resist the urge to continually try as many algorithms as possible to improve accuracy. At some point, I came to the realization that I could not improve accuracy any further.

<!--
#### _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_
-->

Finally, the model is unlikely to be practical as a predictor of traffic stop outcomes.  It does marginally better than just guessing that the outcome will be "Ticket."  The data shows that most outcomes are speeding tickets.  Arrests are rare, at least in the state of Connecticut.  


### Improvement
<!--
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_
-->

<!--
#### _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
-->

It may be possible to increase accuracy by finding ways to incorporate the **`officer_id`** field.  I suspect that increased hardware RAM might make it possible, but there is also the risk that any model created with such granular data might be more prone to overfitting.  But, incorporating this field might reveal and incorporate biases of certain officers into our model which might improve accuracy.  Other than that, I am unaware of any other ways to improve this model's accuracy.  Perhaps, if SOPP collects additional features in the future, better accuracy can be achieved.

<!--
#### _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
-->

Although I did not detail my experiments with GPU-leveraging algorithms, I did try out **`XGBoost`** with my GPU, but found that accuracy was no better than **`GradientBoostingClassifier`**.  Further, some believe that **`Light GBM`** has slightly better accuracy and is more performant than XGBoost as it splits its trees leaf-wise, instead of level-wise like XGBoost does, but I did not experiment with it.  I do not have much hope that it would do much better than the results I achieved with **`GradientBoostingClassifier`**.

<!--
As I experimented and researched solutions for problems I encountered while developing my model, I came across another boosting framework called, Light GBM, that, at least by one benchmark test by Analytics Vidya, may be slightly more accurate and significantly more performant (see "Which algorithm takes the crown: Light GBM vs XGBOOST?" at https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/).  Light GBM splits its trees leaf-wise, instead of level-wise as XGBoost does.
-->

<!--
#### - _If you used your final solution as the new benchmark, do you think an even better solution exists?_
-->

<!--
My benchmark of 95% accuracy is quite high and may be difficult to improve.  Of course, there is always the possibility thtat an even better solution exists.  Further hyperparameter tuning might achieve this.  Plus, experiments with Light GBM or a convolutional neural network might also improve performance.
-->

-----------

<!--
**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
-->

## References



[1] Cheryl Phillips, Journalism Professor at Stanford University, interview. Stanford Open Policing Project (July 17, 2007).
Retrieved from https://youtu.be/iwOWcuFjNfw?t=4s.

[2] Stanford Open Policing Project (https://openpolicing.stanford.edu/)

[3] scikit-learn web pag for sklearn.metrics.accuracy_score:  http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html.

[4] Wikipedia entry for Jaccard index:  https://en.wikipedia.org/wiki/Jaccard_index]

[5] 2010 Census: [Connecticut] Apportionment Data Map:  https://www.census.gov/2010census/popmap/ipmtext.php?fl=09




```python

```
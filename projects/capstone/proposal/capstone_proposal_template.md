# Machine Learning Engineer Nanodegree
## Capstone Proposal
Patrick Poon
April 18, 2018

## Proposal
<!-- _(approx. 2-3 pages)_ -->

### Domain Background
<!-- _(approx. 1-2 paragraphs)_ -->

> Most people...the interaction that they're going to have with a police officer is because [...] they're stopped for speeding.  Or, forgetting to turn their blinker off.[1]
> 
>-- Cheryl Phillips, Journalism Professor at Stanford University

On a typical day in the United States, police officers make more than 50,000 traffic stops.[2]  In recent years, there have been numerous incidents that have made national headlines that involve an officer shooting and, in some cases, killing the driver or an occupant.  Many cite racial biases against Blacks and Hispanics for the disproportionate amount of such incidents for these communities.  Here are some relevant articles:

- Was the Sandra Bland traffic stop legal -- and fair? (https://www.cnn.com/2015/07/23/opinions/cevallos-sandra-bland-traffic-stop/index.html)
- Philando Castile shooting: Dashcam video shows rapid event (https://www.cnn.com/2017/06/20/us/philando-castile-shooting-dashcam/index.html)

This Capstone project will not attempt to prove or disprove this controversial topic, and will attempt to  avoid making any controversial statements on either side of the conversation.

<!--
In this section, provide brief details on the background information of the domain from which the project is proposed. Historical information relevant to the project should be included. It should be clear how or why a problem in the domain can or should be solved. Related academic research should be appropriately cited in this section, including why that research is relevant. Additionally, a discussion of your personal motivation for investigating a particular problem in the domain is encouraged but not required.
-->

### Problem Statement
<!-- _(approx. 1 paragraph)_ -->

Instead, this project aims to create a multiclass classifier that takes various discrete traffic stop situational values to predict the outcome of a traffic stop, specifically in the state of Connecticut.  Given a driver's age, gender, race, violation, and the county where the traffic stop occurs, can we reliably predict whether the traffic stop will result in a verbal/written warning, a ticket, a summons to appear in court, or an arrest?

<!--
In this section, clearly describe the problem that is to be solved. The problem described should be well defined and should have at least one relevant potential solution. Additionally, describe the problem thoroughly such that it is clear that the problem is quantifiable (the problem can be expressed in mathematical or logical terms) , measurable (the problem can be measured by some metric and clearly observed), and replicable (the problem can be reproduced and occurs more than once).
-->

### Datasets and Inputs
<!-- _(approx. 2-3 paragraphs)_ -->

The dataset I will be using comes from the The Stanford Open Policing Project (SOPP), which gathers, analyzes, and releases records from millions of traffic stops by law enforcement agencies across the country. The organization aims to help researchers, journalists, and policymakers investigate and improve interactions between police and the public.

For this project, I will be using a small subset of SOPP's data collections[3], specifically for the state of Connecticut [4].  Most American states have their own policies and practices for collecting traffic stop data, so there is no standard policy common across all states.  In fact, SOPP has collected data for only 31 out of 50 states, that consumes 21G of disk space at the time of this project.  Analyzing each of the states' data, Connecticut had the cleanest and most consistent set of records compared to the others.  It has a total of 318,669 records of traffic stops made between 2013 to 2015 with the following fields:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 318669 entries, 0 to 318668
Data columns (total 24 columns):
id                       318669 non-null object
state                    318669 non-null object
stop_date                318669 non-null object
stop_time                318447 non-null object
location_raw             318628 non-null object
county_name              318627 non-null object
county_fips              318627 non-null float64
fine_grained_location    317006 non-null object
police_department        318669 non-null object
driver_gender            318669 non-null object
driver_age_raw           318669 non-null int64
driver_age               318395 non-null float64
driver_race_raw          318669 non-null object
driver_race              318669 non-null object
violation_raw            318669 non-null object
violation                318669 non-null object
search_conducted         318669 non-null bool
search_type_raw          4846 non-null object
search_type              4846 non-null object
contraband_found         318669 non-null bool
stop_outcome             313313 non-null object
is_arrested              313313 non-null object
officer_id               318669 non-null object
stop_duration            318669 non-null object
dtypes: bool(2), float64(2), int64(1), object(19)
memory usage: 54.1+ MB
```

318,669 records should be sufficient for training and testing purposes.  I will extract the `stop_outcome` column to use as output labels and ground truth.


<!--
In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.
-->

### Solution Statement
<!-- _(approx. 1 paragraph)_ -->

To create a multiclass classifier to predict the outcome of a traffic stop, I plan to evaluate and experiment with various supervised learning models that are available in the scikit-learn Python library.  Under consideration are the following:
* Gaussian Naive Bayes (GaussianNB)
* Decision Trees
* Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
* K-Nearest Neighbors (KNeighbors)
* Stochastic Gradient Descent Classifier (SGDC)
* Support Vector Machines (SVM)
* Logistic Regression


<!--
In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).
-->

### Benchmark Model
<!-- _(approximately 1-2 paragraphs)_ -->

As far as I know, there is no external benchmark to compare the results to, so I propose generating a naive predictor to set the benchmark.  The most common `stop_outcome` value in this dataset is 'Ticket' which comprises 70% of all stop outcomes as described in the following table:

| Outcome | Count | % |
|:--------|------:|--:|
|Arrest           |   7,312 |  2.33% |
|Summons          |  12,205 |  3.90% |
|Ticket           | 218,973 | 69.89% |
|Verbal Warning	  |  47,753 | 15.24% |
|Written Warning  |  27,070 |  8.64% |
|                 | 313,313 |        |

As a base model without any intelligence, predicting every traffic stop will result in a 'Ticket' will generate an accuracy score of around 0.70 and serve as our benchmark model.

<!--
In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.
-->

### Evaluation Metrics
<!-- _(approx. 1-2 paragraphs)_ -->

Initially, I had planned on proposing to use the F-beta score for my evaluation metric, but after some tests, I encountered the following error when attempting to do so:
```
ValueError: Sample-based precision, recall, fscore is not meaningful outside multilabel classification. See the accuracy_score instead.
```
Instead, I shall use **accuracy classification score** as my evaluation metric.  According to the sklearn page for the `accuracy_score` function[5], in the context of multiclass classification, the function is equivalent to the `jaccard_similarity_score` which calculates the Jaccard index, also known as "Intersection over Union," as illustrated in the following formula:
<center>
![Intersection over Union formula](./images/IoU.svg)
</center>

<!--
In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).
-->

### Project Design
<!-- _(approx. 1 page)_ -->

To start, I will **explore the data**, and see if I can extract any insights or build some intuition about it.  I will graph different columns to determine which may be unbalanced.

Next, I will **prepare the data** by performing a number of operations, such as the following:

1. **Remove rows that have no `stop_outcome` value**:  Since my main objective is to predict the outcome of a traffic stop, null values for this field render the associated record unusable.  It would not make sense to attempt to fill the empty values with a median or average value.

1. **Remove empty columns**:  There are a few columns that have no values at all, so it makes sense to drop these entirely.

1. **Handle columns with some missing values**:  Some may need to be dropped, while others will be filled in with median or average values as appropriate.
1. **Recategorize very granular values to broader categorical values**:  For instance, day values are very specific, and may not lend themselves to provide much insight.  However, they may be able to provide some insight if they were categorized by annual seasons, like Spring, Summer, Fall, and Winter.  Similarly, the `stop_time` values are very specific, and would benefit from being converted to time of day, like morning, afternoon, evening, and small hours.
1. **Convert binary values to boolean values**: The `driver_gender` column has values of **M** and **F**.  It would be better to convert these to boolean and rename the column to `is_male`.
1. **Clean up messy column data**: The `violation_raw` and `violation` columns seemed to have repetitive and inconsistent data entry issues.  For example, two different violations can be phrased differently yet mean the same thing.  I will need to settle on a standard value for these duplicate values.  I will also need to manually perform one-hot encoding for each class value in the hopes that violations can provide predictive power to my classifier.
1. **Normalize numerical data**: For this project, the only column that might be considered numerical is `driver_age`, even though it is more characteristic of a discrete value.
1. **Extract `stop_outcome` column for classification labels and ground truth**:  These are the output values for the predictions.

After these operations are performed, I will **preprocess the data**, by one-hot encoding columns that have stable categorical values.  Once completed, I will move on to **shuffling and splitting the data** into training and testing sets, which I will use as inputs to **evaluate 3-5 supervised learning models** as I specified in the **Datasets and Inputs** section above.

Finally, I will choose the **best performing model** and **tune its hyperparameters** by doing a grid search, then determine whether certain features can be dropped by performing feature selection, which involves determining which features have the highest prediction power.
 
<!--
In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.
-->

### References

[1] Cheryl Phillips, Journalism Professor at Stanford University, interview.  Stanford Open Policing Project  (July 17, 2007). Retrieved from https://youtu.be/iwOWcuFjNfw?t=4s.

[2] Stanford Open Policing Project (https://openpolicing.stanford.edu/)

[3] Stanford Open Policing Project: About the Data (https://openpolicing.stanford.edu/data/)

[4] Stanford Open Policing Project: Data Download for Connecticut (https://stacks.stanford.edu/file/druid:py883nd2578/CT-clean.csv.gz)

[5] sklearn.metrics.accuracy_score (http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)

-----------
<!--
**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
-->
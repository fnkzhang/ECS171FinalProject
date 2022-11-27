# ECS171FinalProjectGroup17
Group 17

Dataset PDF: https://cseweb.ucsd.edu/classes/sp15/cse190-c/reports/sp15/048.pdf

Link to Colab NTBK: https://colab.research.google.com/drive/1D4CDiRiPCTuvCX6Iq1XjwSHPKIyKh6p_?usp=sharing 

## Abstract

- Members: George Luu, Tiching Kao, Frank Zhang, Teresa Lin, Pallavi Gupta, Nicolas Conrad
- Link: https://archive.ics.uci.edu/ml/datasets/Adult
- Description: Data set contains age, workclass, income, education, marital status, occupation, relationship, race, sex, capital gain and loss, workhours per week, native country, and <50k or >50k income


_'A person’s earned income is correlated with numerous factors about the individual. This dataset provides information on whether a person earns more than 50k annually as well as other features about the person including the country they came from, their highest education level, age, occupation, etc. Neural networks are supervised learning models that accept multiple attributes in order to try to predict an outcome. We will train a neural network model to predict an individual’s financial status based on factors such as age, workclass, education, etc. Attributes will be weighed. Thus, our goal with this supervised learning model is to predict whether a person earns above or below 50k annually using the person’s other features. This is useful because it allows us to analyze what are the attributes that determine who earns more than 50,000 dollars, and thus elevate poorer communities out of poverty. '_
## Initial Cleaning of Dataset

- Some values of our dataset include categorical values of strings. As such, we removed any white-spaces that were present in string values. 
- Some features had missing/null values. We removed them from our dataset.
## Data Exploration

- Some features are a numerical representation of another feature: the column 'education' was pre-encoded and attached to the dataset as the column 'education-­num'.
- The feature 'relationship' included specific values pertaining the individual's self-identified role in their relationship with a partner. Those in no romantic/guardian relationship were labeled as 'Not-in-family', or 'Unmarried'. This specific feature was lacking in terms of:
    - single point of comparison. The self-identified role was relative and not unique enough to a single individual.
    - vague attributes and minimal impact on our income_values; we chose to remove this feature from our dataset.
- 'native-country' and 'race' were extremely skewed data points, and so we kept that in mind as moving forward with the pre-processing. Extremely skewed data could prove to be unhelpful/irrelevant in our model.
- From the description of 'workclass', we noticed some overlap between the categories of the values. 
    - 'Without­pay' and 'Never­worked' both meant the individual did not have an income.
    - 'Federal-­gov', 'Local-­gov', 'State-­gov' were all government jobs.
    Still, we chose to see all the data points individual impact on our income through the heatmaps before choosing to simplify any of the 'workclass' categories.
- the 'occupation' vs '50k' (ie. income_values) heatmap showed that there were specific jobs that were proving to have a great impact on the income, while all others were under 0.1 measure of correlation. We decided to single out those specific values and combine the rest of the jobs. 
- 'fnlwgt' included values that described how common an individual's position reflected in society. This specific features poses irrelevant to our goal, and so we eventually removed it from our selected features. 
- 'education' had values that spanned lower than hs graduate. Its heatmap against '50k' showed that those specifics didnt really alter our income_values.
## Pre-processing Data

- We first did an initial check to verify that our data included no null values.
- We dropped the feature 'relationship' as it seemed insufficient for our problem.
- We then encoded all our categorical values and used a correlation matrix and pair-plots to compare our features. We could see from the pair plot that there don't seem to be any apparent correlation between the features. As a result, we continued analyzing the features further as relatively independent features.
- We grouped any of the grade-school level education values to a 'non-hs graduate', to simply our values.
- We then normalized and standardized our numerical data in order to give all features equal weightage for a second round of comparison.
- After a second round of heatmaps and correlation matrix, our final list of selected columns were as follows: 'edu_generalized', 'sex_values', 'marital_values', 'hours-per-week', 'capital-loss', 'capital-gain', 'age', 'occupation_values', and 'workclass_values'. Our final column will be our result column: 'income_values'.
- Finally, we simplified the categories of 'occupation' and 'workclass' to highlight the predominant features that would affect our outcome and also simplify our model's complexity.
- With a last correlation matrix check, we could verify that our selected features play a greater role in determining the outcome of our income_values, and can proceed with a simplified dataset. 

## First Model (Neural Network)

- We split our preprocessed data with a 70/30 split.
- The layers used in the neural network are as follows (activation, nodes):
    -Relu, 7
    -Tanh, 4
    -Selu, 6
    -Softplus, 5
    -Sigmoid, 1
- The loss function used is binary cross-entropy.
- We trained the model with 10 epochs.
- After using the trained model on our test split, we used a threshold of 0.5 to covnert the predicted values to 0 and 1.
- The classification report gave us a 1.00 accuracy.

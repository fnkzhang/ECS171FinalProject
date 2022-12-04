# ECS171FinalProjectGroup17
Group 17

Dataset PDF: https://cseweb.ucsd.edu/classes/sp15/cse190-c/reports/sp15/048.pdf

Link to Colab NTBK: https://colab.research.google.com/drive/1D4CDiRiPCTuvCX6Iq1XjwSHPKIyKh6p_?usp=sharing 

## Abstract

- Members: George Luu, Tiching Kao, Frank Zhang, Teresa Lin, Pallavi Gupta, Nicolas Conrad
- Link: https://archive.ics.uci.edu/ml/datasets/Adult
- Description: Data set contains age, workclass, income, education, marital status, occupation, relationship, race, sex, capital gain and loss, workhours per week, native country, and <50k or >50k income


_'A person’s earned income is correlated with numerous factors about the individual. This dataset provides information on whether a person earns more than 50k annually as well as other features about the person including the country they came from, their highest education level, age, occupation, etc. Neural networks are supervised learning models that accept multiple attributes in order to try to predict an outcome. We will train a neural network model to predict an individual’s financial status based on factors such as age, workclass, education, etc. Attributes will be weighed. Thus, our goal with this supervised learning model is to predict whether a person earns above or below 50k annually using the person’s other features. This is useful because it allows us to analyze what are the attributes that determine who earns more than 50,000 dollars, and thus elevate poorer communities out of poverty. '_
## Introduction 
- We chose this project because we wanted to explore the factors that contribute to an individual's financial success. Given that in the United States there is a wealth inequality problem, the project will help us understand the factors that can impact or limit an individual's income. We want to know the strongest indicators that can lead or predict income mobility. The burden of helping Americans obtain income mobility is the responsiblity of the government and the legislature. The idealistic goal of the project is to find what determines financial success, in this case, 50,000 dollars per year in 2005. Using that information, it, ideally, should influence legislation and policy making. For example, it should not be a requirement to be a business owner to make money since not everyone could be a business owner due to supply and demand. If that were the case supported by this project and other studies, the government should change that possibly by increasing minimum wage or supporting unions. Another example could be found that college is a determining factor and resultantly there could be a higher cultural emphasis on education as well as efforts to lower cost of tuition. Our project intends to be a reflection on the profile of the American working class and to positively impact income mobility.
## Methods

### Initial Cleaning of Dataset

- Some values of our dataset include categorical values of strings. As such, we removed any white-spaces that were present in string values. 
- Some features had missing/null values. We removed them from our dataset.
### Data Exploration

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
### Pre-processing Data

[Preprocessing Code](Preprocessing.ipynb)
- We first did an initial check to verify that our data included no null values.
- We dropped the feature 'relationship' as it seemed insufficient for our problem.
- We then encoded all our categorical values and used a correlation matrix and pair-plots to compare our features. We could see from the pair plot that there don't seem to be any apparent correlation between the features. As a result, we continued analyzing the features further as relatively independent features.
- We grouped any of the grade-school level education values to a 'non-hs graduate', to simply our values.
- We then normalized and standardized our numerical data in order to give all features equal weightage for a second round of comparison.
- After a second round of heatmaps and correlation matrix, our final list of selected columns were as follows: 'edu_generalized', 'sex_values', 'marital_values', 'hours-per-week', 'capital-loss', 'capital-gain', 'age', 'occupation_values', and 'workclass_values'. Our final column will be our result column: 'income_values'.
- Finally, we simplified the categories of 'occupation' and 'workclass' to highlight the predominant features that would affect our outcome and also simplify our model's complexity.
- With a last correlation matrix check, we could verify that our selected features play a greater role in determining the outcome of our income_values, and can proceed with a simplified dataset. 

### First Model (Neural Network)

- We split our preprocessed data with a 70/30 split.
- The layers used in the neural network are as follows (activation, nodes):
    - Relu, 7
    - Tanh, 4
    - Selu, 6
    - Softplus, 5
    - Sigmoid, 1
```
model = Sequential()
model.add(Dense(input_dim=9, units=7, activation='relu'))
model.add(Dense(units = 4, activation = 'tanh'))
model.add(Dense(units = 6, activation = 'selu'))
model.add(Dense(units = 5, activation = 'softplus'))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy')
```
- The loss function used is binary cross-entropy.
- We trained the model with 10 epochs.
- After using the trained model on our test split, we used a threshold of 0.5 to covnert the predicted values to 0 and 1.
- The classification report gave us an accuracy of 0.83.
- We think our model isn't really overfitting nor underfittiing as our loss graph quickly flattens out.

### Second Model (Decision Tree)
- We used the same preprocessed data as with our previous model.
	- Train data had attributes of education, sex, marital status, hours per week, capital loss, capital gain, age, occupation, work class.
	- Test data carried values of whether the individual's income was greater than or less than 50k.
- We split our preproccesed data with a 70/30 split.
- The model had a max depth layer of 5.
- Gini impurity was used.
```
model2 = tree.DecisionTreeClassifier(max_depth = 5)
```
## Results

### Preprocessing Data
Our data after preprocessing.
![Preprocessed Data](https://github.com/fnkzhang/ECS171FinalProject/blob/main/images/Preprocessed%20Data.png?raw=true)

### First Model (Neural Network)
![Classification Report](https://github.com/fnkzhang/ECS171FinalProject/blob/main/images/NN%20ClassReport.png?raw=true)
![Training Loss](https://github.com/fnkzhang/ECS171FinalProject/blob/main/images/NN%20Training%20Loss.png?raw=true)

### Second Model (Decision Tree)
![decision tree class report](https://github.com/fnkzhang/ECS171FinalProject/blob/main/images/decisiontreeclassreport.png?raw=true)
![decision tree](https://github.com/fnkzhang/ECS171FinalProject/blob/main/images/decisiontree.jpg?raw=true)


## Discussion

## Conclusion
- Most of what we did went well; we were able to achieve our goal of figuring out what factors into people earning more.
- Our models were also decently accurate, meaning that these results are also valid.
- After looking at the results, we likely should have kept native country as a data column, as while a decent bit of our important features correlated well with earning 50k, others did not.
- Thus, we should have checked native country, as it could still be an important feature.
## Collaboration

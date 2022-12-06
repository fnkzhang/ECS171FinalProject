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
- We chose this project because we wanted to explore the factors that contribute to an individual's financial success. It has become more and more apparent that, in the United States, there is a wealth inequality problem. The project will help us understand the factors that can impact or limit an individual's income. We want to know the strongest indicators that can lead or predict income mobility. As a government for and by the people, the burden of helping Americans obtain income mobility is the responsibility of the government and the legislature that is enacted. The idealistic goal of the project is to find what determines financial success, in this case, 50,000 dollars per year in 2005. Using that information, it should influence legislation and policy making in a perfect world. For example, it should not be a requirement to be a business owner to make money since not everyone could be a business owner due to supply and demand. If it is concluded that only business owners make money, supported by this project and additional other studies, the government should change that, possibly by increasing minimum wage or supporting unions. Another example could be found that college is a determining factor and as a result, there could be a higher societal emphasis placed on education as well as efforts to lower the cost of tuition. Our project intends to be a reflection on the profile of the American working class and to positively impact income mobility. 
## Methods

### Initial Cleaning of Dataset

- Some values of our dataset include categorical values of strings. As such, we removed any white-spaces that were present in string values. 
- Some features had missing/null values. We removed them from our dataset.
### Data Exploration
[Data Exploration in NTBK](https://colab.research.google.com/drive/1D4CDiRiPCTuvCX6Iq1XjwSHPKIyKh6p_#scrollTo=RN_h-hiQ5L9K)

- Some features are a numerical representation of another feature: the column 'education' was pre-encoded and attached to the dataset as the column 'education-­num'.
- The feature 'relationship' included specific values pertaining the individual's self-identified role in their relationship with a partner. Those in no romantic/guardian relationship were labeled as 'Not-in-family', or 'Unmarried'.
- 'native-country' and 'race' were extremely skewed data points.
- From the description of 'workclass', we noticed some overlap between the categories of the values. 
    - 'Without­pay' and 'Never­worked' both meant the individual did not have an income.
    - 'Federal-­gov', 'Local-­gov', 'State-­gov' were all government jobs.
- the 'occupation' vs '50k' (ie. income_values) heatmap showed that there were specific jobs that were proving to have a great impact on the income, while all others were under 0.1 measure of correlation.
- 'fnlwgt' included values that described how common an individual's position reflected in society.
- 'education' had values that spanned lower than hs graduate. Its heatmap against '50k' showed that those specifics didnt really alter our income_values.
### Pre-processing Data

[Preprocessing in NTBK](https://colab.research.google.com/drive/1D4CDiRiPCTuvCX6Iq1XjwSHPKIyKh6p_#scrollTo=lZs2vDDQ4ewI)
- We first did an initial check to verify that our data included no null values.
- We dropped the feature 'relationship'.
- We then encoded all our categorical values and used a correlation matrix and pair-plots to compare our features.
- We grouped any of the grade-school level education values to a 'non-hs graduate'.
- We then normalized and standardized our numerical data.
- After a second round of heatmaps and correlation matrix, our final list of selected columns were as follows: 'edu_generalized', 'sex_values', 'marital_values', 'hours-per-week', 'capital-loss', 'capital-gain', 'age', 'occupation_values', and 'workclass_values'. Our final column will be our result column: '50k'.
- Finally, we simplified the categories of 'occupation' and 'workclass' to highlight the predominant features that would affect our outcome and also simplify our model's complexity.
- With a last correlation matrix check, we could verify that our selected features play a greater role in determining the outcome of our '50k' values, and can proceed with a simplified dataset. 

### First Model (Neural Network)
[First Model in NTBK](https://colab.research.google.com/drive/1D4CDiRiPCTuvCX6Iq1XjwSHPKIyKh6p_#scrollTo=pXvzeWvx8VwA&line=1&uniqifier=1)
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
[Second Model in NTBK](https://colab.research.google.com/drive/1D4CDiRiPCTuvCX6Iq1XjwSHPKIyKh6p_#scrollTo=8rpFpV6u8iiQ)
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
- This was our final model.
- As seen in the tree, some of the most important features were capital-gain, capital-loss, marriage-status, and education level.
## Discussion
### Data Exploration
- There is no need to store the same feature values in two different ways; one of the columns can be dropped.
- The feature 'relationship' values were lacking in terms of:
  - single point of comparison. The self-identified role was relative and not unique enough to a single individual.
  - vague attributes and minimal impact on our income_values. Seems to be not entirely helpful or descriptive.
  Ultimately, it feels like subjective descriptors for our problem, and therefore not could result in a biased model. We decided to remove it.
- 'native-country' and 'race' were extremely skewed data points and could prove to be unhelpful/irrelevant in our model.
- Feature 'workclass' values where scattered and pretty niche fields, yet we chose to see all the data points individual impact on our income through the heatmaps before choosing to simplify any of the 'workclass' categories. 
- the 'occupation' vs '50k' (ie. income_values) heatmap showed that there were specific jobs that were proving to have a great impact on the income, while all others were under 0.1 measure of correlation. We decided to single out those specific values and combine the rest of the jobs so we could use the niche jobs as a factor in our model.
- 'fnlwgt' poses irrelevant to our goal, and so we eventually removed it from our selected features.
- 'education' had values that spanned lower than hs graduate; these specifics aren't necessarily crucial in determining a persons income, as verified through the heatmaps.
- 'capital gain/loss' seemed like a obvious, and therefore possible repetitive feature as one gaining money would obviously mean that they earn more money. However, we kept it due to believing that it represented good or bad investments of an individual.

### Preprocessing Data
- The initial check for null values was to verify we weren't dealing with any missing values.
- We dropped the feature 'relationship' as it seemed insufficient for our problem.
- We could see from the pair plot that there don't seem to be any apparent correlation between the features. As a result, we continued analyzing the features further as relatively independent features.
- We grouped any of the grade-school level education values to a 'non-hs graduate', to simply our values.
- We then normalized and standardized our numerical data in order to give all features equal weightage for a second round of comparison.
- We simplified the categories of 'occupation' and 'workclass' to highlight the predominant features that would affect our outcome and also simplify our model's complexity.

### Model 1 (Neural Network)
- We chose this model because it was decently familiar to us: it was used in lectures and discussion on the iris dataset, and we also had past experience with it in HW2
- However, after we built it, we realized it was not very good for deducing feature importance, leading us to model 2.
- The increase in training loss could have been an indication of overfitting, as the model was getting too specific resulting in higher loss for the rest of the data. Regardless, this specific model was displaying the indecisive case of increasing training loss and yet relatively high test accuracy, and so we thought to try a different model.
### Model 2 (Decision Tree)
- With model 2, we used a decision tree as it is easy to deduce feature importance by looking at the actual tree.
- Again, some of the most important features were capital-gain, capital-loss, marriage-status, and education level.
- Marriage-status as a feature seems like it would be affected by income levels and not the other way around, however.
    - It seems more likely that with money, you are more likely to be able to support a family, rather than with a family, you earn more.
### Interpretation
We found that the most important features of financial success were capital gain and loss, marital status, and education. Capital gain can refer to an increase in assets such as cars or an increase in stock values. These capital gains are an indicator of the wealth and assets of an individual. We rationalize that wealth is important to predict income because it offers economic safety nets for riskier careers such as arts or athletics and it also betters access to education and college.<sub>1</sub> This connects to our next finding in that education is another leading factor in predicting financial success. We found that higher education is important to increasing the chances of an individual making more money. Circling back on capital gains and losses, wealthier people have the ability to move to better locations with better schools and pay for college tuition so the students do not have to worry about part time jobs, rent, or food.<sub>2</sub> This is not to generalize all college students as having generational wealth or wealthy parents but a connection we made with our model.  

The data set did have native country and race however, the data was extremely skewed and dropped. In a future study, we would like to explore this type of data set but with more or better details about race, geographical location, and college majors. With better race data and immigration, we could have explore the arguments of reparations and its impacts to fix the racial wealth gap. In the end, it is clear that having wealth and/or more education is the best indicator of future success. The barriers of entry to college needs to be lowered beyond just tuition which can mean cheaper housing, lower application costs, and accessibility to more local colleges.
<sub>1</sub> https://www.americanprogress.org/article/eliminating-black-white-wealth-gap-generational-challenge/
<sub>2</sub> https://www.bestcolleges.com/blog/generational-wealth-first-generation-students/

## Conclusion
We found that having assets and higher education are the most significant factors in determining an individual’s salary. The models are consistent between published and unpublished models, indicating the result’s validity. 
While native country and race were possible attributes that could have been kept as data, our results can be incorporated with other studies that can explore wealth gaps between people of different backgrounds. Our findings hope to influence future decisions that can help American society.

## Collaboration
### Nicolas Conrad
- Organized meeting times
- Created logistic model to compare to our 1st NN model. Compared the two and determined the NN was better.
- Discussed the inclusion of the capital gains feature. 
- Discussed which model to choose in the model 2 phase (Decision Tree vs. Gaussian Naive Bayes).
### Pallavi Gupta
- Worked on Data exploration and Pre-processing the data with Frank's heatmaps and initial dataset clean by Tiching.
- Fixed selected data feature bug to help run model 1.
- Added ReadMe sections 'Initial Cleaning of Dataset', 'Pre-processing', 'Data Exploration', and contributed to 'Discussion' sections.

### Tiching Kao
- Cleaned up initial data by converting file to .csv and removing whitespace.
- Wrote part of abstract.
- Added Model 1 into the Methods section of writeup and some figure into the Results section.
- Played with buggy version of Model 1 though did not figure out why it did nto work.

### Teresa Lin
- Tried to do the data with some models(such as using Logistic Regression) but it does not seem to work better than using Neural Network(which is the one we all agree to use). 
### George Luu
- Wrote introduction, interpretation, conclusion, parts of Abstract
- Created and tested a Gaussian Naive Bayes model for Model 2
- Discussed which model should be used for Model 2
### Frank Zhang
- Created heatmaps to check correlation between features and earning 50k
- Implemented buggy version of model1 (issues fixed by Pallavi)
- Implemented Model2 after discussing with other groupmates
- Added Model2 Methods & Results sections and contributed to the Conclusion section in the final writeup

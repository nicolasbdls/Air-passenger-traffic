#!/usr/bin/env python
# coding: utf-8

# # <a href="http://www.datascience-paris-saclay.fr">Paris Saclay Center for Data Science</a>
# # <a href=https://www.ramp.studio/problems/air_passengers>RAMP</a> on predicting the number of air passengers
# 
# <i> Balázs Kégl (LAL/CNRS), Alex Gramfort (Inria), Djalel Benbouzid (UPMC), Mehdi Cherti (LAL/CNRS) </i>

# ## Introduction
# The data set was donated to us by an unnamed company handling flight ticket reservations. The data is thin, it contains
# <ul>
# <li> the date of departure
# <li> the departure airport
# <li> the arrival airport
# <li> the mean and standard deviation of the number of weeks of the reservations made before the departure date
# <li> a field called <code>log_PAX</code> which is related to the number of passengers (the actual number were changed for privacy reasons)
# </ul>
# 
# The goal is to predict the <code>log_PAX</code> column. The prediction quality is measured by RMSE. 
# 
# The data is obviously limited, but since data and location informations are available, it can be joined to external data sets. <b>The challenge in this RAMP is to find good data that can be correlated to flight traffic</b>.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
pd.set_option('display.max_columns', None)


# ## Load the dataset using pandas

# The training and testing data are located in the folder `data`. They are compressed `csv` file (i.e. `csv.bz2`). We can load the dataset using pandas.

# In[2]:


data = pd.read_csv(
    os.path.join('data', 'train.csv.bz2')
)


# In[3]:


data.info()


# In[61]:


data.head()


# So as stated earlier, the column `log_PAX` is the target for our regression problem. The other columns are the features which will be used for our prediction problem. If we focus on the data type of the column, we can see that `DateOfDeparture`, `Departure`, and `Arrival` are of `object` dtype, meaning they are strings.

# In[4]:


data[['DateOfDeparture', 'Departure', 'Arrival']].head()


# While it makes `Departure` and `Arrival` are the code of the airport, we see that the `DateOfDeparture` should be a date instead of string. We can use pandas to convert this data.

# In[5]:


data.loc[:, 'DateOfDeparture'] = pd.to_datetime(data.loc[:, 'DateOfDeparture'])


# In[6]:


data.info()


# When you will create a submission, `ramp-workflow` will load the data for you and split into a data matrix `X` and a target vector `y`. It will also take care about splitting the data into a training and testing set. These utilities are available in the module `problem.py` which we will load.

# In[10]:


import problem


# The function `get_train_data()` loads the training data and returns a pandas dataframe `X` and a numpy vector `y`.

# In[11]:


X, y = problem.get_train_data()


# In[12]:


type(X)


# In[13]:


type(y)


# We can check the information of the data `X`

# In[14]:


X.info()


# Thus, this is important to see that `ramp-workflow` does not convert the `DateOfDeparture` column into a `datetime` format. Thus, keep in mind that you might need to make a conversion when prototyping your machine learning pipeline later on. Let's check some statistics regarding our dataset.

# In[15]:


print(min(X['DateOfDeparture']))
print(max(X['DateOfDeparture']))


# In[16]:


X['Departure'].unique()


# In[17]:


_ = plt.hist(y, bins=50)


# In[18]:


_ = X.hist('std_wtd', bins=50)


# In[19]:


_ = X.hist('WeeksToDeparture', bins=50)


# In[20]:


X.describe()


# In[21]:


X.shape


# In[22]:


print(y.mean())
print(y.std())


# ## Preprocessing dates

# Getting dates into numerical columns is a common operation when time series data is analyzed with non-parametric predictors. The code below makes the following transformations:
# 
# - numerical columns for year (2011-2012), month (1-12), day of the month (1-31), day of the week (0-6), and week of the year (1-52)
# - number of days since 1970-01-01

# In[23]:


# Make a copy of the original data to avoid writing on the original data
X_encoded = X.copy()

# following http://stackoverflow.com/questions/16453644/regression-with-date-variable-using-scikit-learn
X_encoded['DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])
X_encoded['year'] = X_encoded['DateOfDeparture'].dt.year
X_encoded['month'] = X_encoded['DateOfDeparture'].dt.month
X_encoded['day'] = X_encoded['DateOfDeparture'].dt.day
X_encoded['weekday'] = X_encoded['DateOfDeparture'].dt.weekday
X_encoded['week'] = X_encoded['DateOfDeparture'].dt.week
X_encoded['n_days'] = X_encoded['DateOfDeparture'].apply(lambda date: (date - pd.to_datetime("1970-01-01")).days)


# In[24]:

X_encoded.drop(columns=["DateOfDeparture"], inplace=True)
X_encoded.tail(5)

# In[25]:
#merge external data
ext_data = pd.read_excel("CY12CommercialServiceEnplanements.xlsx")
ext_data.columns

# We will perform all preprocessing steps within a scikit-learn [pipeline](https://scikit-learn.org/stable/modules/compose.html) which chains together tranformation and estimator steps. This offers offers convenience and safety (help avoid leaking statistics from your test data into the trained model in cross-validation) and the whole pipeline can be evaluated with `cross_val_score`.
# 
# To perform the above encoding within a scikit-learn [pipeline](https://scikit-learn.org/stable/modules/compose.html) we will a function and using `FunctionTransformer` to make it compatible with scikit-learn API.

# In[26]:
ext_data = ext_data.iloc[:20]
new_ext_df = ext_data[['Locid', 'CY 12 Enplanements', 'CY 11 Enplanements']]

# In[27]:
new_ext_df = new_ext_df.rename(columns={'Locid': 'Departure', 'CY 12 Enplanements': 'Enplanements 2012', 'CY 11 Enplanements': 'Enplanements 2011'})
X_passengers = new_ext_df.copy()

# %%
#merge data
X_merged = pd.merge(X_encoded, X_passengers, how='left', on=['Departure'], sort=False)
X_merged.shape

# %%
X_merged['passengers load 2011'] = X_merged.loc[X_merged['year']==2011, ['Enplanements 2011']]
X_merged['passengers load 2011'].fillna(0, inplace=True)
X_merged['passengers load 2012'] = X_merged.loc[X_merged['year']==2012, ['Enplanements 2012']]
X_merged['passengers load 2012'].fillna(0, inplace=True)
X_merged['passengers load'] = X_merged['passengers load 2011'] + X_merged['passengers load 2012']
X_merged.drop(['passengers load 2011', 'passengers load 2012', 'Enplanements 2011', 'Enplanements 2012'], axis=1, inplace=True)
X_merged

# %%
X_merged.to_csv(r'C:\Users\nicol\OneDrive\Documents\Python4DS\airpassengers\air_passengers_py4ds2020\newdata.csv', index=False)

# %%
location_df = pd.read_csv("location.csv")

# %%
location_df.head()
# %%
location_data = location_df.loc[location_df['AIRPORT'].isin(['ORD', 'LAS', 'DEN', 'ATL', 'SFO', 'EWR', 'IAH', 'LAX', 'DFW',
       'SEA', 'JFK', 'PHL', 'MIA', 'DTW', 'BOS', 'MSP', 'CLT', 'MCO',
       'PHX', 'LGA'])]
location_data.drop(['Unnamed: 3'], axis=1, inplace=True)
location_data.rename(columns={'AIRPORT':'Departure'}, inplace=True)
location_data.drop_duplicates(subset=['Departure'],inplace=True)
location_data['location_dep'] = list(zip(location_data.LATITUDE, location_data.LONGITUDE))
location_data.drop(['LATITUDE', 'LONGITUDE'], axis=1, inplace=True)
#location_data['Location'] = location_data['LONGITUDE'] + ',' + location_data['LATITUDE']
#location_data['Location'] = location_data[['LATITUDE', 'LONGITUDE']].apply(lambda x: ', '.join(x.map(str)), axis = 1)
#location_data['location'] = location_data['LATITUDE'] + location_data['LONGITUDE']
#location_data.drop(['LATITUDE', 'LONGITUDE'], axis=1, inplace=True)
location_arr = location_data.copy()
location_arr.rename(columns={'Departure':'Arrival', 'location_dep':'location_arr'}, inplace=True)
location_arr
# %%
X_merged2 = pd.merge(X_merged, location_data, how='left', on=['Departure'], sort=False)
X_merged2 = pd.merge(X_merged2, location_arr, how='left', on=['Arrival'], sort=False)
X_merged2

# %%
from geopy.distance import distance
X_merged2['distance km'] = X_merged2['location_arr']
for i in range(X_merged2.shape[0]):
    X_merged2['distance km'].iloc[i] = distance(X_merged2['location_dep'].iloc[i],X_merged2['location_arr'].iloc[i]).km
X_merged2.drop(['location_dep', 'location_arr'], axis=1, inplace=True)
#print(distance(X_merged2['location_dep'].iloc[0], X_merged2['location_arr'].iloc[0]).km)


# %%
X_merged2.to_csv(r'C:\Users\nicol\OneDrive\Documents\Python4DS\airpassengers\air_passengers_py4ds2020\testemoica.csv', index=False)
# In[25]:
from sklearn.preprocessing import FunctionTransformer

def _encode_dates(X):
    # With pandas < 1.0, we wil get a SettingWithCopyWarning
    # In our case, we will avoid this warning by triggering a copy
    # More information can be found at:
    # https://github.com/scikit-learn/scikit-learn/issues/16191
    X_encoded = X.copy()

    # Make sure that DateOfDeparture is of datetime format
    X_encoded.loc[:, 'DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])
    # Encode the DateOfDeparture
    X_encoded.loc[:, 'year'] = X_encoded['DateOfDeparture'].dt.year
    X_encoded.loc[:, 'month'] = X_encoded['DateOfDeparture'].dt.month
    X_encoded.loc[:, 'day'] = X_encoded['DateOfDeparture'].dt.day
    X_encoded.loc[:, 'weekday'] = X_encoded['DateOfDeparture'].dt.weekday
    X_encoded.loc[:, 'week'] = X_encoded['DateOfDeparture'].dt.week
    X_encoded.loc[:, 'n_days'] = X_encoded['DateOfDeparture'].apply(
        lambda date: (date - pd.to_datetime("1970-01-01")).days
    )
    # Once we did the encoding, we will not need DateOfDeparture
    return X_encoded.drop(columns=["DateOfDeparture"])

date_encoder = FunctionTransformer(_encode_dates)

# In[26]:


date_encoder.fit_transform(X).head()


# ## Random Forests

# Tree-based algorithms requires less complex preprocessing than linear-models. We will first present a machine-learning pipeline where we will use a random forest. In this pipeline, we will need to:
# 
# - encode the date to numerical values (as presented in the section above);
# - oridinal encode the other categorical values to get numerical number;
# - keep numerical features as they are.
# 
# Thus, we want to perform three different processes on different columns of the original data `X`. In scikit-learn, we can use [`make_column_transformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_transformer.html) to perform such processing.

# In[27]:


from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer

date_encoder = FunctionTransformer(_encode_dates)
date_cols = ["DateOfDeparture"]

categorical_encoder = OrdinalEncoder()
categorical_cols = ["Arrival", "Departure"]

preprocessor = make_column_transformer(
    (date_encoder, date_cols),
    (categorical_encoder, categorical_cols),
    remainder='passthrough',  # passthrough numerical columns as they are
)


# We can combine our preprocessor with an estimator (`RandomForestRegressor` in this case), allowing us to make predictions.

# In[28]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

n_estimators = 10
max_depth = 10
max_features = 10

regressor = RandomForestRegressor(
    n_estimators=n_estimators, max_depth=max_depth, max_features=max_features
)

pipeline = make_pipeline(preprocessor, regressor)


# We can cross-validate our `pipeline` using `cross_val_score`. Below we will have specified `cv=5` meaning KFold cross-valdiation splitting will be used, with 8 folds. The mean squared error regression loss is calculated for each split. The output `score` will be an array of 5 scores from each KFold. The score mean and standard deviation of the 5 scores is printed at the end.

# In[29]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    pipeline, X, y, cv=5, scoring='neg_mean_squared_error'
)
rmse_scores = np.sqrt(-scores)

print(
    f"RMSE: {np.mean(rmse_scores):.4f} +/- {np.std(rmse_scores):.4f}"
)


# ## Linear regressor
# 
# When dealing with a linear model, we need to one-hot encode categorical variables instead of ordinal encoding and standardize numerical variables. Thus we will:
# 
# - encode the date;
# - then, one-hot encode all categorical columns, including the encoded date as well;
# - standardize the numerical columns.

# In[38]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

date_encoder = FunctionTransformer(_encode_dates)
date_cols = ["DateOfDeparture"]

categorical_encoder = OneHotEncoder(handle_unknown="ignore")
categorical_cols = [
    "Arrival", "Departure", "year", "month", "day",
    "weekday", "week", "n_days"
]

numerical_scaler = StandardScaler()
numerical_cols = ["WeeksToDeparture", "std_wtd"]

preprocessor = make_column_transformer(
    (categorical_encoder, categorical_cols),
    (numerical_scaler, numerical_cols)
)


# We can now combine our `preprocessor` with the `LinearRegression` estimator in a `Pipeline`:

# In[39]:


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

pipeline = make_pipeline(date_encoder, preprocessor, regressor) 
#first the date data is split, then onehotencoded and scaled, and then the regression is applied


# And we can evaluate our linear-model pipeline:

# In[40]:


scores = cross_val_score(
    pipeline, X, y, cv=5, scoring='neg_mean_squared_error'
)
rmse_scores = np.sqrt(-scores)

print(
    f"RMSE: {np.mean(rmse_scores):.4f} +/- {np.std(rmse_scores):.4f}"
)


# # Merging external data

# The objective in this RAMP data challenge is to find good data that can be correlated to flight traffic. We will use some weather data (saved in `submissions/starting_kit`) to provide an example of how to merge external data in a scikit-learn pipeline.
# 
# Your external data will need to be included in your submissions folder - see [RAMP submissions](#RAMP-submissions) for more details.
# 
# First we will define a function that merges the external data to our feature data.

# In[42]:


# when submitting a kit, the `__file__` variable will corresponds to the
# path to `estimator.py`. However, this variable is not defined in the
# notebook and thus we must define the `__file__` variable to imitate
# how a submission `.py` would work.
__file__ = os.path.join('submissions', 'starting_kit', 'estimator.py')
filepath = os.path.join(os.path.dirname(__file__), 'external_data.csv')
filepath


# In[43]:


pd.read_csv(filepath).head()


# In[44]:


def _merge_external_data(X):
    filepath = os.path.join(
        os.path.dirname(__file__), 'external_data.csv'
    )
    
    X = X.copy()  # to avoid raising SettingOnCopyWarning
    # Make sure that DateOfDeparture is of dtype datetime
    X.loc[:, "DateOfDeparture"] = pd.to_datetime(X['DateOfDeparture'])
    # Parse date to also be of dtype datetime
    data_weather = pd.read_csv(filepath, parse_dates=["Date"])

    X_weather = data_weather[['Date', 'AirPort', 'Max TemperatureC']]
    X_weather = X_weather.rename(
        columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival'})
    X_merged = pd.merge(
        X, X_weather, how='left', on=['DateOfDeparture', 'Arrival'], sort=False
    )
    return X_merged

data_merger = FunctionTransformer(_merge_external_data)


# Double check that our function works:

# In[45]:


data_merger.fit_transform(X).head()


# Use `FunctionTransformer` to make our function compatible with scikit-learn API:

# We can now assemble our pipeline using the same `data_merger` and `preprocessor` as above:

# In[46]:


from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer

date_encoder = FunctionTransformer(_encode_dates)
date_cols = ["DateOfDeparture"]

categorical_encoder = OrdinalEncoder()
categorical_cols = ["Arrival", "Departure"]

preprocessor = make_column_transformer(
    (date_encoder, date_cols),
    (categorical_encoder, categorical_cols),
    remainder='passthrough',  # passthrough numerical columns as they are
)


# In[47]:


n_estimators = 10
max_depth = 10
max_features = 10

regressor = RandomForestRegressor(
    n_estimators=n_estimators, max_depth=max_depth, max_features=max_features
)

pipeline = make_pipeline(data_merger, preprocessor, regressor)


# In[48]:


scores = cross_val_score(
    pipeline, X, y, cv=5, scoring='neg_mean_squared_error'
)
rmse_scores = np.sqrt(-scores)

print(
    f"RMSE: {np.mean(rmse_scores):.4f} +/- {np.std(rmse_scores):.4f}"
)


# ## Feature importances
# 
# We can check the feature importances using the function [`sklearn.inspection.permutation_importances`](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html). Since the first step of our pipeline adds the new external feature `Max TemperatureC`, we want to apply this transformation after adding `Max TemperatureC`, to check the importances of all features. Indeed, we can perform `sklearn.inspection.permutation_importances` at any stage of the pipeline, as we will see later on.
# 
# 
# The code below:
# 
# * performs `transform` on the first step of the pipeline (`pipeline[0]`) producing the transformed train (`X_train_augmented`) and test (`X_test_augmented`) data
# * the transformed data is used to fit the pipeline from the second step onwards
# 
# Note that pipelines can be sliced. `pipeline[0]` obtains the first step (tuple) of the pipeline. You can further slice to obtain either the transformer/estimator (first item in each tuple) or column list (second item within each tuple) inside each tuple. For example `pipeline[0][0]` obtains the transformer of the first step of the pipeline (first item of the first tuple).

# In[49]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42
)

merger = pipeline[0]
X_train_augmented = merger.transform(X_train)
X_test_augmented = merger.transform(X_test)

predictor = pipeline[1:]
predictor.fit(X_train_augmented, y_train).score(X_test_augmented, y_test)


# With the fitted pipeline, we can now use `permutation_importance` to calculate feature importances:

# In[50]:


from sklearn.inspection import permutation_importance

feature_importances = permutation_importance(
    predictor, X_train_augmented, y_train, n_repeats=10
)


# Here, we plot the permutation importance using the training set. The higher the value, more important the feature is.

# In[51]:


sorted_idx = feature_importances.importances_mean.argsort()

fig, ax = plt.subplots()
ax.boxplot(feature_importances.importances[sorted_idx].T,
           vert=False, labels=X_train_augmented.columns[sorted_idx])
ax.set_title("Permutation Importances (train set)")
fig.tight_layout()
plt.show()


# We can replicate the same processing on the test set and see if we can observe the same trend.

# In[52]:


from sklearn.inspection import permutation_importance

feature_importances = permutation_importance(
    predictor, X_test_augmented, y_test, n_repeats=10
)


# In[53]:


sorted_idx = feature_importances.importances_mean.argsort()

fig, ax = plt.subplots()
ax.boxplot(feature_importances.importances[sorted_idx].T,
           vert=False, labels=X_test_augmented.columns[sorted_idx])
ax.set_title("Permutation Importances (test set)")
fig.tight_layout()
plt.show()


# With the current version of scikit-learn, it is not handy but still possible to check the feature importances at the latest stage of the pipeline (once all features have been preprocessed).
# 
# The difficult part is to get the name of the features.

# In[54]:


preprocessor = pipeline[:-1]
predictor = pipeline[-1]

X_train_augmented = preprocessor.transform(X_train)
X_test_augmented = preprocessor.transform(X_test)


# Let's find out the feature names (in the future, scikit-learn will provide a `get_feature_names` function to handle this case).

# In[55]:


date_cols_name = (date_encoder.transform(X_train[date_cols])
                              .columns.tolist())
categorical_cols_name = categorical_cols
numerical_cols_name = (pipeline[0].transform(X_train)
                                  .columns[pipeline[1].transformers_[-1][-1]]
                                  .tolist())


# In[56]:


feature_names = np.array(
    date_cols_name + categorical_cols_name + numerical_cols_name
)
feature_names


# We can repeat the previous processing at this finer grain, where the transformed date columns are included.

# In[57]:


from sklearn.inspection import permutation_importance

feature_importances = permutation_importance(
    predictor, X_train_augmented, y_train, n_repeats=10
)


# Here, we plot the permutation importance using the training set. Basically, higher the value, more important is the feature.

# In[58]:


sorted_idx = feature_importances.importances_mean.argsort()

fig, ax = plt.subplots()
ax.boxplot(feature_importances.importances[sorted_idx].T,
           vert=False, labels=feature_names[sorted_idx])
ax.set_title("Permutation Importances (train set)")
fig.tight_layout()
plt.show()


# We can replicate the same processing on the test set and see if we can observe the same trend.

# In[59]:


from sklearn.inspection import permutation_importance

feature_importances = permutation_importance(
    predictor, X_test_augmented, y_test, n_repeats=10
)


# In[60]:


sorted_idx = feature_importances.importances_mean.argsort()

fig, ax = plt.subplots()
ax.boxplot(feature_importances.importances[sorted_idx].T,
           vert=False, labels=feature_names[sorted_idx])
ax.set_title("Permutation Importances (test set)")
fig.tight_layout()
plt.show()


# ## Submission
# 
# To submit your code, you can refer to the [online documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html).

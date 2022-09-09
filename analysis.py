import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LinearRegression, LogisticRegression

####################################################
# Need to run apple_watch_data.py to get StepCount.csv
# Need to run Find_EKG.py to get EKG_by_day.csv
# Need to run glucose_food_export.py to get freestyle_by_day.csv
# Need to run symptoms.py to get higest_symptom.csv
####################################################
st.set_page_config(layout="wide")
watch_path = './apple_watch_data/'
glucose_path = './'
ekg_path = './'


replace_dict = {'AppleWalkingSteadiness': ['%', 'float', 'mean'], 'WalkingSpeed': ['mi/hr', 'float', 'mean'],
                'BloodPressureSystolic': ['mmHg', 'int', 'mean'], 'WalkingAsymmetryPercentage': ['%', 'float', 'mean'],
                'WalkingStepLength': ['in', 'float', 'mean'], 'SixMinuteWalkTestDistance': ['m', 'int', 'mean'],
                'AppleExerciseTime': ['min', 'int', 'sum'], 'HeartRate': ['count/min', 'float', 'mean'],
                'DietaryFiber': ['g', 'float', 'sum'], 'BloodPressureDiastolic': ['mmHg', 'int', 'mean'],
                'FlightsClimbed': ['count', 'int', 'sum'], 'DietaryCalcium': ['mg', 'float', 'sum'],
                'WalkingDoubleSupportPercentage': ['%', 'float', 'mean'], 'Height': ['ft', 'float', 'mean'],
                'BloodGlucose': ['mg/dL', 'int', 'mean'], 'DietaryCarbohydrates': ['g', 'float', 'sum'],
                'WalkingHeartRateAverage': ['count/min', 'float', 'mean'], 'DietarySugar': ['g', 'float', 'sum'],
                'BodyFatPercentage': ['%', 'float', 'mean'], 'DietaryFatPolyunsaturated': ['g', 'float', 'sum'],
                'DietaryCholesterol': ['mg', 'float', 'sum'], 'EnvironmentalAudioExposure': ['dBASPL', 'float', 'mean'],
                'AppleStandTime': ['min', 'int', 'sum'], 'DistanceWalkingRunning': ['mi', 'float', 'sum'],
                'VO2Max': ['mL/min·kg', 'float', 'mean'], 'DietaryFatMonounsaturated': ['g', 'float', 'sum'],
                'DietaryIron': ['mg', 'float', 'sum'], 'RespiratoryRate': ['count/min', 'float', 'mean'],
                'BodyMass': ['lb', 'float', 'mean'], 'DietaryFatSaturated': ['g', 'float', 'mean'],
                'HeadphoneAudioExposure': ['dBASPL', 'float', 'mean'], 'BasalEnergyBurned': ['Cal', 'float', 'sum'],
                'RestingHeartRate': ['count/min', 'int', 'mean'], 'DietaryProtein': ['g', 'float', 'sum'],
                'BodyMassIndex': ['count', 'float', 'mean'], 'DietarySodium': ['mg', 'float', 'sum'],
                'DietaryFatTotal': ['g', 'float', 'sum'], 'ActiveEnergyBurned': ['Cal', 'float', 'sum'],
                'StepCount': ['count', 'int', 'sum'], 'DietaryEnergyConsumed': ['Cal', 'float', 'sum'],
                'HeartRateVariabilitySDNN': ['ms', 'float', 'mean'], 'DietaryVitaminC': ['mg', 'float', 'sum'],
                'LeanBodyMass': ['lb', 'float', 'mean'], 'DietaryPotassium': ['mg', 'float', 'sum']}

root_types = ['Record', 'ActivitySummary', 'Workout']
ekg_file = 'EKG_by_day.csv'
glucose_file = 'freestyle_by_day.csv'
watch_list = os.listdir(watch_path)
watch_file = 'StepCount.csv'

# set potential y_cols
target = st.sidebar.radio('What is the target?', ['PACs and AFib', 'Histamine Headache'])
if target == 'PACs and AFib':
    potential_y = ['afib', 'PACs', 'day_before_afib']
if target == 'Histamine Headache':
    potential_y = ['more_than_1_next_day', 'more_than_2_next_day',
                   'more_than_3_next_day', 'more_than_4_next_day',
                   'greater_than_1', 'greater_than_2', 'greater_than_3', 'greater_than_4']


# load pac/afib data
ekg_df = pd.read_csv(ekg_file)
ekg_df.date = pd.to_datetime(ekg_df.date)


# load glucose data
glucose_df = pd.read_csv(glucose_file)
# st.write(glucose_df[glucose_df.date.str.contains('HEAD')])
glucose_df = glucose_df[~glucose_df.date.str.contains('>>>')]
# st.write(glucose_df[glucose_df.date.str.contains('>>>')])
glucose_df = glucose_df[~glucose_df.date.str.contains('===')]
# st.write(glucose_df[glucose_df.date.str.contains('===')])
# st.write(glucose_df[glucose_df.date.str.contains('date')])

glucose_df['date'] = pd.to_datetime(glucose_df['date'], errors='coerce', format='%Y-%m-%d')
# st.write(glucose_df)
combined_df = ekg_df.merge(glucose_df, on='date', how='outer')
# load symptom data
symptom_df = pd.read_csv('highest_symptom.csv')
symptom_df['date'] = pd.to_datetime(symptom_df['date'], errors='coerce')
# st.write(symptom_df)

combined_df = combined_df.merge(symptom_df, on='date', how='outer')
# st.write(combined_df)
# load watch data
watch_df = pd.read_csv(watch_path+watch_file)
watch_df = watch_df.drop(['type', 'unit'], axis=1)
watch_df.rename(columns={'value': 'steps'}, inplace=True)
watch_df['date'] = pd.to_datetime(watch_df['date'], errors='coerce')

# st.write(ekg_df.columns)
# st.write(glucose_df)
# st.write('printed')
# combined_df = pd.merge(ekg_df, glucose_df, on='date', how='outer')
# combined_df = combined_df.merge(wat÷, on='date', how='outer')
# st.write(combined_df)
combined_df = combined_df.merge(watch_df, on='date', how='outer')
# combined_df = combined_df[~combined_df.date.str.contains('HEAD')]
# combined_df = combined_df[~combined_df.date.str.contains('===')]
# combined_df = combined_df[~combined_df.date.str.contains('>>>')]
combined_df.date = pd.to_datetime(combined_df.date)
combined_df.sort_values(by='date', inplace=True)
# combined_df = combined_df.dropna()

columns = combined_df.columns
columns = [x for x in columns if x != 'date']
for col in columns:
    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    # combined_df[col] = combined_df[col].astype(int)

# add oura data
oura_df = pd.read_csv('oura.csv')
oura_df = oura_df.drop(['HRV Balance Score', 'Bedtime End', 'Bedtime Start'],
                       axis=1).rename(columns={'Steps': 'oura_steps'})
oura_df.date = pd.to_datetime(oura_df.date)
oura_df = oura_df.loc[1:, :].reset_index(drop=True)
strings = []
for col in oura_df.columns:
    if col == 'date':
        pass
    else:
        oura_df[col] = pd.to_numeric(oura_df[col], errors='coerce')

combined_df['day_before_afib'] = combined_df.afib.shift(-1)
combined_df = pd.merge(combined_df, oura_df, on='date', how='outer')
# remove nans in Y cols
combined_df = combined_df[combined_df.PACs.notna()]
combined_df = combined_df[combined_df.day_before_afib.notna()]
# st.write(combined_df)

# view combined_df by nan
combined_nan = combined_df.isna().sum().reset_index(
    drop=False).rename(columns={'index': 'cols', 0: 'nans'})
# drop cols that have more than max nans
max_nans = combined_nan.nans.max()
min_nans = combined_nan.nans.min()
nan_cutoff = st.sidebar.slider('Choose number of NaNs as a max',
                               min_value=min_nans, max_value=max_nans, value=max_nans)
surviving_cols = combined_nan[combined_nan.nans <= nan_cutoff].cols.tolist()
combined_df = combined_df[surviving_cols]

#
# set canonical combined_df
combined_df.dropna(inplace=True)

# create 1 hot encoded df for classification algorithms
# st.write(combined_df)
binary_combined = combined_df.copy()
for col in binary_combined.columns:
    if col in ['date']:
        pass
    elif col in potential_y:
        pass
    else:
        if binary_combined[col].isin([0, 1]).all():
            pass
        else:
            quant_25 = binary_combined[col].quantile(.25)
            median = binary_combined[col].quantile(.5)
            quant_75 = binary_combined[col].quantile(.75)
            binary_combined[f'{col}_1st_quartile'] = (binary_combined[col] < quant_25).astype(int)
            binary_combined[f'{col}_2nd_quartile'] = ((
                binary_combined[col] >= quant_25) & (binary_combined[col] < median)).astype(int)
            binary_combined[f'{col}_3rd_quartile'] = ((
                binary_combined[col] >= median) & (binary_combined[col] < quant_75)).astype(int)
            binary_combined[f'{col}_4th_quartile'] = (binary_combined[col] >= quant_75).astype(int)
            binary_combined.drop(col, axis=1, inplace=True)
# st.write(binary_combined)
# for col in ['PACs', 'afib', 'day_before_afib']:
for col in potential_y:
    binary_combined[col] = binary_combined[col].astype(int)
# st.write(binary_combined)


# set up for logistic regression
# potential_y = ['afib', 'PACs', 'day_before_afib']
# y_col = st.sidebar.selectbox('Choose a y', potential_y)
# X_cols = combined_df.columns.tolist()
# X_cols = [x for x in X_cols if x not in (potential_y+['date'])]
# X = combined_df[X_cols]
# y = combined_df[y_col]
# # split into training and testing
# test_fraction = (st.sidebar.slider('Select % of data for testing',
#                                    min_value=5, max_value=50, value=10))/100
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=(test_fraction))


models = ['GaussianNB', 'BernoulliNB', 'LogisticRegression']
model_type = st.sidebar.selectbox('Select a model', models)
if model_type == 'GaussianNB':
    # potential_y = ['afib', 'PACs', 'day_before_afib']
    y_col = st.sidebar.selectbox('Choose a y', potential_y)
    X_cols = binary_combined.columns.tolist()
    X_cols = [x for x in X_cols if x not in (potential_y+['date'])]
    X = binary_combined[X_cols]
    y = binary_combined[y_col]
    # split into training and testing
    test_fraction = (st.sidebar.slider('Select % of data for testing',
                                       min_value=5, max_value=50, value=10))/100
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(test_fraction))

    model = GaussianNB()
    y_pred = model.fit(X_train, y_train).predict(X_test)
    # st.write(model.class_prior_)
    odds_df = pd.DataFrame(model.theta_, columns=model.feature_names_in_)
    odds_df = odds_df.replace({0: np.nan})
    odds_df = odds_df.dropna(axis=1)
    for col in odds_df.columns:
        # odds_df[col] = odds_df[col].astype(float)
        odds_df.loc[3, col] = odds_df.loc[1, col]/odds_df.loc[0, col]

    # .dropna()
    Gaussian_odds = odds_df.T.rename(
        columns={0: 'odds_of_class_0', 1: 'odds_of_class_1', 3: 'odds_ratio'}).reset_index(drop=False)
    Gaussian_odds.rename(columns={'index': 'factors'}, inplace=True)
    Gaussian_odds.sort_values(by='odds_ratio', ascending=False, inplace=True)

    st.write(
        f'There are {(y_test != y_pred).sum()} missed out of {X_test.shape[0]} in the testing set from {combined_df.shape[0]} rows of data for accuracy of {round(1-((y_test != y_pred).sum()/X_test.shape[0]),2)}')
    # st.write(Gaussian_odds)
    top_ten = Gaussian_odds.head(10)
    bottom_ten = Gaussian_odds.tail(10)
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.bar(top_ten.factors, top_ten.odds_ratio)
    plt.title(f"Top ten associated with {y_col}")
    plt.xticks(rotation=70, ha='right')
    st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.bar(bottom_ten.factors, bottom_ten.odds_ratio)
    plt.title(f"Top ten associated with absence of {y_col}")
    plt.xticks(rotation=70, ha='right')
    st.pyplot(fig)
    # st.write(f'{model_type} has not yet been implemented')
    # end of Gaussian

elif model_type == 'BernoulliNB':  # before this will work need to change all factors to 0 and 1
    # potential_y = ['afib', 'PACs', 'day_before_afib']
    y_col = st.sidebar.selectbox('Choose a y', potential_y)
    X_cols = binary_combined.columns.tolist()
    X_cols = [x for x in X_cols if x not in (potential_y+['date'])]
    X = binary_combined[X_cols]
    y = binary_combined[y_col]
    # split into training and testing
    test_fraction = (st.sidebar.slider('Select % of data for testing',
                                       min_value=5, max_value=50, value=10))/100
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(test_fraction))

    model = BernoulliNB()
    y_pred = model.fit(X_train, y_train).predict(X_test)
    st.write(
        f'There are {(y_test != y_pred).sum()} missed out of {X_test.shape[0]} in the testing set from {combined_df.shape[0]} rows of data for accuracy of {round(1-((y_test != y_pred).sum()/X_test.shape[0]),2)}')
    # st.write(model.n_features_in_)
    factors = pd.DataFrame(model.feature_names_in_).rename(
        columns={0: 'factor'})
    factors = factors.reset_index(drop=False)
    # st.write(factors)
    odds = pd.DataFrame(model.feature_log_prob_.T).rename(
        columns={0: 'Odds of Class 0', 1: 'Odds of Class 1'})
    odds = odds.reset_index(drop=False)
    # st.write(odds)
    # st.write(model.classes_)
    # st.write(f'{model_type} has not yet been implemented')
    bernoulli_df = pd.merge(factors, odds, on='index', how='outer')
    bernoulli_df['odds_ratio'] = bernoulli_df['Odds of Class 1']/bernoulli_df['Odds of Class 0']
    bernoulli_df.sort_values(by='odds_ratio', inplace=True, ascending=False)
    # st.write(bernoulli_df)
    top_ten = bernoulli_df.head(10)
    bottom_ten = bernoulli_df.tail(10)
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.bar(top_ten.factor, top_ten.odds_ratio)
    plt.title(f"Top ten associated with {y_col}")
    plt.xticks(rotation=70, ha='right')
    st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.bar(bottom_ten.factor, bottom_ten.odds_ratio)
    plt.title(f"Top ten associated with absence of {y_col}")
    plt.xticks(rotation=70, ha='right')
    st.pyplot(fig)

elif model_type == 'LogisticRegression':
    # st.write(combined_df)
    # potential_y = ['afib', 'PACs', 'day_before_afib']
    y_col = st.sidebar.selectbox('Choose a y', potential_y)
    X_cols = combined_df.columns.tolist()
    X_cols = [x for x in X_cols if x not in (potential_y+['date'])]
    X = combined_df[X_cols]
    y = combined_df[y_col]
    # split into training and testing
    test_fraction = (st.sidebar.slider('Select % of data for testing',
                                       min_value=5, max_value=50, value=10))/100
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(test_fraction))

    model = LogisticRegression()
    y_pred = model.fit(X_train, y_train).predict(X_test)
    st.write(
        f'There are {(y_test != y_pred).sum()} missed out of {X_test.shape[0]} in the testing set from {combined_df.shape[0]} rows of data for accuracy of {round(1-((y_test != y_pred).sum()/X_test.shape[0]),2)}')

    importance = model.coef_

    positive_factors = {}
    negative_factors = {}
    for i in range(len(importance[0])):
        if importance[0][i] > .001:
            positive_factors[X_train.columns[i]] = importance[0][i]

    for i in range(len(importance[0])):
        if importance[0][i] < -.0001:
            negative_factors[X_train.columns[i]] = importance[0][i]

    positive_df = pd.DataFrame.from_dict(positive_factors, orient='index')
    if positive_df.shape[0] > 0:
        positive_df.columns = ['coefficient']
        positive_df = positive_df.sort_values('coefficient', ascending=False)
        if positive_df.shape[0] > 10:
            positive_df = positive_df.head(10)

        fig, ax = plt.subplots(figsize=(15, 4))
        plt.bar(positive_df.index, positive_df.coefficient)
        plt.xticks(rotation=70, ha='right', fontsize=20, fontweight=10)
        plt.title(f'Largest 10 factors contributing to production of {y_col}')
        st.pyplot(fig)

    negative_df = pd.DataFrame.from_dict(negative_factors, orient='index')
    # st.write(negative_df)
    negative_df.columns = ['coefficient']
    negative_df = negative_df.sort_values('coefficient')
    if negative_df.shape[0] > 10:
        negative_df = negative_df.head(10)

    fig, ax = plt.subplots(figsize=(15, 4))
    plt.bar(negative_df.index, negative_df.coefficient)
    plt.xticks(rotation=70, ha='right', fontsize=20, fontweight=10)
    plt.title(f'Largest 10 factors contributing to avoidance of {y_col}')
    st.pyplot(fig)

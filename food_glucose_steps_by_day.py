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
####################################################

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
# load pac/afib data
ekg_df = pd.read_csv(ekg_file)
glucose_df = pd.read_csv(glucose_file)

# load watch data
watch_df = pd.read_csv(watch_path+watch_file)
watch_df = watch_df.drop(['type', 'unit'], axis=1)
watch_df.rename(columns={'value': 'steps'}, inplace=True)

combined_df = ekg_df.merge(glucose_df, on='date', how='outer')
combined_df = combined_df.merge(watch_df, on='date', how='outer')
combined_df.date = pd.to_datetime(combined_df.date)
combined_df.sort_values(by='date', inplace=True)
combined_df = combined_df.dropna()

columns = combined_df.columns
columns = [x for x in columns if x != 'date']
for col in columns:
    combined_df[col] = combined_df[col].astype(int)

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
# st.write(oura_df)


combined_df['day_before_afib'] = combined_df.afib.shift(-1)
combined_df.dropna(inplace=True)
combined_df.day_before_afib = combined_df.day_before_afib.astype(int)
combined_df = pd.merge(combined_df, oura_df, on='date', how='outer')
combined_df.dropna(inplace=True)

potential_y = ['afib', 'PACs', 'day_before_afib']
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

models = ['GaussianNB', 'BernoulliNB', 'LinearRegression', 'LogisticRegression']
model_type = st.sidebar.selectbox('Select a model', models)
if model_type == 'GaussianNB':
    model = GaussianNB()
    y_pred = model.fit(X_train, y_train).predict(X_test)
    st.write(model.class_prior_)
    odds_df = pd.DataFrame(model.theta_, columns=model.feature_names_in_)
    odds_df = odds_df.replace({0: np.nan})
    odds_df = odds_df.dropna(axis=1)
    for col in odds_df.columns:
        # odds_df[col] = odds_df[col].astype(float)
        odds_df.loc[3, col] = odds_df.loc[1, col]/odds_df.loc[0, col]

    # .dropna()
    st.write(odds_df)
    st.write(
        f'There are {(y_test != y_pred).sum()} missed out of {X_test.shape[0]} in the testing set from {combined_df.shape[0]} rows of data')
    st.write(f'{model_type} has not yet been implemented')
    st.write('This likely also requires encoding values to zero or 1')
elif model_type == 'BernoulliNB':  # before this will work need to change all factors to 0 and 1
    model = BernoulliNB()
    y_pred = model.fit(X_train, y_train).predict(X_test)
    st.write(
        f'There are {(y_test != y_pred).sum()} missed out of {X_test.shape[0]} in the testing set from {combined_df.shape[0]} rows of data')
    st.write(model.n_features_in_)
    st.write(model.feature_names_in_)
    st.write(model.feature_log_prob_)
    st.write(model.classes_)
    st.write(f'{model_type} has not yet been implemented')
elif model_type == 'LinearRegression':
    model = LinearRegression()
    y_pred = model.fit(X_train, y_train).predict(X_test)
    st.write(
        f'There are {(y_test != y_pred).sum()} missed out of {X_test.shape[0]} in the testing set from {combined_df.shape[0]} rows of data')
    st.write(f'{model_type} has not yet been implemented')
elif model_type == 'LogisticRegression':
    model = LogisticRegression()
    y_pred = model.fit(X_train, y_train).predict(X_test)
    st.write(
        f'There are {(y_test != y_pred).sum()} missed out of {X_test.shape[0]} in the testing set from {combined_df.shape[0]} rows of data')

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
    negative_df.columns = ['coefficient']
    negative_df = negative_df.sort_values('coefficient')
    if negative_df.shape[0] > 10:
        negative_df = negative_df.head(10)

    fig, ax = plt.subplots(figsize=(15, 4))
    plt.bar(negative_df.index, negative_df.coefficient)
    plt.xticks(rotation=70, ha='right', fontsize=20, fontweight=10)
    plt.title(f'Largest 10 factors contributing to avoidance of {y_col}')
    st.pyplot(fig)

<<<<<<< HEAD
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
import numpy as np
# from sklearn.metrics import accuracy_score
# from sklearn.datasets import make_classification
# from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
=======
import pandas as pd
>>>>>>> 6a75303bf571345a944acaf0ec6b1352eeb654a9

ekg_file = 'EKG_by_day.csv'
freestyle_file = 'freestyle_by_day.csv'
ekg = pd.read_csv(ekg_file)
freestyle = pd.read_csv(freestyle_file)

ekg.date = pd.to_datetime(ekg.date)
ekg.sort_values(by='date')
freestyle.date = pd.to_datetime(freestyle.date)
freestyle.sort_values(by='date')
<<<<<<< HEAD

df = pd.merge(ekg, freestyle, on='date', how='outer')
df = df.dropna()
st.set_page_config(layout="wide")
st.write(f'There are {df.afib.sum()} days with afib')
=======
print(freestyle)
print(ekg)
df = pd.merge(ekg, freestyle, on='date', how='outer')
df = df.dropna()
>>>>>>> 6a75303bf571345a944acaf0ec6b1352eeb654a9

cols = df.columns
floats = ['max_glucose', 'mean_glucose']
times = ['date']
ints = [x for x in cols if x not in floats+times]
for col in ints:
    df[col] = df[col].astype(int)
<<<<<<< HEAD
for col in floats:
    df[col] = df[col].astype(float)
for col in times:
    df[col] = pd.to_datetime(df[col])

plot_df = df.copy()
coffee = 'coffee' in df.columns
# st.write(df.coffee.sum())
# st.write(f'coffee {coffee}')
# st.write(plot_df)
plot_df['day_before_afib'] = plot_df.afib.shift(-1)
plot_df['day_after_afib'] = plot_df.afib.shift(1)
plot_df['two_days_before'] = plot_df.afib.shift(-2)
# st.write(plot_df)
plot_df.dropna(inplace=True)
plot_df.day_before_afib = plot_df.day_before_afib.astype(int)
plot_df.day_after_afib = plot_df.day_after_afib.astype(int)
# st.write(plot_df)
# st.write('Variable', '\t', '+mean', '\t', '-mean')
# st.write(f'Afib on days after {col}')

# get odds ratios on individual foods
# st.write('Odds ratio of foods on day before afib')
minimum_n = st.sidebar.slider('Min N', min_value=1, max_value=100, value=5)


for col in ['two_days_before', 'day_before_afib', 'afib', 'day_after_afib']:
    # for col in ['day_before_afib', 'afib', 'day_after_afib']:
    st.write(col)

    x = []
    y = []
    pos_afib = plot_df[plot_df[col] == 1]
    neg_afib = plot_df[plot_df[col] == 0]
    for colum in ints:
        n = df[colum].sum()
        pos_mean = pos_afib[colum].mean()
        neg_mean = neg_afib[colum].mean()
        if neg_mean == 0:
            neg_mean = .000001
        odds_ratio = pos_mean/neg_mean
        if (odds_ratio > 0) and (n >= minimum_n):
            x.append(colum)
            y.append(round(pos_mean/neg_mean, 2))
    fig, ax = plt.subplots(figsize=(15, 4))
    plt.bar(x, y)
    plt.xticks(rotation=70)
    left, right = plt.xlim()
    plt.hlines(1, xmin=left, xmax=right, color='r', linestyles='--')
    plt.title(f'Odds Ratio of Foods Eaten the {col}')
    st.pyplot(fig)


pos_afib = plot_df[plot_df.afib == 1]
neg_afib = plot_df[plot_df.afib == 0]
pos_max = pos_afib.max_glucose.mean()
pos_std = pos_afib.max_glucose.std()
neg_max = neg_afib.max_glucose.mean()
neg_std = neg_afib.max_glucose.std()
st.write(f'The mean max glucose on days with afib is {pos_max} with a sd of {pos_std}\
while the max glucose on days without afib is {neg_max} with a sd of {neg_std}')

# for col in ints:
#     fig, ax = plt.subplots()
#     x = plot_df[col]
#     y = plot_df.PACs
#     sns.regplot(x, y)
#     st.pyplot(fig)

st.write(df)
# st.write(df.shape)
# df.reset_index(inplace=True, drop=True)
# X = df.iloc[:, 3::]
# y = df.iloc[:, 1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
# model = LogisticRegression()
# y_pred = model.fit(X_train, y_train).predict(X_test)
# st.write(f'There are {(y_test != y_pred).sum()} missed out of {X_test.shape[0]}')
=======

print(df)
>>>>>>> 6a75303bf571345a944acaf0ec6b1352eeb654a9

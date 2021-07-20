from math import sqrt
import os
import bls
import pandas as pd
import numpy as np
from fredapi import Fred
import quandl
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import VAR
from sklearn.preprocessing import RobustScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa import seasonal
from statsmodels.tsa import filters
from scipy import signal
from statsmodels.tsa.seasonal import STL
# from stldecompose import decompose, forecast
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from statsmodels.api import Logit
from pingouin import corr, partial_corr
import xlsxwriter
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, distance
from scipy.cluster import hierarchy
from scipy.stats import ttest_ind as ttest
from sklearn.decomposition import PCA
from pandas.plotting import register_matplotlib_converters
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler, RobustScaler, KBinsDiscretizer
from scipy.stats import norm, skewnorm, skew
from matplotlib import dates as mdates
from pingouin import partial_corr, corr
from sklearn.decomposition import PCA
from pyppca import ppca
import plotly.graph_objects as go
import warnings
import time
from beepy import beep
from datetime import date, datetime
from matplotlib.dates import DateFormatter
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
sns.set()
#%% To fix the date fuck-ups
old_epoch = '0000-12-31T00:00:00'
new_epoch = '1970-01-01T00:00:00'
mdates.set_epoch(old_epoch)
plt.rcParams['date.epoch'] = '000-12-31'
plt.rcParams['axes.facecolor'] = 'w'
plt.style.use('seaborn')
register_matplotlib_converters()
#%% Clock utility function
def TicTocGenerator():
    ti = 0
    tf = time.time()
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti
TicToc = TicTocGenerator()
def toc(tempBool=True):
    tempTimeInterval = next(TicToc)
    if tempBool:
        print("Elapsed time: %f seconds.\n" % tempTimeInterval)
def tic():
    toc(False)

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, 'r:')

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y'], str(point['val']))
# %% Get BLS series id
os.chdir('/Users/anusarfarooqui/Docs/Matlab/Inflation')
url = 'https://download.bls.gov/pub/time.series/cu/cu.series'
Series = pd.read_csv(url, header=0, engine='python', sep='\t', skipinitialspace=True)
Series.columns = Series.columns.str.rstrip()
idx = np.logical_and(np.logical_and(Series.area_code=='0000', Series.seasonal=='S'), np.logical_and(Series.periodicity_code=='R', Series.base_code=='S'))
Series = Series[idx]
Series = Series[['series_id', 'series_title', 'begin_year', 'end_year']]
print(Series)
Series.to_excel('BLS_Price_Series.xlsx')
CPI = Series[np.logical_and(Series.series_title.str.contains('All items'), Series.begin_year<1996)]
Detail = Series[np.logical_and(np.logical_not(Series.series_title.str.contains('All items')), Series.begin_year<1996)]
# %% Get detailed price series
bls_api = 'b7cb522d08ac4e51ba17454b6311cfb2'

df = pd.DataFrame()
for i in range(len(Detail)):
    tic()
    print(i, 'of 179')
    df[Detail.series_id.iloc[i].strip()] = bls.get_series(series=Detail.series_id.iloc[i].strip(), startyear=1995, endyear=2021, key=bls_api).asfreq('M')
    toc()
df.to_pickle('detailed_price_series.pkl')
# %% We have 174 series
df = pd.read_pickle('detailed_price_series.pkl')
prices = np.log(df).diff(periods=12)
prices.to_excel('detailed_prices.xlsx')
price_volatility = prices.std()
# %% Get cpi
cpi = bls.get_series(series='CUSR0000SA0', startyear=1995, endyear=2021, key=bls_api).asfreq('M')
cpi = pd.DataFrame({'cpi': np.log(cpi).diff(periods=12), 'Month': cpi.index.to_timestamp()})
cpi['fwd'] = cpi.cpi.shift(periods=-12)
cpi.to_excel('cpi.xlsx')
wtd_avg = np.ma.average(np.ma.masked_array(prices, mask=prices.isna()), axis=1, weights=price_volatility**-1)
q_ = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
vol_quintile = pd.qcut(price_volatility, 5, labels=q_)
q_price = pd.DataFrame({'Q1': prices[prices.columns[vol_quintile == 'Q1']].mean(axis=1),
                        'Q2': prices[prices.columns[vol_quintile == 'Q2']].mean(axis=1),
                        'Q3': prices[prices.columns[vol_quintile == 'Q3']].mean(axis=1),
                        'Q4': prices[prices.columns[vol_quintile == 'Q4']].mean(axis=1),
                        'Q5': prices[prices.columns[vol_quintile == 'Q5']].mean(axis=1),
                        'wtd_avg': wtd_avg})
q_price['Month'] = q_price.index.to_timestamp()
#%% panel plot of volatility quintile
plt.figure(dpi=300)
for x in q_:
    sns.lineplot(x='Month', y=x, data=q_price)
plt.legend(q_)
plt.title('Inflation conditional on volatility of price series')
plt.show()

labels = np.array([['Q1', 'Q2'], ['Q3', 'Q4']])

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(16, 10), dpi=400)
for i in range(2):
    for j in range(2):
        sns.lineplot(x='Month', y=labels[i, j], data=q_price, ax=ax[i, j])
plt.show()
# %% Merge tables
data = q_price.merge(cpi, how='inner', left_on='Month', right_on='Month')
# %% Fit models
plt.figure(dpi=300)
sns.lineplot(x='Month', y='cpi', data=data)
sns.lineplot(x='Month', y='Q1', data=data)
plt.legend(['CPI', 'Q1'])
plt.show()

model = smf.ols(formula='fwd~wtd_avg', missing='drop', data=data).fit()
print(model.summary())

model = smf.ols(formula='fwd~Q1+cpi', missing='drop', data=data).fit()
print(model.summary())

model = smf.ols(formula='fwd~Q2+cpi', missing='drop', data=data).fit()
print(model.summary())

model = smf.ols(formula='fwd~Q3+cpi', missing='drop', data=data).fit()
print(model.summary())

model = smf.ols(formula='fwd~Q4+cpi', missing='drop', data=data).fit()
print(model.summary())

model = smf.ols(formula='fwd~Q5+cpi', missing='drop', data=data).fit()
print(model.summary())

model = smf.ols(formula='fwd~Q1+Q2+Q3+Q4+Q5', missing='drop', data=data).fit()
print(model.summary())
print(sm.stats.anova_lm(model, robust='HC3'))

labels = np.array([['Q1', 'Q2'], ['Q3', 'Q4']])

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(16, 10), dpi=400)
for i in range(2):
    for j in range(2):
        sns.regplot(x=labels[i, j], y='fwd', data=data, ax=ax[i, j])
plt.show()
# %% Q1 probability
plt.figure(dpi=400)
sns.lineplot(x='Month', y='Q1', data=q_price)
plt.title('Lowest volatility quintile of CPI')
plt.show()
# %%
mpd = pd.read_excel('mpd.xlsx', sheet_name='stats', header=3)
market_prob = mpd[mpd.market=='infl5y']
market_prob = market_prob[['idt', 'prInc', 'prDec']]
market_prob.set_index(keys='idt', inplace=True)
market_prob = market_prob.asfreq(freq='W', method='pad')
# %%
plt.figure(dpi=400)
sns.lineplot(x=market_prob.index, y=market_prob.prInc)
plt.title('Prob(CPI>3% over 5 years)')
plt.show()
# %% Persistence
b = np.empty(shape=len(q_))
StdErr = np.empty(shape=len(q_))
for i in range(len(q_)):
    model = sm.OLS(endog=data[q_[i]], exog=sm.add_constant(data[q_[i]].shift(1)), missing='drop').fit()
    b[i] = model.params[-1]
    StdErr[i] = model.bse[-1]

plt.figure(dpi=300)
plt.errorbar(x=q_, y=b, yerr=StdErr)
plt.title('Persistence parameters by volatility quintile of CPI series')
plt.ylabel('AR(1) slope')
plt.show()
# %%
os.chdir('/Users/anusarfarooqui/Docs/Matlab/Inflation')
df = pd.read_pickle('detailed_price_series.pkl')
prices = np.log(df).diff(periods=12)

r = np.empty(shape=(len(prices.columns), len(prices.columns)), dtype='float64')
rho = np.empty(shape=(len(prices.columns), len(prices.columns)), dtype='float64')

lagged_prices = prices.shift(1)

for i in range(len(prices.columns)):
    tic()
    print(i, 'of 179')
    for j in range(len(prices.columns)):
        r[i, j] = corr(x=prices.iloc[:, i], y=prices.iloc[:, j], method='spearman')['r'].values[0]
        rho[i, j] = corr(x=lagged_prices.iloc[:, i], y=prices.iloc[:, j], method='spearman')['r'].values[0]
    toc()

corr_data = pd.DataFrame(r, index=prices.columns, columns=prices.columns)
corr_data.to_pickle('corr_data.pkl')
price_volatility = prices.std(axis=0)
prices.to_pickle('prices.pkl')
# %% scatter
os.chdir('/Users/anusarfarooqui/Docs/Matlab/Inflation')
prices = pd.read_pickle('prices.pkl')
corr_data = pd.read_pickle('corr_data.pkl')
price_volatility = prices.std(axis=0)

# plt.figure(dpi=400)
# sns.scatterplot(x=np.mean(corr_data, axis=1), y=price_volatility)
# plt.show()
wts = pd.Series(np.mean(corr_data, axis=1) * price_volatility.values ** -1, index=price_volatility.index)
wts.where(cond=wts>0)
wtd_avg = np.ma.average(np.ma.masked_array(prices, mask=prices.isna()), axis=1, weights=wts)
wtd = pd.DataFrame({'supercore': wtd_avg}, index=prices.index)
wtd['Month'] = wtd.index.to_timestamp()
wtd['headline'] = cpi.cpi
wtd_long = wtd.melt(id_vars='Month', var_name='Type', value_name='Inflation')
# %%
plt.figure(dpi=400)
sns.lineplot(x='Month', y='Inflation', hue='Type', palette='pastel', data=wtd_long)
plt.title('Supercore consumer price inflation')
plt.ylabel('annual rate')
plt.show()

plt.figure(dpi=400)
sns.lineplot(x='Month', y='supercore', data=wtd)
plt.show()
# %% Double cut
p_ = ['low', 'mid', 'high']
vol_q = pd.qcut(price_volatility, 3, labels=p_)
corr_q = pd.qcut(np.mean(corr_data, axis=1), 3, labels=p_)

data = pd.DataFrame()
for p in p_:
    for q in p_:
        data[p + '-' + q] = prices[prices.columns[np.logical_and(vol_q==p, corr_q==q)]].mean(axis=1)

plt.figure(dpi=300)
sns.heatmap(data.corr())
plt.show()


# %% Correlation quintiles
ax = 1
q_ = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
pred_quintile = pd.qcut(np.mean(rho, axis=ax), 5, labels=q_)


q_ = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

q_price = pd.DataFrame({'Q1': prices[prices.columns[pred_quintile == 'Q1']].mean(axis=ax),
                        'Q2': prices[prices.columns[pred_quintile == 'Q2']].mean(axis=ax),
                        'Q3': prices[prices.columns[pred_quintile == 'Q3']].mean(axis=ax),
                        'Q4': prices[prices.columns[pred_quintile == 'Q4']].mean(axis=ax),
                        'Q5': prices[prices.columns[pred_quintile == 'Q5']].mean(axis=ax),
                        'wtd_avg': wtd_avg})
q_price['Month'] = q_price.index.to_timestamp()

plt.figure(dpi=400)
sns.lineplot(x='Month', y='wtd_avg', data=q_price)
sns.lineplot(x='Month', y='Q5', data=q_price)
plt.legend(['Wtd', 'Q5'])
plt.show()

plt.figure(dpi=400)
for x in q_:
    sns.lineplot(x='Month', y=x, data=q_price)
plt.legend(q_)
plt.show()
# %%
data = q_price.merge(cpi, how='inner', left_on='Month', right_on='Month')

model = smf.ols(formula='fwd~wtd_avg+cpi', missing='drop', data=data).fit()
result = model.get_robustcov_results(cov_type='HC1', use_t=True)
print(result.summary())

for x in q_:
    model = smf.ols(formula='fwd~cpi+' + x, missing='drop', data=data).fit()
    print(model.summary())
# %%
C, ss, M, X, Ye = ppca(Y=prices.to_numpy(), d=5, dia=False)

plt.figure(dpi=300)
plt.plot(X[:, 0])
plt.show()

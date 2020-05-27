import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm
import gc

monthly_rtn = pd.read_csv('./monthly_rtn.csv', index_col=0, parse_dates=True)
monthly_rtn = monthly_rtn.loc['1961':'2019']
max_val = pd.read_csv('./max_factor.csv', index_col=0, parse_dates=True)
beta = pd.read_csv('./dimson_fac.csv', index_col=0, parse_dates=True)
size = np.log(pd.read_csv('./market_equity.csv', index_col=0, parse_dates=True).replace(0, np.nan)).shift(1)
bm = pd.read_csv('./book_to_market.csv', index_col=0, parse_dates=True)
mom = pd.read_csv('./mom_fac.csv', index_col=0, parse_dates=True).shift(1)
rev = pd.read_csv('./monthly_rtn.csv', index_col=0, parse_dates=True).shift(1)
illiq = pd.read_csv('./illiq_fac.csv', index_col=0, parse_dates=True)
ivol = pd.read_csv('./idio_fac.csv', index_col=0, parse_dates=True)

key = set(monthly_rtn.columns) & set(max_val.columns) & set(beta.columns) & set(size.columns) & set(bm.columns) & \
      set(mom.columns) & set(rev.columns) & set(rev.columns) & set(illiq.columns) & set(ivol.columns)
key = list(key)
key.sort()
max_lag = max_val.shift(1)
monthly_rtn = monthly_rtn[key].loc['1962':'2019']
max_lag = max_lag[key].loc['1962':'2019']
max_val = max_val[key].loc['1962':'2019']
beta = beta[key].loc['1962':'2019']
size = size[key].loc['1962':'2019']
bm = bm[key].loc['1962':'2019']
mom = mom[key].loc['1962':'2019']
rev = rev[key].loc['1962':'2019']
illiq = illiq[key].loc['1962':'2019']
ivol = ivol[key].loc['1962':'2019']

result = pd.DataFrame(data=0, index=range(1, 10), columns=['MAX', 'BETA', 'SIZE', 'BM', 'MOM', 'REV', 'ILLIQ', 'IVOL', 'R^2'])
tvalue = pd.DataFrame(data=0, index=range(1, 10), columns=['MAX', 'BETA', 'SIZE', 'BM', 'MOM', 'REV', 'ILLIQ', 'IVOL', 'R^2'])


x_list = [max_lag, beta, size, bm, mom, rev, illiq, ivol]

for m in tqdm(range(len(max_val.index))):
    tmp_Y = max_val.iloc[m].dropna().astype('float')
    tmp_X_list = [x.iloc[m].dropna().astype('float') for x in x_list]
    tmp_key_list = [set(x.index) for x in tmp_X_list]
    tmp_key = list(set(tmp_Y.index).intersection(*tmp_key_list))
    tmp_key.sort()

    tmp_Y = tmp_Y.loc[tmp_key]
    tmp_X_list = [x.loc[tmp_key] for x in tmp_X_list]

    for i in range(len(tmp_X_list)):
        model = sm.OLS(tmp_Y, sm.add_constant(tmp_X_list[i])).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
        result.iloc[i, i] += model.params.values[1]
        tvalue.iloc[i, i] += model.tvalues[1]
        result.iloc[i, -1] += model.rsquared

    tmp_total_X = pd.concat(tmp_X_list, axis=1)
    tmp_total_X.columns = ['MAX', 'BETA', 'SIZE', 'BM', 'MOM', 'REV', 'ILLIQ', 'IVOL']
    tmp_total_X = sm.add_constant(tmp_total_X)

    model = sm.OLS(tmp_Y, tmp_total_X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    result.iloc[-1, :-1] += model.params.values[1:]
    tvalue.iloc[-1, :-1] += model.tvalues[1:]
    result.iloc[-1, -1] += model.rsquared
    gc.collect()

result /= len(max_val)
tvalue /= len(max_val)

result2 = (result.replace(0, np.nan)).round(4).astype('str') + tvalue.replace(0, np.nan).round(2).astype('str').applymap(lambda x: '('+x+')')

result2.to_csv('./table2.csv')
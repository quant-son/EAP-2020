import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm

monthly_rtn = pd.read_csv('./monthly_rtn.csv', index_col=0, parse_dates=True)
monthly_rtn = monthly_rtn.loc['1961':'2019']
max_val = pd.read_csv('./max_factor.csv', index_col=0, parse_dates=True)
max_val = max_val[monthly_rtn.columns]
max_val = max_val.loc['1961':'2019']
market_equity = pd.read_csv('./market_equity.csv', index_col=0, parse_dates=True)
market_equity = market_equity[monthly_rtn.columns]
market_equity = market_equity.loc['1961':'2019']

ff3_monthly = pd.read_csv('./ff3_monthly.CSV', index_col=0)
mom_monthly = pd.read_csv('./mom_monthly.CSV', index_col=0)
mom_monthly.index = pd.to_datetime(mom_monthly.index, format='%Y%m')
ff3_monthly.index = pd.to_datetime(ff3_monthly.index, format='%Y%m')
monthly_rtn.index = monthly_rtn.index
market_equity.index = market_equity.index

ff4_monthly = pd.concat([ff3_monthly.loc['1961':'2019'], mom_monthly.loc['1961':'2019']], axis=1)
ff4_monthly /= 100

monthly_rtn.index = monthly_rtn.index.strftime('%Y-%m')
ff4_monthly.index = ff4_monthly.index.strftime('%Y-%m')
market_equity.index = market_equity.index.strftime('%Y-%m')
max_val.index = max_val.index.strftime('%Y-%m')

max_crit = max_val.shift(1).dropna(how='all', axis=0).replace(0, np.nan)
max_port_vw = pd.DataFrame(index=max_crit.index, columns=range(1,11))
max_port_ew = pd.DataFrame(index=max_crit.index, columns=range(1,11))
decile_info = pd.DataFrame(index=max_crit.index, columns=max_crit.columns)

for m in tqdm(max_crit.index):
    crit = pd.qcut(max_crit.loc[m].dropna(), 10, labels=range(1, 11))
    decile_info.loc[m] = crit

    for d in range(1, 11):
        tic_list = crit[crit == d].index
        tmp_rtn = monthly_rtn.loc[m, tic_list].dropna()
        tmp_mv = market_equity.loc[m, tic_list].dropna()

        if len(tmp_rtn.index) != len(tmp_mv.index):
            new_tic_list = list(set(tmp_rtn.index) & set(tmp_mv.index))
            new_tic_list.sort()
            tmp_rtn = tmp_rtn.loc[new_tic_list]
            tmp_mv = tmp_mv.loc[new_tic_list]

        tmp_vw = tmp_mv/tmp_mv.sum()
        tmp_eq = tmp_mv.copy()
        tmp_eq[:] = 1
        tmp_ew = tmp_eq/tmp_eq.sum()
        max_port_vw.loc[m, d] = np.dot(tmp_rtn, tmp_vw)
        max_port_ew.loc[m, d] = np.dot(tmp_rtn, tmp_ew)

max_port_vw = max_port_vw.loc['1962':]
max_port_ew = max_port_ew.loc['1962':]

max_port_vw['10-1'] = max_port_vw[10]-max_port_vw[1]
max_port_ew['10-1'] = max_port_ew[10]-max_port_ew[1]
max_port_vw.columns = [str(x) for x in max_port_vw.columns]
max_port_ew.columns = [str(x) for x in max_port_ew.columns]

ff4_monthly = ff4_monthly.loc['1962':].astype('float')

result = pd.DataFrame(index=max_port_vw.columns, columns=['average-vw', 'alpha-vw', 'average-ew', 'alpha-ew'])

for d in tqdm(max_port_vw.columns):
    Y1 = max_port_vw[d].astype('float')
    Y2 = max_port_ew[d].astype('float')
    result.loc[d, 'average-vw'] = np.round(Y1.mean(),4)*100
    result.loc[d, 'average-ew'] = np.round(Y2.mean(),4)*100
    X = sm.add_constant(ff4_monthly)

    model1 = sm.OLS(Y1, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    model2 = sm.OLS(Y2, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})

    alpha1 = str((model1.params['const']*100).round(2)) + '(' + str(model1.tvalues['const'].round(2)) + ')'
    alpha2 = str((model2.params['const']*100).round(2)) + '(' + str(model2.tvalues['const'].round(2)) + ')'

    result.loc[d, 'alpha-vw'] = alpha1
    result.loc[d, 'alpha-ew'] = alpha2

average_max = pd.DataFrame(data=0, index=[int(x) for x in max_port_vw.columns[:-1]], columns=['sum', 'count'])
decile_info = decile_info.loc['1962':]

for i in tqdm(range(len(decile_info))):

    tmp_decile = decile_info.iloc[i].dropna()
    average_max['count'] += tmp_decile.groupby(tmp_decile.values).count()
    tmp_max = max_crit.loc[tmp_decile.name, tmp_decile.index]
    tot = pd.concat([tmp_decile, tmp_max], axis=1)
    tot.columns = ['d', 'max']
    average_max['sum'] += tot.groupby('d')['max'].sum()

average_max['average'] = average_max['sum']/average_max['count']

average_max['average'] *= 100
average_max['average'] = average_max['average'].round(2)

result.loc[[str(x) for x in average_max.index], 'average_max'] = average_max['average'].values

Y1 = max_port_vw['10-1'].astype('float')
Y2 = max_port_ew['10-1'].astype('float')

from scipy.stats import ttest_1samp
result.loc['10-1', 'average-vw'] = str(result.loc['10-1', 'average-vw'])+ '('+str(ttest_1samp(Y1, 0)[0].round(2))+')'
result.loc['10-1', 'average-ew'] = str(result.loc['10-1', 'average-ew'])+'('+str(ttest_1samp(Y2, 0)[0].round(2))+')'
result.to_csv('./table1.csv')

import pandas as pd
import numpy as np
import gc
import statsmodels.api as sm
from multiprocessing import Manager, cpu_count, Queue
from tqdm import tqdm
import parmap

daily = pd.read_csv('./daily.csv')
daily['RET'] = daily['RET'].apply(pd.to_numeric, errors='coerce')
daily_rtn = daily.pivot_table(index='date', columns='PERMNO', values='RET')

#MAX
daily_rtn.index = pd.to_datetime(daily_rtn.index, format='%Y%m%d')
max_factor = daily_rtn.resample('M').apply(lambda x: x.max())
daily_rtn.to_csv('./daily_rtn.csv')
max_factor.to_csv('./max_factor.csv')


monthly_d = pd.read_csv('./monthly.csv')
monthly_d['ME'] = abs(monthly_d['PRC'])*monthly_d['SHROUT']
market_equity = monthly_d.pivot_table(index='date', columns='PERMNO', values='ME')
market_equity.index = pd.to_datetime(market_equity.index, format='%Y%m%d')
market_equity = market_equity.dropna(how='all', axis=0).dropna(how='all', axis=1)

monthly_d = pd.read_csv('./monthly.csv')
monthly_d['DLRET'] = monthly_d['DLRET'].apply(pd.to_numeric, errors='coerce')
monthly_d['RET'] = monthly_d['RET'].apply(pd.to_numeric, errors='coerce')

for i in tqdm(monthly_d.index):
    if np.isnan(monthly_d.loc[i, 'RET']) and not np.isnan(monthly_d.loc[i, 'DLRET']) and not np.isnan(monthly_d.loc[i, 'DLSTCD']):
        monthly_d.loc[i, 'RET'] = monthly_d.loc[i, 'DLRET']

monthly_rtn = monthly_d.pivot_table(index='date', columns='PERMNO', values='RET')
monthly_rtn.index = pd.to_datetime(monthly_rtn.index, format='%Y%m%d')
monthly_rtn.to_csv('./monthly_rtn.csv')

#TVOL
tvol = daily_rtn.resample('M').std()
tvol.to_csv('./tvol_factor.csv')

#IVOL
ff3_daily = pd.read_csv('./ff3_daily.CSV', index_col=0, parse_dates=True)
ff3_daily /= 100
daily_rtn = pd.read_csv('./daily_rtn.csv', index_col=0, parse_dates=True)
monthly_rtn = pd.read_csv('./monthly_rtn.csv', index_col=0, parse_dates=True)
daily_rtn = daily_rtn[monthly_rtn.columns]
index = pd.read_csv('./index_rtn.csv', index_col=0, parse_dates=True)
mkt_rf = index.loc['1961':].subtract(ff3_daily['RF'].loc['1961':], axis=0)
d_rf = daily_rtn.loc['1961':].subtract(ff3_daily['RF'].loc['1961':], axis=0)
d_rf = d_rf.dropna(how='all', axis=0).dropna(how='all', axis=1)

idio_y = d_rf.loc['1961-11':]
idio_x = mkt_rf.loc['1961-11':]

ind = np.arange(len(idio_y.columns))
ind_list = np.array_split(ind, 10)

mgr = Manager()
q = mgr.Queue()

from parallel_ols import idio_vol

parmap.map(idio_vol, ind_list, idio_y, idio_x, q, pm_pbar=True, pm_processes=10)
gc.collect()
result_l = []

while q.qsize():
    result_l.append(q.get())

idio_fac = pd.concat(result_l, axis=1)
idio_fac_sorted = idio_fac.T.sort_index().T
idio_fac_sorted.to_csv('./idio_fac.csv')
gc.collect()

#BETA
dimson_x = pd.concat([mkt_rf.shift(2), mkt_rf.shift(1), mkt_rf], axis=1)
dimson_x.columns = ['b1', 'b2', 'b3']
dimson_y = d_rf.shift(1)

dimson_y = dimson_y.loc['1961-11':]
dimson_x = dimson_x.loc['1961-11':]

ind = np.arange(len(dimson_y.columns))
ind_list = np.array_split(ind, 10)

mgr = Manager()
q = mgr.Queue()

from parallel_ols import dimson

parmap.map(dimson, ind_list, dimson_y, dimson_x, q, pm_pbar=True, pm_processes=10)

result_l = []

while q.qsize():
    result_l.append(q.get())

dimson_fac = pd.concat(result_l, axis=1)
dimson_fac.to_csv('./dimson_fac.csv')


#make SIZE, BM, MOM, REV, ILLIQ from monthly data

#size
monthly_d = pd.read_csv('./monthly.csv')
monthly_d['ME'] = abs(monthly_d['PRC'])*monthly_d['SHROUT'].replace(0, np.nan)
market_equity = monthly_d.pivot_table(index='date', columns='PERMNO', values='ME')
market_equity.index = pd.to_datetime(market_equity.index, format='%Y%m%d')
market_equity = market_equity.dropna(how='all', axis=0).dropna(how='all', axis=1)
market_equity.to_csv('./market_equity.csv')

#BM
account = pd.read_csv('./account.csv')
linked_table = pd.read_csv('./linked_table2.csv')
account = account[account['gvkey'].isin(linked_table['gvkey'].unique())]
account['gvkey'] = account['gvkey'].replace(linked_table['gvkey'].values, linked_table['PERMNO'].values)
account['book_value'] = account['ceq'] + account['txditc'].replace(np.nan, 0)
book_value = account.pivot_table(index='datadate', columns='gvkey', values='book_value')
book_value.index = pd.to_datetime(book_value.index, format='%Y%m%d')
book_value.index = book_value.index.strftime('%Y-%m')

book_value_f = book_value.fillna(method='ffill', limit=12)

market_equity = pd.read_csv('./market_equity.csv', index_col=0, parse_dates=True)
market_equity.index = market_equity.index.strftime('%Y-%m')
book_value = book_value.reindex(market_equity.index)
book_value_f = book_value.fillna(method='ffill', limit=12)

permno = list(set([str(x) for x in book_value_f.columns]) & set(market_equity.columns))
permno.sort()
market_equity = market_equity[permno]
book_value_f.columns = [str(x) for x in book_value_f.columns]
book_value_f = book_value_f[permno]

book_value_f.index = pd.to_datetime(book_value_f.index, format='%Y-%m') + pd.offsets.MonthEnd(0)
market_equity.index = pd.to_datetime(market_equity.index, format='%Y-%m') + pd.offsets.MonthEnd(0)
market_equity = market_equity.loc['1961':'2019']
book_value_f = book_value_f.loc['1961':'2019']

market_equity.loc[[x for x in market_equity.index if x.month != 12]] = np.nan
market_equity_decem = market_equity.shift(1).fillna(method='ffill', limit=11)
book_to_market = book_value_f / market_equity_decem.replace(0, np.nan)

#winsorizing
book_to_market = book_to_market.clip(np.nanpercentile(book_to_market.values, 0.5), np.nanpercentile(book_to_market.values, 99.5))
book_to_market.to_csv('./book_to_market.csv')

#MOM

monthly_rtn = pd.read_csv('./monthly_rtn.csv', index_col=0, parse_dates=True)
bm = pd.read_csv('./book_to_market.csv', index_col=0, parse_dates=True)
key = list(set(monthly_rtn.columns) & set(bm.columns))
key.sort()
monthly_rtn = monthly_rtn[key]
annual_rtn = monthly_rtn.rolling(12).apply(lambda x: (1+x).prod()-1)
mom = (1+annual_rtn)/(1+monthly_rtn).replace(0, np.nan) - 1
mom.to_csv('./mom_fac.csv')

#ILLIQ

monthly_d = pd.read_csv('./monthly.csv')
monthly_vol = monthly_d.pivot_table(index='date', columns='PERMNO', values='VOL')
monthly_vol.columns = [str(x) for x in monthly_vol.columns]
key = list(set(monthly_vol.columns)&set(bm.columns)&set(monthly_rtn.columns))
key.sort()
monthly_vol.index = pd.to_datetime(monthly_vol.index, format='%Y%m%d')
monthly_vol = monthly_vol[key]
monthly_rtn = monthly_rtn[key]
monthly_rtn.to_csv('./monthly_rtn.csv')
illiq = abs(monthly_rtn)/monthly_vol.replace(0, np.nan)
illiq.to_csv('./illiq_fac.csv')
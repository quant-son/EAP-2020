import pandas as pd
import numpy as np
import get_table

#자산가격실증연구 HW1 20203314 손창호


# 데이터 로드 및 nan값 제거
index = pd.read_csv('./index.csv',index_col=0,parse_dates=True)
#tbill = pd.read_csv('./fama-onem.csv',index_col=0,parse_dates=True).dropna(axis=0) #fama-bliss one month
tbill = pd.read_csv('./fama_3m.csv',index_col=0,parse_dates=True).dropna(axis=0) #fama-bliss three month
data_tot = pd.read_csv('./data_tot.csv',index_col='date',parse_dates=True).drop(['DISTCD'],axis=1).dropna(axis=0,how='all')
#dividend = pd.read_csv('./div.csv',index_col='EXDT',parse_dates=True).drop(['DISTCD'],axis=1).dropna(axis=0) # only cash dividend
#dividend = pd.read_csv('./stock_div.csv',index_col='EXDT',parse_dates=True).drop(['DISTCD'],axis=1).dropna(axis=0) # cash + stock dividend
#price = pd.read_csv('./data2.csv',index_col='date',parse_dates=True).drop(['ALTPRC','ALTPRCDT'],axis=1).dropna(axis=0)

# NYSE 주식 선별
data_tot['ME'] = abs(data_tot['PRC'])*data_tot['SHROUT']
data_tot['DIV'] = data_tot['DIVAMT'].replace(np.nan,0)*data_tot['SHROUT']
data_tot = data_tot[data_tot['HEXCD']==1]

#price = price[price['EXCHCD']==1].drop(['EXCHCD','PRIMEXCH'],axis=1)
#dividend = dividend[dividend['HEXCD']==1].drop('HEXCD',axis=1)
#price = price.drop(['EXCHCD','PRIMEXCH'],axis=1)
#dividend = dividend.drop('HEXCD',axis=1)

# date index에서 days 제거

index_m = index['vwindx'].copy()
tbill_m = tbill.copy()
#price_m = price.copy()
#dividend_m = dividend.copy()
data_tot_m = data_tot.copy()

index_m.index = index_m.index.strftime('%Y-%m')
tbill_m.index = tbill_m.index.strftime('%Y-%m')
#price_m.index = price.index.strftime('%Y-%m')
#dividend_m.index = dividend.index.strftime('%Y-%m')
data_tot_m.index = data_tot_m.index.strftime('%Y-%m')
#price_m.index.name = 'date'
#dividend_m.index.name = 'date'
data_tot_m.index.name = 'date'

tbill_m /= 100

#div_df = dividend_m.pivot_table(index=dividend_m.index,columns='PERMNO',values='DIVAMT')
div_df = data_tot_m.pivot_table(index=data_tot_m.index,columns='PERMNO',values='DIV').replace(0,np.nan)
#price_df = abs(price_m.pivot_table(index=price_m.index,columns='PERMNO',values='PRC').iloc[1:])
#share_df = price_m.pivot_table(index=price_m.index,columns='PERMNO',values='SHROUT').iloc[1:]*1000


#tmp_div = (div_df*share_df).sum(axis=1)

# 연별 총 배당지급액 계산
tmp_div = div_df.sum(axis=1)
tmp_div.index = pd.to_datetime(tmp_div.index)

tmp_div2 = tmp_div.resample('Y').sum().shift(1).replace(0,np.nan)
tmp_div2.index = tmp_div2.index.strftime('%Y-%m')

div_total = tmp_div2.reindex(div_df.index).fillna(method='bfill',limit=11).dropna()
market_value = (data_tot_m.pivot_table(index=data_tot_m.index,columns='PERMNO',values='ME')).sum(axis=1)
dps = np.log(div_total/market_value).dropna()

log_rtn = np.log(index_m.shift(1))-np.log(index_m)
log_rtn = log_rtn.dropna().iloc[12:]

#rrel 계산
rrel = (tbill_m-tbill_m.rolling(12).mean()).dropna().loc['1927':]
#rrel=tbill_m.loc['1927':]

df = pd.concat([log_rtn,dps,rrel],axis=1)
df.columns = ['h','d/p','rrel']
df.index = pd.to_datetime(df.index,format='%Y-%m')
df.index = df.index + pd.offsets.MonthEnd(0)


# table 1, 2 출력
table = get_table.Table(df)
print(table.tot_table1.to_string())
print(table.tot_table2.to_string())

# table 1, 2 저장
table.tot_table1.to_csv('./table_1.csv',encoding='cp949')
table.tot_table2.to_csv('./table_2.csv',encoding='cp949')

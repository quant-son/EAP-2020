import pandas as pd
import numpy as np

crsp = pd.read_csv('./CRSP_codes.csv')
compu = pd.read_csv('./Compustat_codes.csv')

num_of_cusip = crsp.groupby('PERMNO')['CUSIP'].apply(lambda x: len(x.unique()))
crsp['CUSIP'].apply(lambda x: len(x)).sort_values(ascending=False)

#only NYSE AMEX NASDAQ
crsp = crsp[crsp['EXCHCD'].isin([1, 2, 3, 31, 32, 33])]

#only common stocks
#crsp = crsp[crsp['SHRCD'].isin([10, 11, 12])]

compu['cusip'] = compu['cusip'].apply(lambda x: str(x)[:-1] if len(str(x)) == 9 else np.nan)

join_key = list(set(compu['cusip']) & set(crsp['CUSIP']))

crsp_new = crsp[crsp['CUSIP'].isin(join_key)]
compu_new = compu[compu['cusip'].isin(join_key)]
crsp_permno = crsp_new.set_index('CUSIP')['PERMNO'].drop_duplicates().sort_index()
compu_gvkey = compu_new.set_index('cusip')['gvkey'].drop_duplicates().sort_index()
crsp_permno.index.name = 'cusip'

linked_table = pd.concat([crsp_permno, compu_gvkey], axis=1)
linked_table.to_csv('./linked_table2.csv')

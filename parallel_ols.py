import numpy as np
from tqdm import tqdm
import statsmodels.api as sm
import gc


def dimson(ind, y, x, q):
    y = y.iloc[:, ind]
    dimson_beta = y.resample('M').count()
    dimson_beta[:] = np.nan

    for col in tqdm(y.columns):
        y2 = y[col].dropna()

        if len(y2) < 15:
            pass
        else:
            month_list = list(set(y2.index.strftime(date_format='%Y-%m')))
            month_list.sort()

            for d in month_list:
                Y = y2.loc[d]
                if len(Y) < 15:
                    pass
                else:
                    X = x.loc[Y.index]
                    X = sm.add_constant(X)
                    model = sm.OLS(Y, X).fit()
                    dimson_beta.loc[d, col] = model.params['b1'] + model.params['b2'] + model.params['b3']
                    del Y, X, model
                gc.collect()
    gc.collect()
    q.put_nowait(dimson_beta)


def idio_vol(ind, y, x, q):
    y = y.iloc[:, ind]
    idio_vol_df = y.resample('M').count()
    idio_vol_df[:] = np.nan

    for col in tqdm(y.columns):
        y2 = y[col].dropna()

        if len(y2) < 15:
            pass
        else:
            month_list = list(set(y2.index.strftime(date_format='%Y-%m')))
            month_list.sort()

            for d in month_list:
                Y = y2.loc[d]
                if len(Y) < 15:
                    pass
                else:
                    X = x.loc[Y.index]
                    X = sm.add_constant(X)
                    model = sm.OLS(Y, X).fit()
                    idio_vol_df.loc[d, col] = model.resid.std()
                    del Y, X, model
                gc.collect()

    gc.collect()
    q.put_nowait(idio_vol_df)

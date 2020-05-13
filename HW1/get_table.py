import pandas as pd
import numpy as np
import VAR_GMM
from RM import OLS


class Table:
    def __init__(self, df):
        self.data = df
        self.data_m = df
        self.data_q = df
        self.tot_table1 = pd.DataFrame()
        self.tot_table2 = pd.DataFrame()
        self.make()

    def make_table1(self, data):
        start_year = data.index[0].strftime('%Y-%m')
        end_year = data.index[-1].strftime('%Y-%m')
        title = start_year + ':' + end_year

        result_df = pd.DataFrame(index=[title, 'h_t+1','D/P_t+1','rrel_t+1'], columns=['h_t','D/P_t','rrel_t','R^2','Joint Significance'])
        result_df.loc[title, :] = ''
        var_gmm = VAR_GMM.VAR(data).fit(maxlags=1, method='gmm', trend='nc')
        params = var_gmm.params.T.applymap(lambda x: str(np.round(x,4)))
        stderrs = var_gmm.stderr.T.applymap(lambda x: '('+str(np.round(x,4))+')')
        print(params+stderrs)

        result_df.loc[['h_t+1','D/P_t+1','rrel_t+1'],['h_t','D/P_t','rrel_t']] = (params+stderrs).values

        ols1 = OLS(var_gmm.endog[1:,0],var_gmm.endog_lagged).fit(beta=var_gmm.params['h'].ravel(), df_model=var_gmm.df_model, df_resid=var_gmm.df_resid)
        ols2 = OLS(var_gmm.endog[1:,1],var_gmm.endog_lagged).fit(beta=var_gmm.params['d/p'].ravel(), df_model=var_gmm.df_model, df_resid=var_gmm.df_resid)
        ols3 = OLS(var_gmm.endog[1:,2],var_gmm.endog_lagged).fit(beta=var_gmm.params['rrel'].ravel(), df_model=var_gmm.df_model, df_resid=var_gmm.df_resid)

        tmp_df = pd.DataFrame([[ols1.rsquared,ols1.f_pvalue],[ols2.rsquared,ols2.f_pvalue],[ols3.rsquared,ols3.f_pvalue]]).applymap(lambda x: str(np.round(x,4)))
        result_df.loc[['h_t+1','D/P_t+1','rrel_t+1'],['R^2','Joint Significance']] = tmp_df.values

        return result_df

    def make_table2(self, data, lags, freq='monthly'):
        title = str(lags)+' lags, '+freq
        start_year = data.index[0].strftime('%Y-%m')
        end_year = data.index[-1].strftime('%Y-%m')
        title2 = start_year + ':' + end_year
        result_df = pd.DataFrame(index=[title,title2], columns=['R^2_h','Var(n_d)','Var(n_h)','-2cov(n_d,n_h)','corr(n_d,n_h)','P_h'])
        result_df.loc[title]=''
        var_gmm = VAR_GMM.VAR(data).fit(maxlags=lags, method='gmm', trend='nc')

        e_1 = np.array([1, 0, 0] + [0, 0, 0] * (lags - 1)).reshape(1, 3 * lags)

        rho = 0.9962
        if(freq == 'quarterly'):
            rho = rho**3

        if lags > 1:
            nob = len(data)
            y = data.iloc[lags:nob].values.T
            z = data.iloc[lags - 1:nob - 1].values.T

            for i in range(1, lags):
                y = np.vstack((y, data.iloc[lags - i:nob - i].values.T))
                z = np.vstack((z, data.iloc[lags - 1 - i:nob - 1 - i].values.T))

            A = np.vstack((var_gmm.params.T, np.kron(np.eye(lags)[:-1], np.eye(3))))
            rhoA = rho * A
            inv_rhoA = np.linalg.inv(np.eye(A.shape[0]) - rhoA)
            lamb = e_1 @ rhoA @ inv_rhoA

            resid = np.dot(np.vstack((np.eye(3), np.zeros(((lags - 1) * 3, 3)))), var_gmm.resid.T)

            eta_h = lamb @ resid
            eta_d = (e_1 + lamb) @ resid
            u_t = e_1 @ A @ resid
            p_h = eta_h.T.std(ddof=1) / u_t.T.std(ddof=1)
            ols1 = OLS((e_1@y).T, z.T).fit(beta=var_gmm.params['h'].ravel(),
                                                                       df_model=var_gmm.df_model,
                                                                       df_resid=var_gmm.df_resid)

        else:
            rhoA = rho * (var_gmm.params.T.values)
            inv_rhoA = np.linalg.inv(np.eye(3) - rhoA)
            lamb = e_1 @ rhoA @ inv_rhoA

            eta_h = lamb @ var_gmm.resid.T.values
            eta_d = (e_1 + lamb) @ var_gmm.resid.T.values
            u_t = e_1 @ (var_gmm.params.T.values) @ var_gmm.resid.T.values
            p_h = eta_h.T.std() / u_t.T.std()
            ols1 = OLS(var_gmm.endog[1:, 0], var_gmm.endog_lagged).fit(beta=var_gmm.params['h'].ravel(),
                                                                       df_model=var_gmm.df_model,
                                                                       df_resid=var_gmm.df_resid)

        n_d_var = eta_d.T.var()
        n_h_var = eta_h.T.var()
        tot_var = var_gmm.resid['h'].values.var()

        v_d = np.round(n_d_var/tot_var,4)
        v_h = np.round(n_h_var/tot_var,4)
        v_cov = 1-v_d-v_h
        corr = np.round(np.corrcoef(eta_d,eta_h)[0,1],4)

        result_df.loc[title2] = [str(np.round(ols1.rsquared,4))+'('+str(np.round(ols1.f_pvalue,4))+')',v_d,v_h,v_cov,corr,p_h]

        return result_df

    def make(self):
        data_m = self.data.loc['1962':]
        data_q = pd.concat([self.data['h'].resample('Q').sum(),self.data[['d/p','rrel']].resample('Q').apply(lambda x: x[-1])],axis=1).loc['1962':]

        self.tot_table1 = self.make_table1(data_m)
        self.tot_table2 = self.make_table2(data_m,lags=1)
        self.tot_table2 = pd.concat([self.tot_table2, self.make_table2(data_m, lags=6)],axis=0)
        self.tot_table2 = pd.concat([self.tot_table2, self.make_table2(data_q, lags=4, freq='quarterly')], axis=0)




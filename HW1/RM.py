from statsmodels.compat.pandas import Appender

import numpy as np
from scipy import stats
from scipy import optimize

from statsmodels.tools.tools import pinv_extended
import statsmodels.base.model as base
from statsmodels.emplike.elregress import _ELRegOpts

from statsmodels.regression.linear_model import WLS, RegressionResultsWrapper, RegressionResults


_fit_regularized_doc =\
        r"""
        Return a regularized fit to a linear regression model.
        Parameters
        ----------
        method : str
            Either 'elastic_net' or 'sqrt_lasso'.
        alpha : scalar or array_like
            The penalty weight.  If a scalar, the same penalty weight
            applies to all variables in the model.  If a vector, it
            must have the same length as `params`, and contains a
            penalty weight for each coefficient.
        L1_wt : scalar
            The fraction of the penalty given to the L1 penalty term.
            Must be between 0 and 1 (inclusive).  If 0, the fit is a
            ridge fit, if 1 it is a lasso fit.
        start_params : array_like
            Starting values for ``params``.
        profile_scale : bool
            If True the penalized fit is computed using the profile
            (concentrated) log-likelihood for the Gaussian model.
            Otherwise the fit uses the residual sum of squares.
        refit : bool
            If True, the model is refit using only the variables that
            have non-zero coefficients in the regularized fit.  The
            refitted model is not regularized.
        **kwargs
            Additional keyword arguments that contain information used when
            constructing a model using the formula interface.
        Returns
        -------
        statsmodels.base.elastic_net.RegularizedResults
            The regularized results.
        Notes
        -----
        The elastic net uses a combination of L1 and L2 penalties.
        The implementation closely follows the glmnet package in R.
        The function that is minimized is:
        .. math::
            0.5*RSS/n + alpha*((1-L1\_wt)*|params|_2^2/2 + L1\_wt*|params|_1)
        where RSS is the usual regression sum of squares, n is the
        sample size, and :math:`|*|_1` and :math:`|*|_2` are the L1 and L2
        norms.
        For WLS and GLS, the RSS is calculated using the whitened endog and
        exog data.
        Post-estimation results are based on the same data used to
        select variables, hence may be subject to overfitting biases.
        The elastic_net method uses the following keyword arguments:
        maxiter : int
            Maximum number of iterations
        cnvrg_tol : float
            Convergence threshold for line searches
        zero_tol : float
            Coefficients below this threshold are treated as zero.
        The square root lasso approach is a variation of the Lasso
        that is largely self-tuning (the optimal tuning parameter
        does not depend on the standard deviation of the regression
        errors).  If the errors are Gaussian, the tuning parameter
        can be taken to be
        alpha = 1.1 * np.sqrt(n) * norm.ppf(1 - 0.05 / (2 * p))
        where n is the sample size and p is the number of predictors.
        The square root lasso uses the following keyword arguments:
        zero_tol : float
            Coefficients below this threshold are treated as zero.
        The cvxopt module is required to estimate model using the square root
        lasso.
        References
        ----------
        .. [*] Friedman, Hastie, Tibshirani (2008).  Regularization paths for
           generalized linear models via coordinate descent.  Journal of
           Statistical Software 33(1), 1-22 Feb 2010.
        .. [*] A Belloni, V Chernozhukov, L Wang (2011).  Square-root Lasso:
           pivotal recovery of sparse signals via conic programming.
           Biometrika 98(4), 791-806. https://arxiv.org/pdf/1009.5689.pdf
        """


class RegressionModel(base.LikelihoodModel):
    """
    Base class for linear regression models. Should not be directly called.
    Intended for subclassing.
    """
    def __init__(self, endog, exog, **kwargs):
        super(RegressionModel, self).__init__(endog, exog, **kwargs)
        self._data_attr.extend(['pinv_wexog', 'wendog', 'wexog', 'weights'])

    def initialize(self):
        """Initialize model components."""
        self.wexog = self.whiten(self.exog)
        self.wendog = self.whiten(self.endog)
        # overwrite nobs from class Model:
        self.nobs = float(self.wexog.shape[0])

        self._df_model = None
        self._df_resid = None
        self.rank = None

    @property
    def df_model(self):
        """
        The model degree of freedom.
        The dof is defined as the rank of the regressor matrix minus 1 if a
        constant is included.
        """
        if self._df_model is None:
            if self.rank is None:
                self.rank = np.linalg.matrix_rank(self.exog)
            self._df_model = float(self.rank - self.k_constant)
        return self._df_model

    @df_model.setter
    def df_model(self, value):
        self._df_model = value

    @property
    def df_resid(self):
        """
        The residual degree of freedom.
        The dof is defined as the number of observations minus the rank of
        the regressor matrix.
        """

        if self._df_resid is None:
            if self.rank is None:
                self.rank = np.linalg.matrix_rank(self.exog)
            self._df_resid = self.nobs - self.rank
        return self._df_resid

    @df_resid.setter
    def df_resid(self, value):
        self._df_resid = value

    def whiten(self, x):
        """
        Whiten method that must be overwritten by individual models.
        Parameters
        ----------
        x : array_like
            Data to be whitened.
        """
        return x

    def fit(self, beta, df_model, df_resid, method="pinv", cov_type='nonrobust', cov_kwds=None,
            use_t=True, **kwargs):

        if method == "pinv":
            if not (hasattr(self, 'pinv_wexog') and
                    hasattr(self, 'normalized_cov_params') and
                    hasattr(self, 'rank')):

                self.pinv_wexog, singular_values = pinv_extended(self.exog)
                self.normalized_cov_params = np.dot(
                    self.pinv_wexog, np.transpose(self.pinv_wexog))

                # Cache these singular values for use later.
                self.wexog_singular_values = singular_values
                self.rank = np.linalg.matrix_rank(np.diag(singular_values))

        self._df_model = df_model
        self._df_resid = df_resid

        if isinstance(self, OLS):
            lfit = OLSResults(
                self, params=beta,
                normalized_cov_params=self.normalized_cov_params,
                cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t)
        return RegressionResultsWrapper(lfit)

    def predict(self, params, exog=None):
        """
        Return linear predicted values from a design matrix.
        Parameters
        ----------
        params : array_like
            Parameters of a linear model.
        exog : array_like, optional
            Design / exogenous data. Model exog is used if None.
        Returns
        -------
        array_like
            An array of fitted values.
        Notes
        -----
        If the model has not yet been fit, params is not optional.
        """
        # JP: this does not look correct for GLMAR
        # SS: it needs its own predict method

        if exog is None:
            exog = self.exog

        return np.dot(exog, params)

    def get_distribution(self, params, scale, exog=None, dist_class=None):
        """
        Construct a random number generator for the predictive distribution.
        Parameters
        ----------
        params : array_like
            The model parameters (regression coefficients).
        scale : scalar
            The variance parameter.
        exog : array_like
            The predictor variable matrix.
        dist_class : class
            A random number generator class.  Must take 'loc' and 'scale'
            as arguments and return a random number generator implementing
            an ``rvs`` method for simulating random values. Defaults to normal.
        Returns
        -------
        gen
            Frozen random number generator object with mean and variance
            determined by the fitted linear model.  Use the ``rvs`` method
            to generate random values.
        Notes
        -----
        Due to the behavior of ``scipy.stats.distributions objects``,
        the returned random number generator must be called with
        ``gen.rvs(n)`` where ``n`` is the number of observations in
        the data set used to fit the model.  If any other value is
        used for ``n``, misleading results will be produced.
        """
        fit = self.predict(params, exog)
        if dist_class is None:
            from scipy.stats.distributions import norm
            dist_class = norm
        gen = dist_class(loc=fit, scale=np.sqrt(scale))
        return gen


class OLS(WLS):
    __doc__ = """
    Ordinary Least Squares
    %(params)s
    %(extra_params)s
    Attributes
    ----------
    weights : scalar
        Has an attribute weights = array(1.0) due to inheritance from WLS.
    See Also
    --------
    WLS : Fit a linear model using Weighted Least Squares.
    GLS : Fit a linear model using Generalized Least Squares.
    Notes
    -----
    No constant is added by the model unless you are using formulas.
    Examples
    --------
    >>> import statsmodels.api as sm
    >>> import numpy as np
    >>> duncan_prestige = sm.datasets.get_rdataset("Duncan", "carData")
    >>> Y = duncan_prestige.data['income']
    >>> X = duncan_prestige.data['education']
    >>> X = sm.add_constant(X)
    >>> model = sm.OLS(Y,X)
    >>> results = model.fit()
    >>> results.params
    const        10.603498
    education     0.594859
    dtype: float64
    >>> results.tvalues
    const        2.039813
    education    6.892802
    dtype: float64
    >>> print(results.t_test([1, 0]))
                                 Test for Constraints
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    c0            10.6035      5.198      2.040      0.048       0.120      21.087
    ==============================================================================
    >>> print(results.f_test(np.identity(2)))
    <F test: F=array([[159.63031026]]), p=1.2607168903696672e-20, df_denom=43, df_num=2>
    """ % {'params': base._model_params_doc,
           'extra_params': base._missing_param_doc + base._extra_param_doc}

    def __init__(self, endog, exog=None, missing='none', hasconst=None,
                 **kwargs):
        super(OLS, self).__init__(endog, exog, missing=missing,
                                  hasconst=hasconst, **kwargs)
        if "weights" in self._init_keys:
            self._init_keys.remove("weights")

    def loglike(self, params, scale=None):
        """
        The likelihood function for the OLS model.
        Parameters
        ----------
        params : array_like
            The coefficients with which to estimate the log-likelihood.
        scale : float or None
            If None, return the profile (concentrated) log likelihood
            (profiled over the scale parameter), else return the
            log-likelihood using the given scale value.
        Returns
        -------
        float
            The likelihood function evaluated at params.
        """
        nobs2 = self.nobs / 2.0
        nobs = float(self.nobs)
        resid = self.endog - np.dot(self.exog, params)
        if hasattr(self, 'offset'):
            resid -= self.offset
        ssr = np.sum(resid**2)
        if scale is None:
            # profile log likelihood
            llf = -nobs2*np.log(2*np.pi) - nobs2*np.log(ssr / nobs) - nobs2
        else:
            # log-likelihood
            llf = -nobs2 * np.log(2 * np.pi * scale) - ssr / (2*scale)
        return llf

    def whiten(self, x):
        """
        OLS model whitener does nothing.
        Parameters
        ----------
        x : array_like
            Data to be whitened.
        Returns
        -------
        array_like
            The input array unmodified.
        See Also
        --------
        OLS : Fit a linear model using Ordinary Least Squares.
        """
        return x

    def score(self, params, scale=None):
        """
        Evaluate the score function at a given point.
        The score corresponds to the profile (concentrated)
        log-likelihood in which the scale parameter has been profiled
        out.
        Parameters
        ----------
        params : array_like
            The parameter vector at which the score function is
            computed.
        scale : float or None
            If None, return the profile (concentrated) log likelihood
            (profiled over the scale parameter), else return the
            log-likelihood using the given scale value.
        Returns
        -------
        ndarray
            The score vector.
        """

        if not hasattr(self, "_wexog_xprod"):
            self._setup_score_hess()

        xtxb = np.dot(self._wexog_xprod, params)
        sdr = -self._wexog_x_wendog + xtxb

        if scale is None:
            ssr = self._wendog_xprod - 2 * np.dot(self._wexog_x_wendog.T,
                                                  params)
            ssr += np.dot(params, xtxb)
            return -self.nobs * sdr / ssr
        else:
            return -sdr / scale

    def _setup_score_hess(self):
        y = self.wendog
        if hasattr(self, 'offset'):
            y = y - self.offset
        self._wendog_xprod = np.sum(y * y)
        self._wexog_xprod = np.dot(self.wexog.T, self.wexog)
        self._wexog_x_wendog = np.dot(self.wexog.T, y)

    def hessian(self, params, scale=None):
        """
        Evaluate the Hessian function at a given point.
        Parameters
        ----------
        params : array_like
            The parameter vector at which the Hessian is computed.
        scale : float or None
            If None, return the profile (concentrated) log likelihood
            (profiled over the scale parameter), else return the
            log-likelihood using the given scale value.
        Returns
        -------
        ndarray
            The Hessian matrix.
        """

        if not hasattr(self, "_wexog_xprod"):
            self._setup_score_hess()

        xtxb = np.dot(self._wexog_xprod, params)

        if scale is None:
            ssr = self._wendog_xprod - 2 * np.dot(self._wexog_x_wendog.T,
                                                  params)
            ssr += np.dot(params, xtxb)
            ssrp = -2*self._wexog_x_wendog + 2*xtxb
            hm = self._wexog_xprod / ssr - np.outer(ssrp, ssrp) / ssr**2
            return -self.nobs * hm / 2
        else:
            return -self._wexog_xprod / scale

    def hessian_factor(self, params, scale=None, observed=True):
        """
        Calculate the weights for the Hessian.
        Parameters
        ----------
        params : ndarray
            The parameter at which Hessian is evaluated.
        scale : None or float
            If scale is None, then the default scale will be calculated.
            Default scale is defined by `self.scaletype` and set in fit.
            If scale is not None, then it is used as a fixed scale.
        observed : bool
            If True, then the observed Hessian is returned. If false then the
            expected information matrix is returned.
        Returns
        -------
        ndarray
            A 1d weight vector used in the calculation of the Hessian.
            The hessian is obtained by `(exog.T * hessian_factor).dot(exog)`.
        """

        return np.ones(self.exog.shape[0])

    @Appender(_fit_regularized_doc)
    def fit_regularized(self, method="elastic_net", alpha=0.,
                        L1_wt=1., start_params=None, profile_scale=False,
                        refit=False, **kwargs):

        # In the future we could add support for other penalties, e.g. SCAD.
        if method not in ("elastic_net", "sqrt_lasso"):
            msg = "Unknown method '%s' for fit_regularized" % method
            raise ValueError(msg)

        # Set default parameters.
        defaults = {"maxiter":  50, "cnvrg_tol": 1e-10,
                    "zero_tol": 1e-8}
        defaults.update(kwargs)

        if method == "sqrt_lasso":
            from statsmodels.base.elastic_net import (
                RegularizedResults, RegularizedResultsWrapper
            )
            params = self._sqrt_lasso(alpha, refit, defaults["zero_tol"])
            results = RegularizedResults(self, params)
            return RegularizedResultsWrapper(results)

        from statsmodels.base.elastic_net import fit_elasticnet

        if L1_wt == 0:
            return self._fit_ridge(alpha)

        # If a scale parameter is passed in, the non-profile
        # likelihood (residual sum of squares divided by -2) is used,
        # otherwise the profile likelihood is used.
        if profile_scale:
            loglike_kwds = {}
            score_kwds = {}
            hess_kwds = {}
        else:
            loglike_kwds = {"scale": 1}
            score_kwds = {"scale": 1}
            hess_kwds = {"scale": 1}

        return fit_elasticnet(self, method=method,
                              alpha=alpha,
                              L1_wt=L1_wt,
                              start_params=start_params,
                              loglike_kwds=loglike_kwds,
                              score_kwds=score_kwds,
                              hess_kwds=hess_kwds,
                              refit=refit,
                              check_step=False,
                              **defaults)

    def _sqrt_lasso(self, alpha, refit, zero_tol):

        try:
            import cvxopt
        except ImportError:
            msg = 'sqrt_lasso fitting requires the cvxopt module'
            raise ValueError(msg)

        n = len(self.endog)
        p = self.exog.shape[1]

        h0 = cvxopt.matrix(0., (2*p+1, 1))
        h1 = cvxopt.matrix(0., (n+1, 1))
        h1[1:, 0] = cvxopt.matrix(self.endog, (n, 1))

        G0 = cvxopt.spmatrix([], [], [], (2*p+1, 2*p+1))
        for i in range(1, 2*p+1):
            G0[i, i] = -1
        G1 = cvxopt.matrix(0., (n+1, 2*p+1))
        G1[0, 0] = -1
        G1[1:, 1:p+1] = self.exog
        G1[1:, p+1:] = -self.exog

        c = cvxopt.matrix(alpha / n, (2*p + 1, 1))
        c[0] = 1 / np.sqrt(n)

        from cvxopt import solvers
        solvers.options["show_progress"] = False

        rslt = solvers.socp(c, Gl=G0, hl=h0, Gq=[G1], hq=[h1])
        x = np.asarray(rslt['x']).flat
        bp = x[1:p+1]
        bn = x[p+1:]
        params = bp - bn

        if not refit:
            return params

        ii = np.flatnonzero(np.abs(params) > zero_tol)
        rfr = OLS(self.endog, self.exog[:, ii]).fit()
        params *= 0
        params[ii] = rfr.params

        return params

    def _fit_ridge(self, alpha):
        """
        Fit a linear model using ridge regression.
        Parameters
        ----------
        alpha : scalar or array_like
            The penalty weight.  If a scalar, the same penalty weight
            applies to all variables in the model.  If a vector, it
            must have the same length as `params`, and contains a
            penalty weight for each coefficient.
        Notes
        -----
        Equivalent to fit_regularized with L1_wt = 0 (but implemented
        more efficiently).
        """

        u, s, vt = np.linalg.svd(self.exog, 0)
        v = vt.T
        q = np.dot(u.T, self.endog) * s
        s2 = s * s
        if np.isscalar(alpha):
            sd = s2 + alpha * self.nobs
            params = q / sd
            params = np.dot(v, params)
        else:
            alpha = np.asarray(alpha)
            vtav = self.nobs * np.dot(vt, alpha[:, None] * v)
            d = np.diag(vtav) + s2
            np.fill_diagonal(vtav, d)
            r = np.linalg.solve(vtav, q)
            params = np.dot(v, r)

        from statsmodels.base.elastic_net import RegularizedResults
        return RegularizedResults(self, params)

class OLSResults(RegressionResults):
    """
    Results class for for an OLS model.
    Parameters
    ----------
    model : RegressionModel
        The regression model instance.
    params : ndarray
        The estimated parameters.
    normalized_cov_params : ndarray
        The normalized covariance parameters.
    scale : float
        The estimated scale of the residuals.
    cov_type : str
        The covariance estimator used in the results.
    cov_kwds : dict
        Additional keywords used in the covariance specification.
    use_t : bool
        Flag indicating to use the Student's t in inference.
    **kwargs
        Additional keyword arguments used to initialize the results.
    See Also
    --------
    RegressionResults
        Results store for WLS and GLW models.
    Notes
    -----
    Most of the methods and attributes are inherited from RegressionResults.
    The special methods that are only available for OLS are:
    - get_influence
    - outlier_test
    - el_test
    - conf_int_el
    """

    def get_influence(self):
        """
        Calculate influence and outlier measures.
        Returns
        -------
        OLSInfluence
            The instance containing methods to calculate the main influence and
            outlier measures for the OLS regression.
        See Also
        --------
        statsmodels.stats.outliers_influence.OLSInfluence
            A class that exposes methods to examine observation influence.
        """
        from statsmodels.stats.outliers_influence import OLSInfluence
        return OLSInfluence(self)

    def outlier_test(self, method='bonf', alpha=.05, labels=None,
                     order=False, cutoff=None):
        """
        Test observations for outliers according to method.
        Parameters
        ----------
        method : str
            The method to use in the outlier test.  Must be one of:
            - `bonferroni` : one-step correction
            - `sidak` : one-step correction
            - `holm-sidak` :
            - `holm` :
            - `simes-hochberg` :
            - `hommel` :
            - `fdr_bh` : Benjamini/Hochberg
            - `fdr_by` : Benjamini/Yekutieli
            See `statsmodels.stats.multitest.multipletests` for details.
        alpha : float
            The familywise error rate (FWER).
        labels : None or array_like
            If `labels` is not None, then it will be used as index to the
            returned pandas DataFrame. See also Returns below.
        order : bool
            Whether or not to order the results by the absolute value of the
            studentized residuals. If labels are provided they will also be
            sorted.
        cutoff : None or float in [0, 1]
            If cutoff is not None, then the return only includes observations
            with multiple testing corrected p-values strictly below the cutoff.
            The returned array or dataframe can be empty if t.
        Returns
        -------
        array_like
            Returns either an ndarray or a DataFrame if labels is not None.
            Will attempt to get labels from model_results if available. The
            columns are the Studentized residuals, the unadjusted p-value,
            and the corrected p-value according to method.
        Notes
        -----
        The unadjusted p-value is stats.t.sf(abs(resid), df) where
        df = df_resid - 1.
        """
        from statsmodels.stats.outliers_influence import outlier_test
        return outlier_test(self, method, alpha, labels=labels,
                            order=order, cutoff=cutoff)

    def el_test(self, b0_vals, param_nums, return_weights=0, ret_params=0,
                method='nm', stochastic_exog=1):
        """
        Test single or joint hypotheses using Empirical Likelihood.
        Parameters
        ----------
        b0_vals : 1darray
            The hypothesized value of the parameter to be tested.
        param_nums : 1darray
            The parameter number to be tested.
        return_weights : bool
            If true, returns the weights that optimize the likelihood
            ratio at b0_vals. The default is False.
        ret_params : bool
            If true, returns the parameter vector that maximizes the likelihood
            ratio at b0_vals.  Also returns the weights.  The default is False.
        method : str
            Can either be 'nm' for Nelder-Mead or 'powell' for Powell.  The
            optimization method that optimizes over nuisance parameters.
            The default is 'nm'.
        stochastic_exog : bool
            When True, the exogenous variables are assumed to be stochastic.
            When the regressors are nonstochastic, moment conditions are
            placed on the exogenous variables.  Confidence intervals for
            stochastic regressors are at least as large as non-stochastic
            regressors. The default is True.
        Returns
        -------
        tuple
            The p-value and -2 times the log-likelihood ratio for the
            hypothesized values.
        Examples
        --------

        """
        params = np.copy(self.params)
        opt_fun_inst = _ELRegOpts()  # to store weights
        if len(param_nums) == len(params):
            llr = opt_fun_inst._opt_nuis_regress(
                [],
                param_nums=param_nums,
                endog=self.model.endog,
                exog=self.model.exog,
                nobs=self.model.nobs,
                nvar=self.model.exog.shape[1],
                params=params,
                b0_vals=b0_vals,
                stochastic_exog=stochastic_exog)
            pval = 1 - stats.chi2.cdf(llr, len(param_nums))
            if return_weights:
                return llr, pval, opt_fun_inst.new_weights
            else:
                return llr, pval
        x0 = np.delete(params, param_nums)
        args = (param_nums, self.model.endog, self.model.exog,
                self.model.nobs, self.model.exog.shape[1], params,
                b0_vals, stochastic_exog)
        if method == 'nm':
            llr = optimize.fmin(opt_fun_inst._opt_nuis_regress, x0,
                                maxfun=10000, maxiter=10000, full_output=1,
                                disp=0, args=args)[1]
        if method == 'powell':
            llr = optimize.fmin_powell(opt_fun_inst._opt_nuis_regress, x0,
                                       full_output=1, disp=0,
                                       args=args)[1]

        pval = 1 - stats.chi2.cdf(llr, len(param_nums))
        if ret_params:
            return llr, pval, opt_fun_inst.new_weights, opt_fun_inst.new_params
        elif return_weights:
            return llr, pval, opt_fun_inst.new_weights
        else:
            return llr, pval

    def conf_int_el(self, param_num, sig=.05, upper_bound=None,
                    lower_bound=None, method='nm', stochastic_exog=True):
        """
        Compute the confidence interval using Empirical Likelihood.
        Parameters
        ----------
        param_num : float
            The parameter for which the confidence interval is desired.
        sig : float
            The significance level.  Default is 0.05.
        upper_bound : float
            The maximum value the upper limit can be.  Default is the
            99.9% confidence value under OLS assumptions.
        lower_bound : float
            The minimum value the lower limit can be.  Default is the 99.9%
            confidence value under OLS assumptions.
        method : str
            Can either be 'nm' for Nelder-Mead or 'powell' for Powell.  The
            optimization method that optimizes over nuisance parameters.
            The default is 'nm'.
        stochastic_exog : bool
            When True, the exogenous variables are assumed to be stochastic.
            When the regressors are nonstochastic, moment conditions are
            placed on the exogenous variables.  Confidence intervals for
            stochastic regressors are at least as large as non-stochastic
            regressors.  The default is True.
        Returns
        -------
        lowerl : float
            The lower bound of the confidence interval.
        upperl : float
            The upper bound of the confidence interval.
        See Also
        --------
        el_test : Test parameters using Empirical Likelihood.
        Notes
        -----
        This function uses brentq to find the value of beta where
        test_beta([beta], param_num)[1] is equal to the critical value.
        The function returns the results of each iteration of brentq at each
        value of beta.
        The current function value of the last printed optimization should be
        the critical value at the desired significance level. For alpha=.05,
        the value is 3.841459.
        To ensure optimization terminated successfully, it is suggested to do
        el_test([lower_limit], [param_num]).
        If the optimization does not terminate successfully, consider switching
        optimization algorithms.
        If optimization is still not successful, try changing the values of
        start_int_params.  If the current function value repeatedly jumps
        from a number between 0 and the critical value and a very large number
        (>50), the starting parameters of the interior minimization need
        to be changed.
        """
        r0 = stats.chi2.ppf(1 - sig, 1)
        if upper_bound is None:
            upper_bound = self.conf_int(.01)[param_num][1]
        if lower_bound is None:
            lower_bound = self.conf_int(.01)[param_num][0]

        def f(b0):
            return self.el_test(np.array([b0]), np.array([param_num]),
                                method=method,
                                stochastic_exog=stochastic_exog)[0] - r0

        lowerl = optimize.brenth(f, lower_bound,
                                 self.params[param_num])
        upperl = optimize.brenth(f, self.params[param_num],
                                 upper_bound)
        #  ^ Seems to be faster than brentq in most cases
        return (lowerl, upperl)

    def wald_test(self, r_matrix, cov_p=None, scale=1.0, invcov=None,
                  use_f=None, df_constraints=None):
        """
        Compute a Wald-test for a joint linear hypothesis.
        Parameters
        ----------
        r_matrix : {array_like, str, tuple}
            One of:
            - array : An r x k array where r is the number of restrictions to
              test and k is the number of regressors. It is assumed that the
              linear combination is equal to zero.
            - str : The full hypotheses to test can be given as a string.
              See the examples.
            - tuple : A tuple of arrays in the form (R, q), ``q`` can be
              either a scalar or a length p row vector.
        cov_p : array_like, optional
            An alternative estimate for the parameter covariance matrix.
            If None is given, self.normalized_cov_params is used.
        scale : float, optional
            Default is 1.0 for no scaling.
            .. deprecated:: 0.10.0
        invcov : array_like, optional
            A q x q array to specify an inverse covariance matrix based on a
            restrictions matrix.
        use_f : bool
            If True, then the F-distribution is used. If False, then the
            asymptotic distribution, chisquare is used. If use_f is None, then
            the F distribution is used if the model specifies that use_t is True.
            The test statistic is proportionally adjusted for the distribution
            by the number of constraints in the hypothesis.
        df_constraints : int, optional
            The number of constraints. If not provided the number of
            constraints is determined from r_matrix.
        Returns
        -------
        ContrastResults
            The results for the test are attributes of this results instance.
        See Also
        --------
        f_test : Perform an F tests on model parameters.
        t_test : Perform a single hypothesis test.
        statsmodels.stats.contrast.ContrastResults : Test results.
        patsy.DesignInfo.linear_constraint : Specify a linear constraint.
        Notes
        -----
        The matrix `r_matrix` is assumed to be non-singular. More precisely,
        r_matrix (pX pX.T) r_matrix.T
        is assumed invertible. Here, pX is the generalized inverse of the
        design matrix of the model. There can be problems in non-OLS models
        where the rank of the covariance of the noise is not full.
        """
        if scale != 1.0:
            import warnings
            warnings.warn('scale is has no effect and is deprecated. It will'
                          'be removed in the next version.',
                          DeprecationWarning)

        if use_f is None:
            # switch to use_t false if undefined
            use_f = (hasattr(self, 'use_t') and self.use_t)

        from patsy import DesignInfo
        names = self.model.data.cov_names
        params = self.params.ravel()
        LC = DesignInfo(names).linear_constraint(r_matrix)
        r_matrix, q_matrix = LC.coefs, LC.constants

        if (self.normalized_cov_params is None and cov_p is None and
                invcov is None and not hasattr(self, 'cov_params_default')):
            raise ValueError('need covariance of parameters for computing '
                             'F statistics')

        cparams = np.dot(r_matrix, params[:, None])
        J = float(r_matrix.shape[0])  # number of restrictions

        if q_matrix is None:
            q_matrix = np.zeros(J)
        else:
            q_matrix = np.asarray(q_matrix)
        if q_matrix.ndim == 1:
            q_matrix = q_matrix[:, None]
            if q_matrix.shape[0] != J:
                raise ValueError("r_matrix and q_matrix must have the same "
                                 "number of rows")
        Rbq = cparams - q_matrix
        if invcov is None:
            cov_p = self.cov_params(r_matrix=r_matrix, cov_p=cov_p)
            if np.isnan(cov_p).max():
                raise ValueError("r_matrix performs f_test for using "
                                 "dimensions that are asymptotically "
                                 "non-normal")
            invcov = np.linalg.pinv(cov_p)
            J_ = np.linalg.matrix_rank(cov_p)
            if J_ < J:
                import warnings
                warnings.warn('covariance of constraints does not have full '
                              'rank. The number of constraints is %d, but '
                              'rank is %d' % (J, J_), ValueWarning)
                J = J_

        # TODO streamline computation, we do not need to compute J if given
        if df_constraints is not None:
            # let caller override J by df_constraint
            J = df_constraints

        if (hasattr(self, 'mle_settings') and
                self.mle_settings['optimizer'] in ['l1', 'l1_cvxopt_cp']):
            F = nan_dot(nan_dot(Rbq.T, invcov), Rbq)
        else:
            F = np.dot(np.dot(Rbq.T, invcov), Rbq)

        df_resid = getattr(self, 'df_resid_inference', self.df_resid)
        if use_f:
            F /= J
            return ContrastResults(F=F, df_denom=df_resid,
                                   df_num=J)  # invcov.shape[0])
        else:
            return ContrastResults(chi2=F, df_denom=J, statistic=F,
                                   distribution='chi2', distargs=(J,))
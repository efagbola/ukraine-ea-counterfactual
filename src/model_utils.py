import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR


def choose_lag(endog, exog=None, maxlags=12, ic="aic"):
    """Select lag order by information criterion."""
    m = VAR(endog=endog, exog=exog)
    sel = m.select_order(maxlags=maxlags)
    p = getattr(sel, ic)
    return int(p) if (p is not None and not np.isnan(p)) else 2


def long_run_multiplier(A_mats):
    """C(1) = (I - A1 - ... - Ap)^{-1}"""
    k = A_mats.shape[1]
    return np.linalg.inv(np.eye(k) - A_mats.sum(axis=0))


def bq_impact_matrix(A_mats, sigma_u):
    """
    Blanchard-Quah identification for a 2-variable VAR.
    Returns B such that u_t = B e_t, Var(e_t)=I,
    with the lower-triangular long-run restriction enforcing that
    shock 2 (demand) has zero long-run effect on variable 1 (output).
    """
    C1       = long_run_multiplier(A_mats)
    omega    = C1 @ sigma_u @ C1.T
    S        = np.linalg.cholesky(omega)  
    B        = np.linalg.inv(C1) @ S
    return B


def get_structural_shocks(resid, B):
    """e_t = B^{-1} u_t"""
    Binv   = np.linalg.inv(B)
    shocks = (Binv @ resid.values.T).T
    return pd.DataFrame(shocks,
                        index=pd.to_datetime(resid.index),
                        columns=["shock_supply", "shock_demand"])


def monetary_autonomy_weight(dates):
    """
    Time-varying weight in [0,1] reflecting Ukraine's genuine monetary sovereignty,
    calibrated exactly to the regime seen in Part A.

    0.0 = full peg with no independent policy so Euro Area membership would have almost no additional change
    1.0 = genuine float with full inflation targeting so Euro Area membership would have the full counterfactual treatment
    """
    w = pd.Series(1.0, index=range(len(dates)))

    d = pd.DatetimeIndex(dates)

    # Hard peg periods from part A (treatment of EA membership almost zero)
    w[(d < "2005-03-01")] = 0.05                                          # conventional peg 5.33
    w[(d >= "2005-03-01") & (d < "2008-04-01")] = 0.05                   # hard band 5.00–5.06
    w[(d >= "2009-02-01") & (d < "2014-02-01")] = 0.05                   # re-peg 7.7–8.0

    # GFC transition so when the peg collapsed but NBU fixed rate Dec 2008. Period of partial sovereignty
    w[(d >= "2008-04-01") & (d < "2009-02-01")] = 0.50

    # 2014–15 crisis: formally floating but constrained
    w[(d >= "2014-02-01") & (d < "2014-09-01")] = 0.60  # forced float, Crimea shock
    w[(d >= "2014-09-01") & (d < "2015-03-01")] = 0.50  # second part, heavy controls

    # Inflation targeting transition. Period of floating ER with crisis restrictions gradually unwinding
    w[(d >= "2015-03-01") & (d < "2016-09-01")] = 0.70

    # Genuine monetary sovereignty periods, so can feel the full counterfactual treatment weight
    w[(d >= "2016-09-01") & (d < "2019-02-07")] = 1.00  # full-fledged IT
    w[(d >= "2019-02-07") & (d < "2022-02-24")] = 1.00  # mature prewar IT

    # Wartime fixed rate so time of minimal sovereignty
    w[(d >= "2022-02-24") & (d < "2023-10-03")] = 0.10

    # Managed flexibility post Oct 2023. Return of partial sovereignty but still constrained by the war, so not back to full 1.0 weight yet.
    w[d >= "2023-10-03"] = 0.35

    return w.values


def simulate_counterfactual(ukr_res, B_ukr, ea_shocks, ukr_endog):
    """
    Re-simulate Ukraine's VAR from t=0 replacing demand shocks with Euro Area demand.
    This is the full re-simulation approach, not just a residual swap on fitted values, 
    so the shock substitution propagates correctly through the VAR lag dynamics.
    
    Steps:
      1. Extract Ukraine structural shocks.
      2. Replace demand column with Euro Area demand shocks on the common date range.
      3. Convert counterfactual structural shocks back to reduced-form residuals.
      4. Re-run the VAR recursion period by period using those residuals.
    """
    p       = ukr_res.k_ar
    k       = ukr_res.neqs
    A_mats  = ukr_res.coefs                          # (p, k, k)
    const   = ukr_res.params.loc["const"].values     # (k,)

    # Ukraine structural shocks
    ukr_struct  = get_structural_shocks(ukr_res.resid, B_ukr)

    # Align on dates present in both
    common_idx  = ukr_struct.index.intersection(ea_shocks.index)
    ukr_struct  = ukr_struct.loc[common_idx].copy()
    ea_dem      = ea_shocks.loc[common_idx, "shock_demand"].values

    # Build counterfactual structural shock matrix so keep supply and change demand and weight by monetary sovereignty (autonomy)
    autonomy_w  = monetary_autonomy_weight(common_idx)
    e_cf        = ukr_struct.values.copy()
    # Weighted blend: w=1 means full EA demand shock; w=0 means keep Ukraine's own demand shock
    e_cf[:, 1]  = autonomy_w * ea_dem + (1.0 - autonomy_w) * ukr_struct["shock_demand"].values

    # Back to reduced-form residuals
    u_cf        = (B_ukr @ e_cf.T).T   # (T, k)

    # Use fittedvalues index to map back to absolute positions
    fitted_idx  = pd.to_datetime(ukr_res.fittedvalues.index)
    fitted_idx  = pd.DatetimeIndex(fitted_idx).drop_duplicates()

    # Position of common_idx[0] in fitted_idx
    # For t < p we fall back to actual data from ukr_res.endog
    full_endog  = ukr_res.endog          
    T           = len(common_idx)

    y_cf        = np.zeros((T, k))

    # Find how many lags precede the first residual date
    # ukr_res.resid has length T_fit so full_endog has T_fit + p rows
    offset      = full_endog.shape[0] - len(ukr_res.resid)  # = p

    for t in range(T):
        y_cf[t] = const.copy()
        for lag in range(1, p + 1):
            if t - lag >= 0:
                prev = y_cf[t - lag]
            else:
                # Use actual data (pre-simulation initial conditions)
                idx_in_full = offset + t - lag   # always >= 0 since lag <= p and t >= 0
                idx_in_full = max(idx_in_full, 0)
                prev = full_endog[idx_in_full]
            y_cf[t] += A_mats[lag - 1] @ prev
        y_cf[t] += u_cf[t]

    return pd.DataFrame(y_cf, index=common_idx, columns=ukr_endog)

def crisis_peak(df, start, end, col):
    """Peak value of a series over a specified window."""
    sub = df[(df["date"] >= start) & (df["date"] <= end)]
    return sub[col].max() if (len(sub) and sub[col].notna().any()) else np.nan
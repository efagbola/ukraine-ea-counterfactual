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

    Returns B such that:
        u_t = B e_t
        Var(e_t) = I

    with the long-run restriction that shock 2 (demand)
    has zero long-run effect on variable 1 (output).
    """
    C1 = long_run_multiplier(A_mats)
    omega = C1 @ sigma_u @ C1.T
    S = np.linalg.cholesky(omega)
    B = np.linalg.inv(C1) @ S
    return B


def get_structural_shocks(resid, B):
    """Recover structural shocks e_t = B^{-1} u_t."""
    Binv = np.linalg.inv(B)
    shocks = (Binv @ resid.values.T).T
    return pd.DataFrame(
        shocks,
        index=pd.to_datetime(resid.index),
        columns=["shock_supply", "shock_demand"],
    )


def euro_benefit_weight(dates):
    """
    Regime-dependent treatment intensity.

    Economic logic:
    - Peg periods: very small treatment effect because Ukraine already had little
      effective monetary autonomy.
    - 2014-2016 crisis/transition: largest treatment effect because euro membership
      would have most strongly reduced devaluation-driven inflation and credibility loss.
    - Post-2016 inflation targeting: still some effect, but smaller than 2014-2016.
    - Wartime fixed-rate regime: very small normal treatment effect again.
    """
    d = pd.DatetimeIndex(dates)
    w = pd.Series(index=d, dtype=float)

    # Peg / near-peg periods -> very small effect
    w[d < "2008-04-01"] = 0.05
    w[(d >= "2008-04-01") & (d < "2014-02-01")] = 0.05

    # Crisis / transition -> largest effect
    w[(d >= "2014-02-01") & (d < "2015-03-01")] = 0.70
    w[(d >= "2015-03-01") & (d < "2016-09-01")] = 0.50

    # Post-2016 inflation targeting -> smaller effect
    w[(d >= "2016-09-01") & (d < "2022-02-24")] = 0.20

    # Wartime fixed-rate and after -> small effect again
    w[d >= "2022-02-24"] = 0.10

    return w.values


def simulate_counterfactual(ukr_res, B_ukr, ea_shocks, ukr_endog):
    """
    Full recursive counterfactual simulation for Ukraine under Euro Area membership.

    Strategy:
    1. Recover Ukraine structural shocks.
    2. Keep Ukraine supply shocks unchanged.
    3. Replace Ukraine demand shocks with a regime-dependent weighted blend of:
         - Euro Area demand shocks
         - Ukraine's own demand shocks
    4. Convert counterfactual structural shocks back to reduced-form residuals.
    5. Re-simulate Ukraine's VAR recursively so the shock substitution propagates
       through the lag structure.
    """
    p = ukr_res.k_ar
    k = ukr_res.neqs
    A_mats = ukr_res.coefs
    const = ukr_res.params.loc["const"].values

    # Ukraine structural shocks
    ukr_struct = get_structural_shocks(ukr_res.resid, B_ukr)

    # Align on common sample
    common_idx = ukr_struct.index.intersection(pd.to_datetime(ea_shocks.index))
    ukr_struct = ukr_struct.loc[common_idx].copy()
    ea_shocks = ea_shocks.loc[common_idx].copy()

    ukr_dem = ukr_struct["shock_demand"].values
    ea_dem = ea_shocks["shock_demand"].values

    # Regime-dependent treatment weight
    w = euro_benefit_weight(common_idx)

    # Structural counterfactual:
    # keep supply shocks, partially replace demand shocks with EA demand shocks
    e_cf = ukr_struct.values.copy()
    e_cf[:, 1] = w * ea_dem + (1.0 - w) * ukr_dem

    # Convert back to reduced-form residuals
    u_cf = (B_ukr @ e_cf.T).T

    # Full recursive simulation
    full_endog = ukr_res.endog
    T = len(common_idx)
    y_cf = np.zeros((T, k))

    # ukr_res.resid starts after p lags, so full_endog has p extra rows
    offset = full_endog.shape[0] - len(ukr_res.resid)

    for t in range(T):
        y_cf[t] = const.copy()

        for lag in range(1, p + 1):
            if t - lag >= 0:
                prev = y_cf[t - lag]
            else:
                idx_in_full = max(offset + t - lag, 0)
                prev = full_endog[idx_in_full]

            y_cf[t] += A_mats[lag - 1] @ prev

        y_cf[t] += u_cf[t]

    return pd.DataFrame(y_cf, index=common_idx, columns=ukr_endog)


def crisis_peak(df, start, end, col):
    """Peak value of a series over a specified window."""
    sub = df[(df["date"] >= start) & (df["date"] <= end)]
    return sub[col].max() if (len(sub) and sub[col].notna().any()) else np.nan
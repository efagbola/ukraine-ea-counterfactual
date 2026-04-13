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
        u_t = B e_t,  Var(e_t) = I

    with the long-run restriction that shock 2 (demand)
    has zero long-run effect on variable 1 (output).
    """
    C1    = long_run_multiplier(A_mats)
    omega = C1 @ sigma_u @ C1.T
    S     = np.linalg.cholesky(omega)
    B     = np.linalg.inv(C1) @ S
    return B


def get_structural_shocks(resid, B):
    """Recover structural shocks: e_t = B^{-1} u_t."""
    Binv   = np.linalg.inv(B)
    shocks = (Binv @ resid.values.T).T
    return pd.DataFrame(
        shocks,
        index=pd.to_datetime(resid.index),
        columns=["shock_supply", "shock_demand"],
    )

def euro_benefit_weight(dates):
    """
    Regime-dependent treatment intensity for the Euro Area counterfactual.

    The weight should reflect BOTH:
    1. constrained sovereignty / fear of floating:
       when Ukraine is effectively pegged, Euro membership changes less;
    2. credibility import:
       before 2016, EA membership should generally matter more than after
       inflation targeting is adopted.

    So the desired pattern is:
    - small during hard-peg periods,
    - large during 2014-15 collapse,
    - still meaningfully positive pre-2016 because credibility was weak,
    - smaller after 2016 once NBU credibility improves.
    """
    d = pd.DatetimeIndex(dates)
    w = pd.Series(index=d, dtype=float)

    # 2002-2004: de facto peg, but very weak credibility.
    # Keep treatment effect present but not huge.
    w[d < "2005-01-01"] = 0.30

    # 2005-2008: still pre-credibility-reform era.
    # Slightly larger benefit from imported credibility.
    w[(d >= "2005-01-01") & (d < "2008-09-01")] = 0.40

    # GFC + re-peg period: autonomy again very limited.
    # Treatment should fall, not rise.
    w[(d >= "2008-09-01") & (d < "2014-02-01")] = 0.12

    # 2014 devaluation crisis: maximum euro benefit.
    w[(d >= "2014-02-01") & (d < "2015-03-01")] = 0.65

    # 2015-2016 transition to inflation targeting:
    # still high, but lower than the peak crisis phase.
    w[(d >= "2015-03-01") & (d < "2016-09-01")] = 0.40

    # Post-2016 inflation targeting:
    # counterfactual should be closer to actual than in pre-2016 period.
    w[(d >= "2016-09-01") & (d < "2022-02-24")] = 0.15

    # Wartime fixed-rate regime:
    # euro treatment effect small; inflation mostly war/supply/admin driven.
    w[d >= "2022-02-24"] = 0.08

    return w.values


def simulate_counterfactual(ukr_res, B_ukr, ea_shocks, ukr_endog):
    """
    Full recursive counterfactual simulation for Ukraine under Euro Area membership.
    Follows the Bayoumi and Eichengreen (1993) shock-substitution approach.

    Steps:
      1. Recover Ukraine structural shocks via BQ decomposition.
      2. Keep Ukraine supply shocks unchanged throughout. These reflect real factors
         such as energy prices and geopolitical shocks that are structural to Ukraine's
         economy and would persist under any currency regime.
      3. Replace Ukraine demand shocks with a regime-dependent weighted blend of
         EA demand shocks and Ukraine's own demand shocks, using euro_benefit_weight().
      4. Convert counterfactual structural shocks back to reduced-form residuals.
      5. Re-simulate Ukraine's VAR recursively so the shock substitution propagates
         correctly through the full lag structure.

    Unlike a simple cross-sectional mean of Euro Area inflation, this approach feeds
    EA structural demand shocks through Ukraine's own estimated VAR dynamics,
    preserving Ukraine-specific transmission mechanisms while only replacing the
    source of demand impulses.
    """
    p      = ukr_res.k_ar
    k      = ukr_res.neqs
    A_mats = ukr_res.coefs
    const  = ukr_res.params.loc["const"].values

    ukr_struct = get_structural_shocks(ukr_res.resid, B_ukr)

    common_idx = ukr_struct.index.intersection(pd.to_datetime(ea_shocks.index))
    ukr_struct = ukr_struct.loc[common_idx].copy()
    ea_shocks  = ea_shocks.loc[common_idx].copy()

    ukr_dem = ukr_struct["shock_demand"].values
    ea_dem  = ea_shocks["shock_demand"].values

    w = euro_benefit_weight(common_idx)

    # Keep supply shocks, partially replace demand shocks with EA demand shocks
    e_cf       = ukr_struct.values.copy()
    e_cf[:, 1] = w * ea_dem + (1.0 - w) * ukr_dem

    u_cf = (B_ukr @ e_cf.T).T  # (T, k)

    full_endog = ukr_res.endog
    T          = len(common_idx)
    y_cf       = np.zeros((T, k))
    offset     = full_endog.shape[0] - len(ukr_res.resid)  # = p lags

    for t in range(T):
        y_cf[t] = const.copy()
        for lag in range(1, p + 1):
            if t - lag >= 0:
                prev = y_cf[t - lag]
            else:
                prev = full_endog[max(offset + t - lag, 0)]
            y_cf[t] += A_mats[lag - 1] @ prev
        y_cf[t] += u_cf[t]

    return pd.DataFrame(y_cf, index=common_idx, columns=ukr_endog)


def crisis_peak(df, start, end, col):
    """Peak value of a series over a specified window."""
    sub = df[(df["date"] >= start) & (df["date"] <= end)]
    return sub[col].max() if (len(sub) and sub[col].notna().any()) else np.nan
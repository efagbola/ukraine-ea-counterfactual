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
    Regime-dependent treatment intensity for Euro Area membership counterfactual.

    This weight controls how much of Ukraine's structural demand shock is replaced
    by the Euro Area demand shock each period. It is calibrated to Ukraine's monetary
    regime chronology established in Part A.

    Economic logic by period:

    - Peg periods (pre-2008, 2009-2014): weight = 0.05
      Ukraine already had very limited monetary autonomy under the de facto dollar
      peg. The NBU subordinated interest rate policy to exchange-rate stability, so
      Euro Area membership would not have represented a large additional change in
      demand conditions. The shock-substitution treatment effect is therefore near zero.

    - 2014-2016 crisis and IT transition: weight = 0.70 / 0.50
      This is the period of largest treatment effect. The hryvnia collapsed by ~90%,
      generating acute devaluation-driven inflation that would have been impossible
      under the euro. EA membership would have eliminated this channel entirely and
      imported the ECB's nominal anchor during the weakest period of NBU credibility.
      The high weight reflects the maximum divergence between Ukrainian and EA demand
      conditions and the largest potential inflation-reduction benefit.

    - Post-2016 inflation targeting: weight = 0.20
      The NBU formally adopted inflation targeting in December 2016. Credibility
      improved and the exchange rate floated freely. EA membership would still have
      aligned Ukraine's demand conditions more closely with the ECB, but the marginal
      benefit is smaller than during 2014-2016 because the NBU had partially converged
      toward a modern monetary framework. The lower weight produces a smaller
      counterfactual gap post-2016, consistent with the Giavazzi-Pagano (1988) sanity
      check that credibility gains from EA membership diminish as domestic credibility
      improves.

    - Wartime and post-wartime (post Feb 2022): weight = 0.10
      The NBU reimposed a fixed exchange rate in February 2022 as an emergency
      stabilisation measure. Inflation dynamics were dominated by war conditions,
      supply disruptions, and administrative controls rather than normal monetary
      transmission. Substituting ECB demand shocks strongly in this period would
      misrepresent the counterfactual, so the weight is kept low.
    """
    d = pd.DatetimeIndex(dates)
    w = pd.Series(index=d, dtype=float)

    # Peg and near-peg periods: minimal treatment effect
    w[d < "2008-04-01"]                                          = 0.05
    w[(d >= "2008-04-01") & (d < "2014-02-01")]                  = 0.05

    # Crisis and IT transition: largest treatment effect
    w[(d >= "2014-02-01") & (d < "2015-03-01")]                  = 0.70
    w[(d >= "2015-03-01") & (d < "2016-09-01")]                  = 0.50

    # Post-2016 IT: smaller but positive treatment effect
    w[(d >= "2016-09-01") & (d < "2022-02-24")]                  = 0.20

    # Wartime and post-wartime: small treatment effect
    w[d >= "2022-02-24"]                                         = 0.10

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
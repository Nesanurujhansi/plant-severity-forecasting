"""Epidemiological Foliar-SEIR ODE State-Space Forecaster.

Models foliar pathogen expansion as a continuous-time, microclimate-forced
compartmental differential equation system (Susceptible-Latent-Infectious-Removed):
    dS/dt = - E(t) * S(t) * (I(t) + omega * L(t)) / K
    dL/dt =   E(t) * S(t) * (I(t) + omega * L(t)) / K - sigma(T(t)) * L(t)
    dI/dt =   sigma(T(t)) * L(t) + v_exp(T(t), VPD(t)) * I(t) * (1 - (I + R)/K) - gamma_sen * I(t)
    dR/dt =   gamma_sen * I(t)

Guarantees:
1. Mechanistic biological mass conservation: S(t) + L(t) + I(t) + R(t) = 1.0.
2. Bounded visible severity: S_visible(t) = I(t) + R(t) <= K <= 1.0.
3. Magarey microclimatic infection threshold driven by Leaf Wetness Duration (LWD) and rain-splash.
4. Incubation latency period modeled via accumulated thermal degree-days (GDD).
5. Continuous 4th-order Runge-Kutta (RK4) integration over arbitrary horizons (1-21 days).
6. Multi-strata canopy scaling relating flag leaf severity to whole-canopy vertical epidemics.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
from pydantic import BaseModel, Field

from ml.forecast.growth import (
    thermal_response_beta,
    humidity_response,
    compute_vpd,
    DEFAULT_GROWTH_PARAMS
)


class FoliarSEIRState(BaseModel):
    """Pydantic model representing state of foliar tissue at a single timepoint."""
    day: float = Field(..., description="Evaluation day.")
    susceptible_pct: float = Field(..., description="Uninfected healthy leaf tissue percentage.")
    latent_pct: float = Field(..., description="Asymptomatic incubating tissue percentage.")
    infectious_chlorotic_pct: float = Field(..., description="Active spreading chlorotic halo percentage.")
    removed_necrotic_pct: float = Field(..., description="Dead necrotic spot percentage.")
    visible_severity_pct: float = Field(..., description="Observed visual severity percentage (I + R).")
    canopy_severity_pct: float = Field(..., description="Estimated whole-canopy vertical severity percentage.")
    effective_reproduction_rate: float = Field(..., description="Instantaneous transmission rate R_eff.")


class FoliarSEIRTrajectoryResult(BaseModel):
    """Pydantic model for continuous Foliar-SEIR epidemiological forecast trajectories."""
    horizons_days: List[float] = Field(..., description="Evaluation time points in days.")
    visible_severity_pct_trajectory: List[float] = Field(..., description="Projected visible severity (I + R) * 100.")
    latent_pct_trajectory: List[float] = Field(..., description="Projected asymptomatic latent tissue L * 100.")
    chlorotic_pct_trajectory: List[float] = Field(..., description="Projected active chlorotic margin I * 100.")
    necrotic_pct_trajectory: List[float] = Field(..., description="Projected dead necrotic core R * 100.")
    canopy_severity_pct_trajectory: List[float] = Field(..., description="Estimated whole-canopy vertical severity trajectory.")
    daily_reproduction_numbers: List[float] = Field(..., description="Daily transmission multiplier R_eff.")
    daily_magarey_risks: List[float] = Field(..., description="Daily environmental Magarey infection risk [0, 1].")
    cumulative_degree_days: List[float] = Field(..., description="Accumulated growing degree-days for latency incubation.")
    carrying_capacity_pct: float = Field(..., description="Theoretical carrying capacity limit K * 100.")
    trajectory_summary: str = Field(..., description="Plain-language epidemiological summary of the forecast.")


class ContinuousTrajectoryResult(BaseModel):
    """Legacy backward-compatible Pydantic model for continuous forecast trajectories."""
    horizons_days: List[float] = Field(..., description="Evaluation time points in days.")
    severity_pct_trajectory: List[float] = Field(..., description="Projected severity percentage trajectory S(t) * 100.")
    canopy_severity_pct_trajectory: List[float] = Field(..., description="Estimated whole-canopy vertical disease severity.")
    carrying_capacity_pct: float = Field(..., description="Theoretical carrying capacity limit K * 100.")
    cumulative_risk_score: float = Field(..., description="Integrated environmental infection risk over horizon.")


def compute_magarey_infection_risk(
    temp_c: float,
    rh_pct: float,
    precip_mm: float = 0.0,
    lwd_hours: float = 0.0,
    t_opt: float = 20.5,
    t_min: float = 4.0,
    t_max: float = 34.0,
    rh_threshold: float = 85.0
) -> float:
    """Compute Magarey microclimate surface wetness infection risk index in [0.0, 1.0].

    Implements the Magarey generic foliar infection model (Magarey et al. 2005):
    Free moisture (leaf wetness duration) and thermal favorability govern spore germination.
    Rain splash adds mechanical spore detachment and canopy dispersal.

    Args:
        temp_c: Ambient / canopy temperature in °C.
        rh_pct: Relative humidity in %.
        precip_mm: Precipitation in mm.
        lwd_hours: Estimated leaf wetness duration in hours.
        t_opt: Optimum infection temperature (°C).
        t_min: Minimum cardinal temperature (°C).
        t_max: Maximum cardinal temperature (°C).
        rh_threshold: Relative humidity threshold for non-precip wetting (default 85%).

    Returns:
        Risk multiplier between 0.0 (no infection possible) and 1.0 (extreme epidemic pressure).
    """
    # 1. Thermal efficiency
    g_t = float(thermal_response_beta(temp_c, t_opt=t_opt, t_min=t_min, t_max=t_max))
    if g_t <= 0.001:
        return 0.0

    # 2. Moisture / leaf wetness factor
    has_rain = precip_mm > 0.1
    has_lwd = lwd_hours >= 3.0
    has_high_rh = rh_pct >= rh_threshold

    if has_rain or has_lwd or has_high_rh:
        # Moisture present: base wetting efficiency scales with duration/humidity
        moisture_score = min(1.0, (rh_pct - 60.0) / 38.0) if rh_pct > 60.0 else 0.1
        if lwd_hours > 0:
            moisture_score = max(moisture_score, min(1.0, lwd_hours / 12.0))
        # Rain splash boosts dispersal kinetically
        splash_boost = 1.0 + 0.30 * np.log1p(max(0.0, precip_mm))
        risk = g_t * moisture_score * splash_boost
    else:
        # Dry foliage: infection is severely restricted by desiccation
        vpd = float(compute_vpd(temp_c, rh_pct))
        risk = g_t * max(0.02, 1.0 - (vpd / 1.8))

    return float(np.clip(risk, 0.0, 1.0))


def scale_flag_leaf_to_canopy(
    flag_severity: Union[float, np.ndarray],
    van_der_plank_gradient: float = 1.75
) -> Union[float, np.ndarray]:
    """Estimate whole-canopy vertical disease severity from flag leaf observation.

    In foliar cereal rust / Septoria epidemics, disease initiates on lower leaves
    and splashes vertically. Flag leaf (F) reflects late epidemic spread;
    F-1 and F-2 leaves consistently exhibit higher baseline severity:
    S_canopy = 0.40 * S_F + 0.35 * S_{F-1} + 0.25 * S_{F-2}
    where S_{F-1} = min(1.0, 1.75 * S_F) and S_{F-2} = min(1.0, 2.4 * S_F).
    """
    s_f = np.array(flag_severity, dtype=np.float64)
    s_f1 = np.clip(van_der_plank_gradient * s_f, 0.0, 1.0)
    s_f2 = np.clip((van_der_plank_gradient * 1.35) * s_f, 0.0, 1.0)

    s_canopy = 0.40 * s_f + 0.35 * s_f1 + 0.25 * s_f2
    return float(s_canopy) if np.isscalar(flag_severity) else s_canopy


def solve_foliar_seir_rk4(
    initial_necrotic_fraction: float,
    initial_chlorotic_fraction: float,
    daily_weather_series: List[Dict[str, float]],
    growth_params: Optional[Dict[str, float]] = None,
    frontier_perimeter_ratio: float = 1.0,
    carrying_capacity: float = 0.95,
    latent_incubation_gdd: float = 90.0,
    t_base: float = 4.0,
    dt: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the 4-compartment Foliar-SEIR differential equation system using 4th-order Runge-Kutta.

    dS/dt = - E(t) * S * (I + 0.15 * L) / K
    dL/dt =   E(t) * S * (I + 0.15 * L) / K - sigma(T) * L
    dI/dt =   sigma(T) * L + v_exp(T, VPD) * I * (1 - (I + R)/K) - gamma_sen * I
    dR/dt =   gamma_sen * I

    Args:
        initial_necrotic_fraction: Day 0 necrotic core fraction R_0 (0.0 to 1.0).
        initial_chlorotic_fraction: Day 0 chlorotic halo fraction I_0 (0.0 to 1.0).
        daily_weather_series: List of daily weather dicts with t_mean, rh_mean, precip, lwd_hours, vpd.
        growth_params: Biological parameters (r0, t_opt, t_min, t_max, k_h, rh_50, K).
        frontier_perimeter_ratio: Vision-derived lesion perimeter ratio (scales radial velocity).
        carrying_capacity: Maximum leaf necrotic carrying capacity K (default 0.95).
        latent_incubation_gdd: Degree-days required for latent incubation to erupt into chlorosis.
        t_base: Base temperature for degree-day accumulation (°C).
        dt: Integration time step in days (default 0.05 days = 1.2 hours).

    Returns:
        Tuple of (t_vals, S_vals, L_vals, I_vals, R_vals, R_eff_vals, Magarey_vals).
    """
    p = DEFAULT_GROWTH_PARAMS if growth_params is None else {**DEFAULT_GROWTH_PARAMS, **growth_params}
    k_cap = float(min(carrying_capacity, p.get("K", 0.95)))
    r0 = float(p.get("r0", 0.15))
    t_opt = float(p.get("t_opt", 20.5))
    t_min = float(p.get("t_min", 4.0))
    t_max = float(p.get("t_max", 34.0))

    n_days = max(1.0, float(len(daily_weather_series)))
    n_steps = int(np.ceil(n_days / dt))
    t_vals = np.linspace(0.0, n_days, n_steps + 1)

    # Initialize state arrays
    S_arr = np.zeros(n_steps + 1, dtype=np.float64)
    L_arr = np.zeros(n_steps + 1, dtype=np.float64)
    I_arr = np.zeros(n_steps + 1, dtype=np.float64)
    R_arr = np.zeros(n_steps + 1, dtype=np.float64)
    Reff_arr = np.zeros(n_steps + 1, dtype=np.float64)
    Risk_arr = np.zeros(n_steps + 1, dtype=np.float64)

    # Clean Day 0 initial state
    r_init = float(np.clip(initial_necrotic_fraction, 0.0, k_cap))
    i_init = float(np.clip(initial_chlorotic_fraction, 0.0, k_cap - r_init))

    # Initial asymptomatic latent reservoir: proportional to active chlorotic frontier
    # reflecting recent spore germination prior to scan
    l_init = float(np.clip(i_init * 0.40, 0.0, max(0.0, k_cap - r_init - i_init)))
    s_init = float(max(0.0, 1.0 - (r_init + i_init + l_init)))

    S_arr[0] = s_init
    L_arr[0] = l_init
    I_arr[0] = i_init
    R_arr[0] = r_init

    def get_weather_at_t(t_curr: float) -> Tuple[float, float, float, float, float]:
        day_idx = min(int(np.floor(t_curr)), len(daily_weather_series) - 1)
        w = daily_weather_series[day_idx]
        t_m = float(w.get("t_mean", 18.0))
        rh_m = float(w.get("rh_mean", 75.0))
        prec = float(w.get("precip", 0.0))
        lwd = float(w.get("lwd_hours", 3.0))
        vpd = float(w.get("vpd", compute_vpd(t_m, rh_m)))
        return t_m, rh_m, prec, lwd, vpd

    # State derivatives function
    def derivatives(
        t_c: float,
        S_c: float,
        L_c: float,
        I_c: float,
        R_c: float
    ) -> Tuple[float, float, float, float, float, float]:
        t_m, rh_m, prec, lwd, vpd = get_weather_at_t(t_c)

        # 1. Magarey microclimate infection risk
        m_risk = compute_magarey_infection_risk(
            temp_c=t_m,
            rh_pct=rh_m,
            precip_mm=prec,
            lwd_hours=lwd,
            t_opt=t_opt,
            t_min=t_min,
            t_max=t_max
        )

        # 2. Infection forcing E(t)
        E_t = r0 * (0.4 + 1.6 * m_risk)

        # 3. Latency transition rate sigma(T) based on thermal degree-days
        gdd_daily = max(0.0, t_m - t_base)
        sigma_t = min(1.0, (gdd_daily / max(10.0, latent_incubation_gdd)) * 2.2)

        # 4. Frontier radial expansion velocity
        g_t = float(thermal_response_beta(t_m, t_opt=t_opt, t_min=t_min, t_max=t_max))
        p_factor = float(np.clip(frontier_perimeter_ratio, 0.5, 2.5))
        v_exp = 0.045 * g_t * (0.5 + 0.5 * m_risk) * p_factor

        # 5. Necrotic senescence rate
        gamma_sen = 0.065 * (1.0 + 0.35 * max(0.0, (t_m - 20.0) / 10.0))

        # Transmission pressure
        infectious_pool = max(0.0, I_c + 0.15 * L_c)
        effective_r = E_t * (S_c / k_cap)

        # Differential terms
        dS = - E_t * S_c * (infectious_pool / k_cap)
        dL =   E_t * S_c * (infectious_pool / k_cap) - sigma_t * L_c
        
        # Lesion expansion is capped by remaining uncolonized vegetative space
        vis_curr = min(k_cap, I_c + R_c)
        space_factor = max(0.0, 1.0 - vis_curr / k_cap)
        dI =   sigma_t * L_c + v_exp * I_c * space_factor - gamma_sen * I_c
        dR =   gamma_sen * I_c

        return dS, dL, dI, dR, effective_r, m_risk

    # 4th-order Runge-Kutta numerical integration
    for step in range(n_steps):
        t_n = t_vals[step]
        S_n = S_arr[step]
        L_n = L_arr[step]
        I_n = I_arr[step]
        R_n = R_arr[step]

        # Step evaluation k1
        dS1, dL1, dI1, dR1, r_eff1, risk1 = derivatives(t_n, S_n, L_n, I_n, R_n)
        Reff_arr[step] = r_eff1
        Risk_arr[step] = risk1

        # Step evaluation k2
        S_mid1 = max(0.0, S_n + 0.5 * dt * dS1)
        L_mid1 = max(0.0, L_n + 0.5 * dt * dL1)
        I_mid1 = max(0.0, I_n + 0.5 * dt * dI1)
        R_mid1 = max(0.0, R_n + 0.5 * dt * dR1)
        dS2, dL2, dI2, dR2, _, _ = derivatives(t_n + 0.5 * dt, S_mid1, L_mid1, I_mid1, R_mid1)

        # Step evaluation k3
        S_mid2 = max(0.0, S_n + 0.5 * dt * dS2)
        L_mid2 = max(0.0, L_n + 0.5 * dt * dL2)
        I_mid2 = max(0.0, I_n + 0.5 * dt * dI2)
        R_mid2 = max(0.0, R_n + 0.5 * dt * dR2)
        dS3, dL3, dI3, dR3, _, _ = derivatives(t_n + 0.5 * dt, S_mid2, L_mid2, I_mid2, R_mid2)

        # Step evaluation k4
        S_end = max(0.0, S_n + dt * dS3)
        L_end = max(0.0, L_n + dt * dL3)
        I_end = max(0.0, I_n + dt * dI3)
        R_end = max(0.0, R_n + dt * dR3)
        dS4, dL4, dI4, dR4, _, _ = derivatives(t_n + dt, S_end, L_end, I_end, R_end)

        # Weighted RK4 state updates
        S_next = S_n + (dt / 6.0) * (dS1 + 2.0 * dS2 + 2.0 * dS3 + dS4)
        L_next = L_n + (dt / 6.0) * (dL1 + 2.0 * dL2 + 2.0 * dL3 + dL4)
        I_next = I_n + (dt / 6.0) * (dI1 + 2.0 * dI2 + 2.0 * dI3 + dI4)
        R_next = R_n + (dt / 6.0) * (dR1 + 2.0 * dR2 + 2.0 * dR3 + dR4)

        # Enforce physical mass conservation and non-negativity
        total_mass = S_next + L_next + I_next + R_next
        if total_mass > 0:
            S_next /= total_mass
            L_next /= total_mass
            I_next /= total_mass
            R_next /= total_mass

        # Cap visible severity at carrying capacity
        vis_next = I_next + R_next
        if vis_next > k_cap:
            scale = k_cap / vis_next
            I_next *= scale
            R_next *= scale
            S_next = 1.0 - (L_next + I_next + R_next)

        S_arr[step + 1] = max(0.0, S_next)
        L_arr[step + 1] = max(0.0, L_next)
        I_arr[step + 1] = max(0.0, I_next)
        R_arr[step + 1] = max(0.0, R_next)

    # Final step metrics
    _, _, _, _, r_eff_end, risk_end = derivatives(t_vals[-1], S_arr[-1], L_arr[-1], I_arr[-1], R_arr[-1])
    Reff_arr[-1] = r_eff_end
    Risk_arr[-1] = risk_end

    return t_vals, S_arr, L_arr, I_arr, R_arr, Reff_arr, Risk_arr


def forecast_foliar_seir_trajectory(
    necrotic_pct: float,
    chlorotic_pct: float,
    daily_weather_series: List[Dict[str, float]],
    growth_params: Optional[Dict[str, float]] = None,
    frontier_perimeter_ratio: float = 1.0,
    carrying_capacity_pct: float = 95.0,
    latent_incubation_gdd: float = 90.0,
    t_base: float = 4.0
) -> FoliarSEIRTrajectoryResult:
    """High-level entrypoint producing continuous Foliar-SEIR forecast curves.

    Args:
        necrotic_pct: Initial dead necrotic percentage (Day 0).
        chlorotic_pct: Initial active spreading chlorotic halo percentage (Day 0).
        daily_weather_series: Sequence of daily weather forecast conditions.
        growth_params: Optional biological growth parameters.
        frontier_perimeter_ratio: Perimeter-to-area scaling factor.
        carrying_capacity_pct: Theoretical carrying capacity limit percentage.
        latent_incubation_gdd: Degree-day requirement for latent symptom eruption.
        t_base: Degree-day base temperature.

    Returns:
        FoliarSEIRTrajectoryResult instance.
    """
    k_cap = carrying_capacity_pct / 100.0
    r0 = necrotic_pct / 100.0
    i0 = chlorotic_pct / 100.0

    t_vals, S_arr, L_arr, I_arr, R_arr, Reff_arr, Risk_arr = solve_foliar_seir_rk4(
        initial_necrotic_fraction=r0,
        initial_chlorotic_fraction=i0,
        daily_weather_series=daily_weather_series,
        growth_params=growth_params,
        frontier_perimeter_ratio=frontier_perimeter_ratio,
        carrying_capacity=k_cap,
        latent_incubation_gdd=latent_incubation_gdd,
        t_base=t_base
    )

    vis_arr = (I_arr + R_arr) * 100.0
    canopy_arr = scale_flag_leaf_to_canopy(vis_arr / 100.0) * 100.0

    # Subsample to integer days
    target_days = list(range(int(np.floor(t_vals[-1])) + 1))
    interp_vis = np.interp(target_days, t_vals, vis_arr)
    interp_lat = np.interp(target_days, t_vals, L_arr * 100.0)
    interp_chl = np.interp(target_days, t_vals, I_arr * 100.0)
    interp_nec = np.interp(target_days, t_vals, R_arr * 100.0)
    interp_can = np.interp(target_days, t_vals, canopy_arr)
    interp_reff = np.interp(target_days, t_vals, Reff_arr)
    interp_risk = np.interp(target_days, t_vals, Risk_arr)

    # Compute cumulative degree-days
    cum_gdd = []
    gdd_sum = 0.0
    for d_idx in target_days:
        if d_idx < len(daily_weather_series):
            w = daily_weather_series[d_idx]
            gdd_sum += max(0.0, float(w.get("t_mean", 18.0)) - t_base)
        cum_gdd.append(round(gdd_sum, 1))

    # Plain language trajectory explanation
    day7_vis = float(interp_vis[-1])
    day0_vis = float(interp_vis[0])
    expansion_pct = day7_vis - day0_vis
    peak_risk = float(np.max(interp_risk))
    avg_risk = float(np.mean(interp_risk))

    if peak_risk >= 0.70:
        risk_text = "High atmospheric moisture and rain-splash events create severe infection pressure, accelerating latent spore incubation."
    elif avg_risk <= 0.30:
        risk_text = "Atmospheric vapor pressure deficit and dry foliage suppress new infection cycles, restricting expansion largely to existing lesion margins."
    else:
        risk_text = "Moderate thermal and humidity conditions support steady vegetative lesion expansion across the foliar canopy."

    summary = (
        f"Foliar-SEIR model forecasts visible disease progression from {day0_vis:.1f}% to {day7_vis:.1f}% (+{expansion_pct:.1f}%) "
        f"over 7 days (asymptomatic latent tissue peaks at {float(np.max(interp_lat)):.1f}%). {risk_text}"
    )

    return FoliarSEIRTrajectoryResult(
        horizons_days=[float(d) for d in target_days],
        visible_severity_pct_trajectory=[round(float(s), 2) for s in interp_vis],
        latent_pct_trajectory=[round(float(l), 2) for l in interp_lat],
        chlorotic_pct_trajectory=[round(float(c), 2) for c in interp_chl],
        necrotic_pct_trajectory=[round(float(n), 2) for n in interp_nec],
        canopy_severity_pct_trajectory=[round(float(k), 2) for k in interp_can],
        daily_reproduction_numbers=[round(float(r), 3) for r in interp_reff],
        daily_magarey_risks=[round(float(m), 3) for m in interp_risk],
        cumulative_degree_days=cum_gdd,
        carrying_capacity_pct=carrying_capacity_pct,
        trajectory_summary=summary
    )


# ---------------------------------------------------------------------------
# Backward Compatibility Wrappers
# ---------------------------------------------------------------------------

def solve_logistic_epidemic_ode(
    initial_severity_fraction: float,
    daily_risk_indices: List[float],
    base_growth_rate: float = 0.12,
    carrying_capacity: float = 0.95,
    dt: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy backward-compatible solver wrapper."""
    total_days = max(1.0, float(len(daily_risk_indices)))
    n_steps = int(np.ceil(total_days / dt))
    t_vals = np.linspace(0.0, total_days, n_steps + 1)
    s_vals = np.zeros(n_steps + 1, dtype=np.float64)

    s_curr = float(np.clip(initial_severity_fraction, 0.0, carrying_capacity))
    s_vals[0] = s_curr
    k_cap = float(carrying_capacity)

    def get_r(t_curr: float) -> float:
        day_idx = min(int(np.floor(t_curr)), len(daily_risk_indices) - 1)
        risk = daily_risk_indices[day_idx]
        return base_growth_rate * (0.6 + 1.4 * risk)

    for step in range(n_steps):
        t_n = t_vals[step]
        s_n = s_vals[step]

        def f(t_in: float, s_in: float) -> float:
            if s_in <= 0.0 or s_in >= k_cap:
                return 0.0
            r_val = get_r(t_in)
            return r_val * s_in * (1.0 - s_in / k_cap)

        k1 = f(t_n, s_n)
        k2 = f(t_n + 0.5 * dt, s_n + 0.5 * dt * k1)
        k3 = f(t_n + 0.5 * dt, s_n + 0.5 * dt * k2)
        k4 = f(t_n + dt, s_n + dt * k3)

        s_next = s_n + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        s_vals[step + 1] = np.clip(s_next, 0.0, k_cap)

    return t_vals, s_vals


def forecast_continuous_trajectory(
    initial_severity_pct: float,
    daily_risk_indices: List[float],
    horizon_days: float = 14.0,
    base_growth_rate: float = 0.12,
    carrying_capacity_pct: float = 95.0
) -> ContinuousTrajectoryResult:
    """Legacy backward-compatible continuous trajectory entrypoint."""
    s0_frac = initial_severity_pct / 100.0
    k_frac = carrying_capacity_pct / 100.0

    t_vals, s_vals = solve_logistic_epidemic_ode(
        initial_severity_fraction=s0_frac,
        daily_risk_indices=daily_risk_indices,
        base_growth_rate=base_growth_rate,
        carrying_capacity=k_frac
    )

    canopy_s_vals = scale_flag_leaf_to_canopy(s_vals)
    target_days = list(range(int(np.floor(t_vals[-1])) + 1))
    interp_leaf = np.interp(target_days, t_vals, s_vals * 100.0)
    interp_canopy = np.interp(target_days, t_vals, canopy_s_vals * 100.0)
    cum_risk = float(np.sum(daily_risk_indices))

    return ContinuousTrajectoryResult(
        horizons_days=[float(d) for d in target_days],
        severity_pct_trajectory=[round(float(s), 2) for s in interp_leaf],
        canopy_severity_pct_trajectory=[round(float(c), 2) for c in interp_canopy],
        carrying_capacity_pct=carrying_capacity_pct,
        cumulative_risk_score=round(cum_risk, 3)
    )

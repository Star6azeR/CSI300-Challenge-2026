"""Ranked alpha optimization with Barra exposure and portfolio constraints."""
import numpy as np
from scipy.optimize import linprog


def _dated(dataset, date, name):
    if dataset.status != "available":
        raise ValueError(f"{name} is unavailable: {dataset.reason}")
    frame = dataset.data
    if not frame.date.eq(date).all():
        raise ValueError(f"{name} must belong to the signal date")
    return frame


def _aligned(dataset, date, codes, name):
    frame = _dated(dataset, date, name)
    if frame.code.duplicated().any():
        raise ValueError(f"{name} contains duplicate stocks")
    indexed = frame.set_index("code")
    if not set(codes).issubset(indexed.index):
        raise ValueError(f"{name} does not cover the legal universe")
    return indexed.loc[codes].drop(columns="date")


def optimize_barra(*, config, date, codes, alpha, benchmark, inputs, current_weights):
    if not codes:
        raise ValueError("Barra optimization requires a nonempty legal universe")
    exposures = _aligned(
        inputs.barra_exposures, date, codes, "Barra exposures",
    ).drop(columns="名称", errors="ignore").to_numpy(dtype=float)
    if exposures.shape[1] == 0:
        raise ValueError("Barra exposures must contain at least one factor column")
    if not np.isfinite(exposures).all():
        raise ValueError("Barra exposures must be complete finite numbers")
    count = len(codes)
    b = np.array([benchmark[code] for code in codes], dtype=float)
    # Centered percentile ranks keep the signal invariant to strictly increasing transforms.
    signal = np.array([alpha[code] - 0.5 for code in codes], dtype=float)
    if current_weights.code.duplicated().any():
        raise ValueError("current weights contain duplicate stocks")
    all_current = current_weights.weight.to_numpy(dtype=float)
    if not np.isfinite(all_current).all() or (all_current < 0).any():
        raise ValueError("current cash-equity weights must be finite and nonnegative")
    current_map = dict(zip(current_weights.code, current_weights.weight))
    current = np.array([current_map.get(code, 0.0) for code in codes], dtype=float)
    outside = sum(float(weight) for code, weight in current_map.items() if code not in set(codes))
    if not np.isfinite(current).all() or (current < 0).any() or outside < 0:
        raise ValueError("current cash-equity weights must be finite and nonnegative")
    use_turnover = config.turnover_limit is not None
    size = count * (2 if use_turnover else 1)
    lower = np.zeros(size)
    upper = np.full(size, np.inf)
    upper[:count] = config.single_name_weight_limit or 1.0
    if config.active_weight_limit is not None:
        lower[:count] = np.maximum(lower[:count], b - config.active_weight_limit)
        upper[:count] = np.minimum(upper[:count], b + config.active_weight_limit)
    if (lower > upper).any():
        raise ValueError("portfolio constraints are infeasible: incompatible weight bounds")

    rows, limits = [], []
    equal_rows, equal_values = [], []
    def inequality(row, limit):
        rows.append(row)
        limits.append(limit)

    budget = np.zeros(size)
    budget[:count] = 1
    if config.fully_invested:
        equal_rows.append(budget)
        equal_values.append(1.0)
    else:
        inequality(budget, 1.0)

    def exposure_limits(matrix, limit):
        for exposure in matrix.T:
            row = np.zeros(size)
            row[:count] = exposure
            center = exposure @ b
            if limit == 0:
                equal_rows.append(row)
                equal_values.append(center)
            else:
                inequality(row, center + limit)
                inequality(-row, -center + limit)

    if config.industry_exposure_limit is not None:
        industries = _aligned(inputs.industries, date, codes, "industries")
        labels = industries.industry.tolist()
        if any(not isinstance(label, str) or not label for label in labels):
            raise ValueError("industry labels must be nonempty strings")
        matrix = np.array([[label == industry for industry in sorted(set(labels))]
                           for label in labels], dtype=float)
        exposure_limits(matrix, config.industry_exposure_limit)
    if config.barra_style_exposure_limit is not None:
        exposure_limits(exposures, config.barra_style_exposure_limit)
    if use_turnover:
        # Two-sided asset turnover; forced exits count even when outside today's index.
        remaining = config.turnover_limit - outside
        if remaining < -1e-10:
            raise ValueError("portfolio constraints are infeasible: exits exceed turnover limit")
        for index in range(count):
            row = np.zeros(size)
            row[index], row[count + index] = 1, -1
            inequality(row, current[index])
            row = np.zeros(size)
            row[index], row[count + index] = -1, -1
            inequality(row, -current[index])
        row = np.zeros(size)
        row[count:] = 1
        inequality(row, max(remaining, 0.0))

    matrix = np.array(rows).reshape(-1, size)
    rhs = np.array(limits)
    equality = np.array(equal_rows) if equal_rows else None
    equal_rhs = np.array(equal_values)
    linear_objective = np.zeros(size)
    linear_objective[:count] = -signal
    solution = linprog(
        linear_objective, A_ub=matrix if rows else None, b_ub=rhs if rows else None,
        A_eq=equality, b_eq=equal_rhs if equality is not None else None,
        bounds=list(zip(lower, upper)), method="highs",
        options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
    )
    if not solution.success:
        reason = "portfolio constraints are infeasible" if solution.status == 2 else "Barra optimization failed"
        raise ValueError(f"{reason}: {solution.message}")
    if not np.isfinite(solution.x).all():
        raise ValueError("Barra optimization returned nonfinite weights")
    weights = np.clip(solution.x[:count], 0, 1)
    if weights.sum() > 1:
        weights /= weights.sum()  # Never publish leverage due to solver roundoff.
    verified = np.r_[weights, np.abs(weights - current)] if use_turnover else weights
    tolerance = 1e-8
    if ((verified < lower - tolerance).any() or (verified > upper + tolerance).any()
            or (rows and (matrix @ verified > rhs + tolerance).any())
            or (equality is not None and (np.abs(equality @ verified - equal_rhs) > tolerance).any())):
        raise ValueError("Barra solution violates portfolio constraints")
    return dict(zip(codes, map(float, weights)))

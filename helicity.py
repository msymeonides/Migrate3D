import multiprocessing as mp
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize


OUTPUT_COLUMNS = [
    "Object ID",
    "Category",
    "Helix Fit Score",
    "Mean Curvature",
    "Median Curvature",
    "Mean Absolute Torsion",
    "Median Absolute Torsion",
]


def _empty_metrics(object_id, category):
    result = {column: np.nan for column in OUTPUT_COLUMNS}
    result["Object ID"] = int(object_id)
    result["Category"] = category
    return result


def _unit_vector_from_angles(angles):
    azimuth, elevation = angles
    c = np.cos(elevation)
    return np.array(
        [c * np.cos(azimuth), c * np.sin(azimuth), np.sin(elevation)]
    )


def _angles_from_unit_vector(vector):
    vector = np.asarray(vector, dtype=float)
    vector /= np.linalg.norm(vector)
    return np.array(
        [np.arctan2(vector[1], vector[0]), np.arcsin(np.clip(vector[2], -1, 1))]
    )


def _orthogonal_basis(axis):
    """Return two unit vectors perpendicular to axis."""
    axis = np.asarray(axis, dtype=float)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(axis, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, reference)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    return e1, e2


def _fit_circle_2d(x, y):
    """Algebraic least-squares circle fit."""
    design = np.column_stack((2.0 * x, 2.0 * y, np.ones(len(x))))
    target = x * x + y * y
    cx, cy, constant = np.linalg.lstsq(design, target, rcond=None)[0]
    radius_squared = constant + cx * cx + cy * cy
    if not np.isfinite(radius_squared) or radius_squared <= 0:
        raise ValueError("Degenerate circle fit")
    return cx, cy, np.sqrt(radius_squared)


def _evaluate_axis(positions, axis):
    """Fit a circular helix for one candidate axis and return diagnostics."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    e1, e2 = _orthogonal_basis(axis)

    origin = positions.mean(axis=0)
    centered = positions - origin
    x = centered @ e1
    y = centered @ e2
    z = centered @ axis

    cx, cy, radius = _fit_circle_2d(x, y)
    dx = x - cx
    dy = y - cy
    observed_radius = np.hypot(dx, dy)
    theta = np.unwrap(np.arctan2(dy, dx))

    # A helix has a linear relationship between axial position and angle.
    regression = np.column_stack((theta, np.ones(len(theta))))
    pitch_per_radian, z_intercept = np.linalg.lstsq(
        regression, z, rcond=None
    )[0]
    z_fit = pitch_per_radian * theta + z_intercept

    radial_residual = observed_radius - radius
    axial_residual = z - z_fit
    point_error = np.hypot(radial_residual, axial_residual)
    rmse = np.sqrt(np.mean(point_error * point_error))

    angle_steps = np.diff(theta)
    total_angular_travel = np.sum(np.abs(angle_steps))
    net_angle = abs(theta[-1] - theta[0])
    monotonicity = (
        net_angle / total_angular_travel if total_angular_travel > 0 else 0.0
    )
    turns = net_angle / (2.0 * np.pi)

    return {
        "axis": axis,
        "rmse": float(rmse),
        "radius": float(radius),
        "pitch_per_turn": float(2.0 * np.pi * pitch_per_radian),
        "turns": float(turns),
        "monotonicity": float(np.clip(monotonicity, 0.0, 1.0)),
        "radial_rmse": float(np.sqrt(np.mean(radial_residual**2))),
        "axial_rmse": float(np.sqrt(np.mean(axial_residual**2))),
    }


def fit_helix(positions, minimum_turns=1.0, axis_optimization_iterations=150):
    """Fit one circular helix to an entire ordered 3-D trajectory.

    The score is in [0, 1].  It combines geometric fit quality, consistent
    progression around the axis, and angular coverage.  Thus a short arc or a
    jittery cluster cannot receive a high score solely by fitting a small radius.
    """
    positions = np.asarray(positions, dtype=float)
    if len(positions) < 6 or not np.all(np.isfinite(positions)):
        raise ValueError("At least six finite positions are required")

    centered = positions - positions.mean(axis=0)
    spatial_scale = np.sqrt(np.mean(np.sum(centered * centered, axis=1)))
    if spatial_scale <= np.finfo(float).eps:
        raise ValueError("Track has no spatial extent")

    # PCA supplies three geometrically distinct starting axes.  Refining all
    # three avoids assuming that the helix axis is always the first PC.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    starting_axes = list(vt)
    starting_axes.extend(np.eye(3))

    def objective(angles):
        try:
            fit = _evaluate_axis(positions, _unit_vector_from_angles(angles))
            return fit["rmse"] / spatial_scale
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return 1e6

    best_fit = None
    for starting_axis in starting_axes:
        initial_angles = _angles_from_unit_vector(starting_axis)
        optimization = minimize(
            objective,
            initial_angles,
            method="Powell",
            options={"maxiter": int(axis_optimization_iterations), "ftol": 1e-8},
        )
        try:
            candidate = _evaluate_axis(
                positions, _unit_vector_from_angles(optimization.x)
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
        if best_fit is None or candidate["rmse"] < best_fit["rmse"]:
            best_fit = candidate

    if best_fit is None:
        raise ValueError("No nondegenerate helix fit was found")

    # Normalize residuals to the fitted radius: deviations comparable to the
    # radius itself indicate a poor helix.  The track-scale floor prevents an
    # almost-zero fitted radius from causing numerical instability.
    error_scale = max(best_fit["radius"], 0.05 * spatial_scale)
    normalized_rmse = best_fit["rmse"] / error_scale
    fit_quality = 1.0 / (1.0 + normalized_rmse**2)
    turn_coverage = min(1.0, best_fit["turns"] / max(minimum_turns, 1e-12))

    best_fit["score"] = float(
        np.clip(
            fit_quality * best_fit["monotonicity"] * turn_coverage,
            0.0,
            1.0,
        )
    )
    return best_fit


def compute_spline_geometry(object_data, smoothing_per_point=0.1):
    """Return spline-derived curvature and signed torsion."""
    times = object_data[:, 1].astype(float)
    positions = object_data[:, 2:5].astype(float)
    finite = np.isfinite(times) & np.all(np.isfinite(positions), axis=1)
    times = times[finite]
    positions = positions[finite]

    if len(times) < 4:
        raise ValueError("At least four finite timepoints are required")

    order = np.argsort(times, kind="stable")
    times = times[order]
    positions = positions[order]
    unique_times, first_indices, counts = np.unique(
        times, return_index=True, return_counts=True
    )
    if np.any(counts > 1):
        positions = np.add.reduceat(positions, first_indices, axis=0) / counts[:, None]
        times = unique_times

    if len(times) < 4 or times[-1] <= times[0]:
        raise ValueError("Four unique increasing timepoints are required")

    u = (times - times[0]) / (times[-1] - times[0])
    smoothing = len(times) * float(smoothing_per_point)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        tck, _ = splprep(positions.T, u=u, s=smoothing, k=3)

    r1 = np.asarray(splev(u, tck, der=1)).T
    r2 = np.asarray(splev(u, tck, der=2)).T
    r3 = np.asarray(splev(u, tck, der=3)).T
    cross = np.cross(r1, r2)
    cross_squared = np.einsum("ij,ij->i", cross, cross)
    speed = np.linalg.norm(r1, axis=1)

    curvature = np.full(len(speed), np.nan)
    valid_speed = speed > np.finfo(float).eps * max(1.0, np.nanmax(speed))
    curvature[valid_speed] = (
        np.sqrt(cross_squared[valid_speed]) / speed[valid_speed] ** 3
    )

    torsion = np.full(len(speed), np.nan)
    # Exclude locally near-straight spline regions, where torsion is undefined
    # and numerically explosive.  This is relative to the track's own bending.
    finite_cross = cross_squared[np.isfinite(cross_squared)]
    if finite_cross.size:
        cross_threshold = max(
            np.finfo(float).eps * max(1.0, np.max(finite_cross)),
            1e-8 * np.max(finite_cross),
        )
        valid_torsion = cross_squared > cross_threshold
        torsion[valid_torsion] = (
            np.einsum("ij,ij->i", cross[valid_torsion], r3[valid_torsion])
            / cross_squared[valid_torsion]
        )
    return curvature, torsion


def compute_metrics_single_object(args):
    (
        object_id,
        object_data,
        category,
        min_timepoints,
        smoothing_per_point,
        minimum_turns,
        axis_iterations,
    ) = args

    if len(object_data) < max(min_timepoints, 6):
        return _empty_metrics(object_id, category)

    result = _empty_metrics(object_id, category)
    positions = object_data[:, 2:5].astype(float)

    try:
        helix = fit_helix(
            positions,
            minimum_turns=minimum_turns,
            axis_optimization_iterations=axis_iterations,
        )
        result["Helix Fit Score"] = helix["score"]
    except Exception:
        pass

    try:
        curvature, torsion = compute_spline_geometry(
            object_data, smoothing_per_point=smoothing_per_point
        )
        if np.any(np.isfinite(curvature)):
            result["Mean Curvature"] = np.nanmean(curvature)
            result["Median Curvature"] = np.nanmedian(curvature)
        absolute_torsion = np.abs(torsion)
        if np.any(np.isfinite(absolute_torsion)):
            result["Mean Absolute Torsion"] = np.nanmean(absolute_torsion)
            result["Median Absolute Torsion"] = np.nanmedian(absolute_torsion)
    except Exception:
        pass

    return result


def compute_metrics_batch(batch_args):
    return [compute_metrics_single_object(args) for args in batch_args]


def compute_helicity_analysis(arr_segments, arr_cats, parameters):
    min_timepoints = int(parameters["moving"])
    smoothing_per_point = float(parameters.get("helicity_smoothing", 0.1))
    minimum_turns = float(parameters.get("helix_minimum_turns", 1.0))
    axis_iterations = int(parameters.get("helix_axis_iterations", 150))

    if arr_segments.size == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    cat_dict = (
        dict(zip(arr_cats[:, 0].astype(int), arr_cats[:, 1].astype(str)))
        if arr_cats.size > 0
        else {}
    )
    sorted_indices = np.lexsort(
        (arr_segments[:, 1].astype(float), arr_segments[:, 0].astype(int))
    )
    sorted_segments = arr_segments[sorted_indices]
    object_ids = sorted_segments[:, 0].astype(int)
    groups = np.split(sorted_segments, np.flatnonzero(np.diff(object_ids)) + 1)

    object_args = []
    for object_data in groups:
        obj_id = int(object_data[0, 0])
        object_args.append(
            (
                obj_id,
                object_data,
                cat_dict.get(obj_id, "0"),
                min_timepoints,
                smoothing_per_point,
                minimum_turns,
                axis_iterations,
            )
        )

    max_workers = max(1, min(61, mp.cpu_count() - 2, len(object_args)))
    batch_size = max(10, len(object_args) // (max_workers * 2))
    batches = [
        object_args[i : i + batch_size]
        for i in range(0, len(object_args), batch_size)
    ]

    if max_workers == 1:
        batch_results = [compute_metrics_batch(batch) for batch in batches]
    else:
        with mp.Pool(processes=max_workers) as pool:
            batch_results = pool.map(compute_metrics_batch, batches)

    results = [item for batch in batch_results for item in batch]
    return (
        pd.DataFrame(results, columns=OUTPUT_COLUMNS)
        .sort_values("Object ID")
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    mp.freeze_support()

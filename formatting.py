import numpy as np

def multi_tracking(arr_segments):
    if arr_segments.size == 0:
        return arr_segments

    obj = arr_segments[:, 0].astype(int)
    t = arr_segments[:, 1].astype(float)
    coords = arr_segments[:, 2:5].astype(float)
    keys = np.stack((obj, t), axis=1)
    uniq_keys, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((uniq_keys.shape[0], 3), dtype=float)
    np.add.at(sums, inv, coords)
    avgs = sums / counts[:, None]

    out = np.concatenate((uniq_keys, avgs), axis=1)
    return out


def _interpolate_single_track(times, coords, dt):
    order = np.argsort(times, kind='mergesort')
    ts = times[order]
    xyz = coords[order]

    uniq, idx_first, counts = np.unique(ts, return_index=True, return_counts=True)
    idx_last = idx_first + counts - 1
    ts_u = ts[idx_last]
    xyz_u = xyz[idx_last]

    if ts_u.size == 0:
        return np.empty((0, 4), dtype=float)

    t_min = float(ts_u[0])
    t_max = float(ts_u[-1])

    grid = np.arange(t_min, t_max + dt * 0.5, dt)
    if grid.size == 0:
        return np.empty((0, 4), dtype=float)

    xg = np.interp(grid, ts_u, xyz_u[:, 0])
    yg = np.interp(grid, ts_u, xyz_u[:, 1])
    zg = np.interp(grid, ts_u, xyz_u[:, 2])

    return np.column_stack((grid, xg, yg, zg))


def interpolate_lazy(arr_segments, timelapse_interval):
    if arr_segments.size == 0:
        return arr_segments

    obj = arr_segments[:, 0].astype(int)
    t = arr_segments[:, 1].astype(float)
    coords = arr_segments[:, 2:5].astype(float)
    order = np.lexsort((t, obj))
    obj_s = obj[order]
    t_s = t[order]
    xyz_s = coords[order]
    uniq_obj, idx_start, counts = np.unique(obj_s, return_index=True, return_counts=True)

    pieces = []
    for i, start in enumerate(idx_start):
        cnt = counts[i]
        o = uniq_obj[i]
        times_i = t_s[start:start + cnt]
        xyz_i = xyz_s[start:start + cnt]

        interp_track = _interpolate_single_track(times_i, xyz_i, float(timelapse_interval))
        if interp_track.size == 0:
            continue
        obj_col = np.full((interp_track.shape[0], 1), o)
        pieces.append(np.concatenate((obj_col, interp_track.astype(float)), axis=1))

    if not pieces:
        return np.empty((0, arr_segments.shape[1]))

    out = np.vstack(pieces)
    return out


def remove_tracks_with_gaps(arr_segments, unique_objects, timelapse_interval):
    if arr_segments.size == 0:
        return np.empty((0, arr_segments.shape[1]))

    obj = arr_segments[:, 0].astype(int)
    t = arr_segments[:, 1].astype(float)

    order = np.lexsort((t, obj))
    arr_sorted = arr_segments[order]
    obj_s = obj[order]

    uniq_obj, idx_start, counts = np.unique(obj_s, return_index=True, return_counts=True)

    keep_slices = []
    dt = float(timelapse_interval)
    for i, start in enumerate(idx_start):
        cnt = counts[i]
        seg = arr_sorted[start:start + cnt]
        times = np.array(seg[:, 1], dtype=float)
        diffs = np.diff(times)
        if diffs.size == 0 or np.allclose(diffs, dt):
            keep_slices.append(seg)

    if keep_slices:
        return np.vstack(keep_slices)
    else:
        return np.empty((0, arr_segments.shape[1]))

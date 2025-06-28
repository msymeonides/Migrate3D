import multiprocessing as mp
import numpy as np
import pandas as pd

window_size = 4     # Tunable parameter for bootstrapping window size
r2_thresh = 0.995   # Tunable parameter for R-squared minimum threshold
n_boot = 1000       # Number of bootstrap iterations

def find_best_linear_window(x, y):
    n = len(x)
    last_good_end = None
    for end in range(window_size, n + 1):
        x_win, y_win = x[0:end], y[0:end]
        coeffs = np.polyfit(x_win, y_win, 1)
        y_fit = np.polyval(coeffs, x_win)
        ss_res = np.sum((y_win - y_fit) ** 2)
        ss_tot = np.sum((y_win - np.mean(y_win)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        if r2 >= r2_thresh:
            last_good_end = end
        else:
            break
    if last_good_end is not None:
        return 0, last_good_end, None
    else:
        return 0, window_size, 0

def bootstrap_worker(task):
    cat_df, object_ids, n_iter = task
    obj_ids_arr = cat_df['Object ID'].to_numpy()
    log_tau_arr = cat_df['log_tau'].to_numpy()
    log_msd_arr = cat_df['log_msd'].to_numpy()
    id_to_indices = {oid: np.where(obj_ids_arr == oid)[0] for oid in object_ids}
    slopes = []
    n_obj = len(object_ids)
    for _ in range(n_iter):
        sampled_ids = np.random.choice(object_ids, size=n_obj, replace=True)
        indices = np.concatenate([id_to_indices[oid] for oid in sampled_ids])
        sample_log_tau = log_tau_arr[indices]
        sample_log_msd = log_msd_arr[indices]
        if len(sample_log_tau) < window_size:
            continue
        unique_tau, inv = np.unique(sample_log_tau, return_inverse=True)
        mean_log_msd = np.zeros_like(unique_tau)
        for i, tau in enumerate(unique_tau):
            mean_log_msd[i] = sample_log_msd[inv == i].mean()
        x_bs = unique_tau
        y_bs = mean_log_msd
        if len(x_bs) < window_size:
            continue
        s, e, _ = find_best_linear_window(x_bs, y_bs)
        x_fit_bs = x_bs[s:e]
        y_fit_bs = y_bs[s:e]
        if len(x_fit_bs) < 2:
            continue
        slope_bs, _ = np.polyfit(x_fit_bs, y_fit_bs, 1)
        slopes.append(slope_bs)
    return slopes

def main(df_msd):
    df = df_msd.copy()
    id_cols = ['Object ID', 'Category']
    tau_cols = [col for col in df.columns if col not in id_cols]
    df_long = df.melt(id_vars=id_cols, value_vars=tau_cols, var_name='tau', value_name='msd')
    df_long['tau'] = df_long['tau'].astype(float)
    df_long = df_long[(df_long['tau'] > 0) & (df_long['msd'] > 0)].copy()
    df_long['log_tau'] = np.log10(df_long['tau'])
    df_long['log_msd'] = np.log10(df_long['msd'])

    df_long['Category'] = df_long['Category'].astype(str)
    categories = sorted([str(cat) for cat in df_long['Category'].unique() if pd.notnull(cat)])
    n_workers = len(categories)

    fit_stats = {}

    tasks = []
    for category in categories:
        cat_df = df_long[df_long['Category'] == category]
        object_ids = cat_df['Object ID'].unique()
        tasks.append((cat_df, object_ids, n_boot))

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(bootstrap_worker, tasks)

    for idx, category in enumerate(categories):
        cat_df = tasks[idx][0]
        mean_log = cat_df.groupby('log_tau')['log_msd'].mean().reset_index()
        x = mean_log['log_tau'].values
        y = mean_log['log_msd'].values

        start, end, _ = find_best_linear_window(x, y)
        x_fit = x[start:end]
        y_fit = y[start:end]
        coeffs = np.polyfit(x_fit, y_fit, 1)
        slope, intercept = coeffs

        y_fit_pred = np.polyval(coeffs, x_fit)
        ss_res = np.sum((y_fit - y_fit_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        slopes = results[idx]
        if slopes:
            ci_low, ci_high = np.percentile(slopes, [5, 95])
        else:
            ci_low, ci_high = np.nan, np.nan

        max_tau = 10 ** x_fit[-1]
        fit_stats[category] = [slope, ci_low, ci_high, r2, max_tau]

    df_msd_loglogfits = pd.DataFrame(
        {cat: fit_stats[cat] for cat in categories},
        index=['Slope', 'Lower 95% CI', 'Upper 95% CI', 'Fit R2', 'Fit Max. Tau']
    )

    return df_msd_loglogfits

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)

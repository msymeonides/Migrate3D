import multiprocessing as mp
import numpy as np
import pandas as pd

window_size = 4     # Tunable parameter for bootstrapping window size
r2_thresh = 0.995   # Tunable parameter for R-squared minimum threshold
n_boot = 500        # Number of bootstrap iterations (keep this between 100 and 1000)

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
    slopes = []
    for _ in range(n_iter):
        sampled_ids = np.random.choice(object_ids, size=len(object_ids), replace=True)
        sample_df = pd.concat([cat_df[cat_df['Object ID'] == oid] for oid in sampled_ids])
        mean_log_bs = sample_df.groupby('log_tau')['log_msd'].mean().reset_index()
        x_bs = mean_log_bs['log_tau'].values
        y_bs = mean_log_bs['log_msd'].values
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
    max_processes = max(1, min(61, mp.cpu_count() - 1))
    n_workers = max_processes


    df = df_msd.copy()
    id_cols = ['Object ID', 'Category']
    tau_cols = [col for col in df.columns if col not in id_cols]
    df_long = df.melt(id_vars=id_cols, value_vars=tau_cols, var_name='tau', value_name='msd')
    df_long['tau'] = df_long['tau'].astype(float)
    df_long = df_long[(df_long['tau'] > 0) & (df_long['msd'] > 0)].copy()
    df_long['log_tau'] = np.log10(df_long['tau'])
    df_long['log_msd'] = np.log10(df_long['msd'])

    categories = sorted([int(cat) for cat in df_long['Category'].unique() if pd.notnull(cat)])

    fit_stats = {}

    for category in categories:
        cat_df = df_long[df_long['Category'] == category]
        mean_log = cat_df.groupby('log_tau')['log_msd'].mean().reset_index()
        x = mean_log['log_tau'].values
        y = mean_log['log_msd'].values

        start, end, _ = find_best_linear_window(x, y)
        x_fit = x[start:end]
        y_fit = y[start:end]
        coeffs = np.polyfit(x_fit, y_fit, 1)
        slope, intercept = coeffs
        y_fit_line = np.polyval(coeffs, x)

        y_fit_pred = np.polyval(coeffs, x_fit)
        ss_res = np.sum((y_fit - y_fit_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        object_ids = cat_df['Object ID'].unique()
        boot_per_worker = n_boot // n_workers
        tasks = [
            (cat_df, object_ids, boot_per_worker)
            for _ in range(n_workers)
        ]
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(bootstrap_worker, tasks)
        slopes = [s for sublist in results for s in sublist]
        if slopes:
            ci_low, ci_high = np.percentile(slopes, [2.5, 97.5])
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
